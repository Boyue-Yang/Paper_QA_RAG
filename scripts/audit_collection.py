"""Verify the local collection against its manifest, without network requests."""
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from collect_papers import normalize

ROOT = Path(__file__).resolve().parents[1]
catalog_path = ROOT / 'data' / 'paper_catalog.json'
catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
errors = []
ids = [r['paper_id'] for r in catalog['documents']]
keys = [r['dedupe_key'] for r in catalog['documents']]
assert len(ids) == len(set(ids)), 'Duplicate document IDs'
assert len(keys) == len(set(keys)), 'Duplicate canonical keys'
source_data = json.loads((ROOT / 'data/sources/cs395t-f26-readings.json').read_text(encoding='utf-8'))
source_count = sum(len(items) for groups in source_data.values() for items in groups.values())
assert sum(len(r['sources']) for r in catalog['documents']) == source_count, 'Source occurrences lost'
total_bytes = 0
for record in catalog['documents']:
    info = record['download']
    if info['status'] not in ('downloaded_pdf', 'downloaded_html'):
        continue
    path = (ROOT / info['local_path']).resolve()
    if not path.is_relative_to(ROOT / 'data'):
        errors.append({'paper_id': record['paper_id'], 'error': 'Path escapes data directory'})
        continue
    data = path.read_bytes()
    total_bytes += len(data)
    if hashlib.sha256(data).hexdigest() != info['sha256']:
        errors.append({'paper_id': record['paper_id'], 'error': 'SHA-256 mismatch'})
    if info['status'] == 'downloaded_pdf':
        pdf = PdfReader(path)
        if len(pdf.pages) != info['page_count']:
            errors.append({'paper_id': record['paper_id'], 'error': 'Page count mismatch'})
        text = pdf.pages[0].extract_text() or ''
        info['first_page_extracted_characters'] = len(text.strip())
        if len(text.strip()) < 100:
            sample = [text] + [pdf.pages[i].extract_text() or '' for i in range(1, min(3, len(pdf.pages)))]
            info['text_screening'] = {'sampled_page_count': len(sample), 'characters_per_page': [len(t.strip()) for t in sample]}
            if max(len(t.strip()) for t in sample) < 100:
                info['identity_review'] = 'needs_ocr_review'
            else:
                words = [t for t in normalize(record['title']).split() if len(t) > 3]
                found = normalize(' '.join(sample))
                coverage = sum(t in found for t in words) / len(words) if words else 0
                info['identity_review'] = 'title_tokens_found_after_cover' if coverage >= .6 else 'check_title'

catalog['source_snapshots'] = [
    {'local_path': 'data/sources/' + filename,
     'sha256': hashlib.sha256((ROOT / 'data/sources' / filename).read_bytes()).hexdigest()}
    for filename in ('cs395t-f26.html', 'cs395t-f26-readings.json')
]
catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
report = {
    'verified_at_utc': datetime.now(timezone.utc).isoformat(),
    'unique_documents': len(ids), 'source_occurrences': source_count,
    'download_status': dict(collections.Counter(r['download']['status'] for r in catalog['documents'])),
    'identity_review': dict(collections.Counter(r['download'].get('identity_review', 'not_applicable') for r in catalog['documents'])),
    'total_download_bytes': total_bytes, 'errors': errors,
    'needs_attention': [
        {'paper_id': r['paper_id'], 'title': r['title'],
         'status': r['download']['status'], 'identity_review': r['download'].get('identity_review'),
         'error': r['download'].get('error')}
        for r in catalog['documents']
        if r['download']['status'] not in ('downloaded_pdf', 'downloaded_html')
        or r['download'].get('identity_review') in ('needs_ocr_review', 'check_title')
    ],
    'limitations': [
        'User Google spreadsheet not imported because authentication is required.',
        'All entries remain candidates, not an approved evaluation corpus.',
        'Title-token checks are a heuristic, not bibliographic identity certification.',
        'Text screening covers the first page, expanded to at most three pages for sparse covers; full parsing remains to be done.',
        'Unversioned sources are snapshotted with SHA-256; arXiv revision numbers are not inferred.',
    ],
}
(ROOT / 'data/collection_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=True, indent=2))
if errors:
    raise SystemExit(1)
