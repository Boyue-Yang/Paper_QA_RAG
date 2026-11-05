"""Derive the HTML corpus manifest from data/corpus_80/manifest.json.

Pins each paper to its selected arXiv version, stores a local HTML snapshot with a SHA-256,
and drops papers whose arXiv HTML has no body text (the author embedded a PDF in the LaTeX).
The original corpus_80 manifest and PDFs are left untouched.

    python scripts/prepare_html_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# HTML pages are "See pages 1-last of <name>.pdf" stubs with no extractable text.
EXCLUDED = {
    'arxiv_1412.6980': 'arXiv HTML for v9 contains no body text (PDF embedded in LaTeX).',
    'arxiv_1707.06347': 'arXiv HTML for v2 contains no body text (PDF embedded in LaTeX).',
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': 'paper-qa-corpus-snapshot (boyueyang@utexas.edu)'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/corpus_80/manifest.json')
    parser.add_argument('--target', type=Path, default=ROOT / 'data/corpus_78')
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    (args.target / 'html').mkdir(parents=True, exist_ok=True)
    documents = []
    for document in source['documents']:
        if document['paper_id'] in EXCLUDED:
            continue
        version = document['download']['selected_version']
        url = f"https://arxiv.org/html/{document['arxiv_id']}{version}"
        path = args.target / 'html' / f"{document['paper_id']}{version}.html"
        if not path.exists():
            path.write_bytes(fetch(url))
            time.sleep(1)
        data = path.read_bytes()
        if b'ltx_document' not in data or len(re.findall(rb'class="ltx_para', data)) < 5:
            raise ValueError(f"{document['paper_id']}: HTML snapshot has no usable body text.")
        entry = dict(document)
        entry['html'] = {'url': url, 'local_path': path.relative_to(ROOT).as_posix(),
                         'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                         'selected_version': version}
        documents.append(entry)
    manifest = {'collection_id': source['collection_id'].replace('_80_', '_78_'),
        'status': 'html_corpus_derived_from_corpus_80', 'derived_from': 'data/corpus_80/manifest.json',
        'excluded_documents': EXCLUDED, 'document_count': len(documents),
        'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'documents': documents}
    (args.target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(documents)} papers -> {args.target / "manifest.json"}')


if __name__ == '__main__':
    main()
