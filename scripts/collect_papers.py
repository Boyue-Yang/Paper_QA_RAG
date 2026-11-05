"""Build a provenance-preserving candidate catalog and download source documents.

Uses only the standard library plus pypdf for PDF validation. No chunking is done.
Run with --build and/or --download. The user spreadsheet is not imported yet.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import logging
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'sources'
CATALOG = ROOT / 'data' / 'paper_catalog.json'
COURSE_URL = 'https://noahgol.github.io/teaching/cs395t-f26/'
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1-obHFhggS7_C5UZ5SnR8Lrv92oEFs2HG/edit?gid=2016396705'
USER_AGENT = 'PaperQAResearchCollection/0.1 (personal academic document collection)'
logging.getLogger('pypdf').setLevel(logging.ERROR)


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def arxiv_id(url):
    match = re.search(r'arxiv\.org/(?:abs|pdf|html)/([^?#]+)', url)
    return match.group(1).removesuffix('.pdf') if match else None


def identity(title, url, citation=''):
    aid = arxiv_id(url)
    if aid:
        return 'arxiv:' + re.sub(r'v\d+$', '', aid)
    if url:
        parsed = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip('/'), parsed.query, ''))
    return 'title:' + normalize(title) + ':' + normalize(citation)


class ScheduleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.heading = False
        self.heading_parts = []
        self.last_heading = ''
        self.topics = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'h3':
            self.heading = True
            self.heading_parts = []
        if 'data-reading-key' in attrs:
            self.topics[attrs['data-reading-key']] = self.last_heading

    def handle_data(self, text):
        if self.heading:
            self.heading_parts.append(text)

    def handle_endtag(self, tag):
        if tag == 'h3' and self.heading:
            self.last_heading = ''.join(self.heading_parts).strip()
            self.heading = False


def scope_for(date):
    if date == '2026-08-24':
        return 'deep_learning_background'
    if date in ('2026-09-14', '2026-09-16', '2026-09-21', '2026-09-30'):
        return 'optimization_and_rl_foundations'
    if date == '2026-10-19':
        return 'math_background'
    if '2026-10-21' <= date <= '2026-11-04':
        return 'diffusion_and_flow_models'
    if '2026-11-09' <= date <= '2026-11-16':
        return 'safety_and_alignment'
    return 'llm_core'


def document_type(url, title):
    if 'deeplearningbook.org' in url or 'link.springer.com/book/' in url:
        return 'book'
    if any(host in url for host in ('kellerjordan.github.io', 'jax-ml.github.io', 'anthropic.com')):
        return 'web_article'
    if 'technical report' in title.lower() or title.startswith(('DeepSeek-', 'Kimi ', 'Olmo ', 'MAI-')):
        return 'technical_report'
    if 'Convex Optimization: Algorithms and Complexity' in title:
        return 'monograph'
    return 'research_paper'


def course_entries():
    parser = ScheduleParser()
    parser.feed((SOURCE / 'cs395t-f26.html').read_text(encoding='utf-8'))
    readings = json.loads((SOURCE / 'cs395t-f26-readings.json').read_text(encoding='utf-8'))
    result = []
    for date, groups in readings.items():
        for group, items in groups.items():
            for index, item in enumerate(items, 1):
                result.append({**item, 'source': {
                    'source_id': 'cs395t_f26', 'source_url': COURSE_URL,
                    'reading_list_url': urllib.parse.urljoin(COURSE_URL, 'readings.json'),
                    'lecture_date': date, 'lecture_topic': parser.topics.get(date, ''),
                    'reading_group': group, 'item_in_group': index,
                    'scope_bucket': scope_for(date),
                }})
    return result


def build():
    previous = {}
    if CATALOG.exists():
        previous = {r['dedupe_key']: r for r in json.loads(CATALOG.read_text(encoding='utf-8'))['documents']}
    entries = course_entries()
    records = {}
    for item in entries:
        key = identity(item['title'], item['url'], item.get('citation', ''))
        if key not in records:
            aid = arxiv_id(item['url'])
            base_id = 'arxiv_' + re.sub(r'v\d+$', '', aid).replace('/', '_') if aid else 'doc_' + hashlib.sha256(key.encode()).hexdigest()[:12]
            year = re.search(r'\((\d{4})\)', item.get('citation', ''))
            kind = document_type(item['url'], item['title'])
            records[key] = {
                'paper_id': base_id, 'dedupe_key': key,
                'title': item['title'].rstrip('.'), 'citation': item.get('citation', ''),
                'year_as_listed': int(year.group(1)) if year else None,
                'document_type': kind, 'arxiv_id': aid,
                'source_url': item['url'],
                'download_url': 'https://arxiv.org/pdf/' + aid if aid else item['url'],
                'selection_status': 'candidate',
                'selection_note': 'Needs scope review before corpus approval.',
                'sources': [], 'download': previous.get(key, {}).get('download', {'status': 'pending'}),
            }
        records[key]['sources'].append(item['source'])
    catalog = {
        'schema_version': '1.0', 'catalog_version': 'candidate_2026-10-23_v1',
        'updated_at_utc': now(), 'status': 'partial_pending_user_sheet',
        'source_status': [
            {'source_id': 'cs395t_f26', 'url': COURSE_URL, 'status': 'imported', 'raw_entries': len(entries)},
            {'source_id': 'user_sheet', 'url': SHEET_URL, 'status': 'blocked_authentication', 'raw_entries': None},
        ],
        'documents': list(records.values()),
    }
    save(catalog)
    return catalog


def save(catalog):
    catalog['updated_at_utc'] = now()
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {
        'catalog_version': catalog['catalog_version'], 'status': catalog['status'],
        'unique_documents': len(catalog['documents']),
        'document_types': dict(collections.Counter(r['document_type'] for r in catalog['documents'])),
        'scope_buckets': dict(collections.Counter(r['sources'][0]['scope_bucket'] for r in catalog['documents'])),
        'download_status': dict(collections.Counter(r['download']['status'] for r in catalog['documents'])),
        'source_status': catalog['source_status'],
    }
    (ROOT / 'data' / 'collection_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


arxiv_lock = threading.Lock()
arxiv_next = 0.0
arxiv_blocked = threading.Event()


def download_one(record):
    from pypdf import PdfReader
    if record['download'].get('status') in ('downloaded_pdf', 'downloaded_html'):
        existing = ROOT / record['download']['local_path']
        if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() == record['download']['sha256']:
            return record['download']
    if record['document_type'] == 'book':
        return {'status': 'manual_review_book', 'error': 'Book/TOC retained as a candidate; not treated as a single paper PDF.'}
    url = record['download_url']
    is_arxiv = bool(record.get('arxiv_id'))
    if is_arxiv:
        global arxiv_next
        with arxiv_lock:
            if arxiv_blocked.is_set():
                return {'status': 'rate_limited', 'error': 'arXiv rate limit encountered earlier in this run; retry later.'}
            wait = max(0, arxiv_next - time.monotonic())
            if wait:
                time.sleep(wait)
            arxiv_next = time.monotonic() + 3.0
    try:
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/pdf,text/html;q=0.8'})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(80 * 1024 * 1024 + 1)
            if len(data) > 80 * 1024 * 1024:
                raise ValueError('File exceeds 80 MiB limit; manual review required.')
            final_url = response.url
            content_type = response.headers.get('Content-Type', '')
        result = {'requested_url': url, 'resolved_url': final_url, 'content_type': content_type,
                  'downloaded_at_utc': now(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        if data.lstrip().startswith(b'%PDF-'):
            reader = PdfReader(io.BytesIO(data))
            result['page_count'] = len(reader.pages)
            if not result['page_count']:
                raise ValueError('PDF contains zero pages.')
            result['pdf_metadata_title'] = str((reader.metadata or {}).get('/Title') or '')
            first_page = reader.pages[0].extract_text() or ''
            result['first_page_extracted_characters'] = len(first_page.strip())
            tokens = [t for t in normalize(record['title']).split() if len(t) > 3]
            page_norm = normalize(first_page)
            matched = sum(t in page_norm for t in tokens)
            result['title_token_coverage_first_page'] = round(matched / len(tokens), 3) if tokens else None
            result['identity_review'] = ('needs_ocr_review' if len(first_page.strip()) < 100
                                         else 'check_title' if tokens and matched / len(tokens) < .6
                                         else 'title_tokens_found')
            ext, status = '.pdf', 'downloaded_pdf'
            folder = ROOT / 'data' / 'papers'
        else:
            if is_arxiv or '.pdf' in urllib.parse.urlsplit(url).path.lower():
                raise ValueError('Expected PDF but received non-PDF content; not saved as a paper.')
            if 'html' not in content_type.lower() or len(data) < 1000:
                raise ValueError('Response is neither a valid PDF nor a usable HTML article.')
            lowered = data[:20000].lower()
            if any(marker in lowered for marker in (b'just a moment...', b'cf-chl-', b'captcha', b'access denied')):
                raise ValueError('HTML response appears to be a challenge/error page.')
            ext, status = '.html', 'downloaded_html'
            folder = ROOT / 'data' / 'web_articles'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (record['paper_id'] + ext)
        temp = path.with_suffix(ext + '.part')
        temp.write_bytes(data)
        temp.replace(path)
        result.update(status=status, local_path=path.relative_to(ROOT).as_posix())
        # Unversioned arXiv URLs are frozen by content hash, not a guessed revision number.
        result['snapshot_policy'] = 'content_sha256'
        return result
    except urllib.error.HTTPError as exc:
        if is_arxiv and exc.code == 429:
            arxiv_blocked.set()
        return {'status': 'rate_limited' if exc.code == 429 else 'failed', 'error': 'HTTP ' + str(exc.code), 'attempted_at_utc': now()}
    except Exception as exc:
        return {'status': 'failed', 'error': str(exc)[:500], 'attempted_at_utc': now()}


def download(catalog, limit=None):
    candidates = catalog['documents'][:limit] if limit else catalog['documents']
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_one, record): record for record in candidates}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = futures[future]
            record['download'] = future.result()
            save(catalog)
            print(json.dumps({'completed': index, 'total': len(candidates), 'paper_id': record['paper_id'], 'status': record['download']['status'], 'error': record['download'].get('error')}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    catalog = build() if args.build else json.loads(CATALOG.read_text(encoding='utf-8'))
    if args.download:
        download(catalog, args.limit)
    print((ROOT / 'data' / 'collection_summary.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
