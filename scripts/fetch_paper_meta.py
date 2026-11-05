"""Fill in "Author et al. (year)" citations for papers whose manifest citation is empty.

  python -m scripts.fetch_paper_meta

Writes data/corpus_78/paper_meta.json (paper_id -> authors, year, citation) from one arXiv API query.
The frozen chunk files are not touched; the retriever reads this file when a chunk's citation is empty.
"""
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'corpus_78' / 'manifest.json'
OUT = ROOT / 'data' / 'corpus_78' / 'paper_meta.json'
ATOM = '{http://www.w3.org/2005/Atom}'


def surname(name):
    return name.split()[-1]


def make_citation(authors, year):
    names = [surname(a) for a in authors]
    who = names[0] if len(names) == 1 else f'{names[0]} & {names[1]}' if len(names) == 2 else f'{names[0]} et al.'
    return f'{who} ({year})'


def fallback_citation(arxiv_id):
    """Used when the API is unavailable: no author names, but the year is encoded in a new-style arXiv id."""
    return f'arXiv:{arxiv_id} (20{arxiv_id[:2]})'


def fetch(arxiv_ids):
    url = 'https://export.arxiv.org/api/query?max_results=%d&id_list=%s' % (len(arxiv_ids), ','.join(arxiv_ids))
    request = urllib.request.Request(url, headers={'User-Agent': 'paper-qa-rag-course-project (boyueyang@utexas.edu)'})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    found = {}
    for entry in root.findall(ATOM + 'entry'):
        arxiv_id = entry.findtext(ATOM + 'id').rsplit('/abs/', 1)[-1].rsplit('v', 1)[0]
        authors = [a.findtext(ATOM + 'name') for a in entry.findall(ATOM + 'author')]
        year = int(entry.findtext(ATOM + 'published')[:4])
        if authors:
            found[arxiv_id] = {'authors': authors, 'year': year}
    return found


def main():
    documents = json.loads(MANIFEST.read_text(encoding='utf-8'))['documents']
    meta = {}
    for d in documents:
        if (d.get('citation') or '').strip():
            meta[d['paper_id']] = {'citation': d['citation'], 'source': 'manifest'}
    todo = [d for d in documents if d['paper_id'] not in meta]
    found = {}
    for attempt in range(2):  # arXiv asks for polite use; one retry after a pause, then fall back
        try:
            found = fetch([d['arxiv_id'] for d in todo])
            break
        except urllib.error.HTTPError as error:
            print(f'arXiv API returned HTTP {error.code}' + (', retrying once in 30s' if attempt == 0 else ''))
            if attempt == 0:
                time.sleep(30)
    for d in todo:
        info = found.get(d['arxiv_id'])
        if info:
            meta[d['paper_id']] = {'citation': make_citation(info['authors'], info['year']),
                                   'authors': info['authors'], 'year': info['year'], 'source': 'arxiv_api'}
        else:
            meta[d['paper_id']] = {'citation': fallback_citation(d['arxiv_id']), 'source': 'fallback_arxiv_id'}
    fallbacks = [k for k, v in meta.items() if v['source'] == 'fallback_arxiv_id']
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(meta)} citations written to {OUT}; from arXiv API: {len(todo) - len(fallbacks)}; '
          f'id-only fallback (rerun later to upgrade): {len(fallbacks)}')
    return 1 if fallbacks else 0


if __name__ == '__main__':
    sys.exit(main())
