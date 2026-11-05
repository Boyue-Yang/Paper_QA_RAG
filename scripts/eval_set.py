"""Tools for building the evaluation set against the frozen chunk run.

  python -m scripts.eval_set papers
  python -m scripts.eval_set grep "label smoothing" [--paper arxiv_1706.03762]
  python -m scripts.eval_set align data/eval/questions.jsonl [--out data/eval/questions.aligned.jsonl]

`align` checks every question and, for each evidence quote, finds it verbatim in the paper text
and records which chunk ids contain it. Authors write quotes; this script writes the chunk ids.
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / 'data' / 'processed' / 'corpus_78'
QUESTION_TYPES = {'single_fact', 'multi_paper', 'comparison', 'unanswerable'}
SPLITS = {'dev', 'test'}
MIN_QUOTE_CHARS = 40
MAX_QUOTE_CHARS = 600

_PUNCT = {'‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-', '−': '-'}


def run_id():
    return json.loads((RUN_ROOT / 'latest_run.json').read_text(encoding='utf-8'))['run_id']


def load_chunks():
    path = RUN_ROOT / 'chunks' / run_id() / 'chunks.jsonl'
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_papers():
    manifest = json.loads((ROOT / 'data' / 'corpus_78' / 'manifest.json').read_text(encoding='utf-8'))
    return {d['paper_id']: d for d in manifest['documents']}


def normalize(text):
    """NFKC, straight quotes and dashes, collapsed whitespace; returns text and a map to original indices."""
    out, index = [], []
    pending_space = False
    for i, ch in enumerate(text):
        for piece in unicodedata.normalize('NFKC', _PUNCT.get(ch, ch)):
            if piece.isspace():
                pending_space = bool(out)
                continue
            if pending_space:
                out.append(' ')
                index.append(i)
                pending_space = False
            out.append(_PUNCT.get(piece, piece).lower())
            index.append(i)
    return ''.join(out), index


def section_texts(chunks):
    """Rebuild each section's text from its chunks (they carry section-relative char offsets)."""
    sections = {}
    for c in chunks:
        key = c['section_id']
        entry = sections.setdefault(key, {'paper_id': c['paper_id'], 'title': c['section_title'],
                                           'path': c['section_path'], 'merged': c['merged_section_titles'],
                                           'chunks': [], 'length': 0})
        entry['chunks'].append(c)
        entry['length'] = max(entry['length'], c['char_end'])
    for entry in sections.values():
        buffer = [' '] * entry['length']
        for c in entry['chunks']:
            buffer[c['char_start']:c['char_end']] = c['text']
        entry['text'] = ''.join(buffer)
        entry['norm'], entry['map'] = normalize(entry['text'])
    return sections


def title_matches(entry, wanted):
    wanted = normalize(wanted)[0]
    names = [entry['title'], *entry['path'], *entry['merged']]
    return any(wanted and wanted in normalize(n)[0] for n in names)


def find_quote(quote, paper_id, section_title, sections):
    """Return (status, hits); each hit is (section, start, end) in original section-text indices."""
    needle = normalize(quote)[0]
    candidates = [s for s in sections.values() if s['paper_id'] == paper_id]
    if section_title:
        scoped = [s for s in candidates if title_matches(s, section_title)]
        if not scoped:
            return 'section_not_found', []
        candidates = scoped
    hits = []
    for s in candidates:
        start = s['norm'].find(needle)
        while start != -1:
            end_norm = start + len(needle) - 1
            hits.append((s, s['map'][start], s['map'][end_norm] + 1))
            start = s['norm'].find(needle, start + 1)
    if not hits:
        return 'quote_not_found', []
    if len(hits) > 1:
        return 'ambiguous', hits
    return 'ok', hits


def chunk_ids_for(section, start, end):
    inside = [c['chunk_id'] for c in section['chunks'] if c['char_start'] <= start and end <= c['char_end']]
    if inside:
        return inside, 'contained'
    touching = [c['chunk_id'] for c in section['chunks'] if c['char_start'] < end and start < c['char_end']]
    return touching, 'spans_chunks'


def check_question(q, papers, errors):
    qid = q.get('question_id', '?')

    def err(message):
        errors.append(f'{qid}: {message}')

    for field in ['question_id', 'split', 'type', 'question', 'gold_paper_ids', 'key_points', 'evidence']:
        if field not in q:
            err(f'missing field {field}')
    if errors and errors[-1].startswith(f'{qid}: missing'):
        return False
    if q['split'] not in SPLITS:
        err(f'split must be one of {sorted(SPLITS)}')
    if q['type'] not in QUESTION_TYPES:
        err(f'type must be one of {sorted(QUESTION_TYPES)}')
    for pid in q['gold_paper_ids']:
        if pid not in papers:
            err(f'unknown paper_id {pid}')
    if q['type'] == 'unanswerable':
        if q['evidence'] or q['key_points'] or q['gold_paper_ids']:
            err('unanswerable questions must have empty evidence, key_points and gold_paper_ids')
        return not any(e.startswith(qid + ':') for e in errors)
    if not q['evidence'] or not q['key_points']:
        err('answerable questions need evidence and key_points')
    if q['type'] in {'multi_paper', 'comparison'} and len(set(q['gold_paper_ids'])) < 2:
        err(f"{q['type']} questions need at least 2 gold papers")
    evidence_ids = [e.get('evidence_id') for e in q['evidence']]
    if len(set(evidence_ids)) != len(evidence_ids):
        err('duplicate evidence_id')
    covered = {eid for k in q['key_points'] for eid in k.get('evidence_ids', [])}
    for k in q['key_points']:
        if not k.get('point_id') or not k.get('text') or not k.get('evidence_ids'):
            err(f"key point {k.get('point_id')} needs point_id, text and evidence_ids")
        for eid in k.get('evidence_ids', []):
            if eid not in evidence_ids:
                err(f"key point {k.get('point_id')} cites unknown evidence {eid}")
    if set(evidence_ids) - covered:
        err(f'evidence not used by any key point: {sorted(set(evidence_ids) - covered)}')
    used_papers = {e.get('paper_id') for e in q['evidence']}
    if used_papers != set(q['gold_paper_ids']):
        err(f'gold_paper_ids {sorted(q["gold_paper_ids"])} must equal papers of the evidence {sorted(used_papers)}')
    return not any(e.startswith(qid + ':') for e in errors)


def align(path, out_path):
    chunks = load_chunks()
    papers = load_papers()
    sections = section_texts(chunks)
    questions = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    errors, aligned, ids = [], [], set()
    for q in questions:
        if q.get('question_id') in ids:
            errors.append(f"{q.get('question_id')}: duplicate question_id")
        ids.add(q.get('question_id'))
        if not check_question(q, papers, errors):
            continue
        for e in q['evidence']:
            tag = f"{q['question_id']}/{e.get('evidence_id')}"
            quote = e.get('quote', '')
            if not (MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS):
                errors.append(f'{tag}: quote length {len(quote)} outside {MIN_QUOTE_CHARS}-{MAX_QUOTE_CHARS} characters')
                continue
            status, hits = find_quote(quote, e['paper_id'], e.get('section_title'), sections)
            if status != 'ok':
                detail = ''
                if status == 'ambiguous':
                    detail = ' in sections ' + ', '.join(sorted({h[0]['title'] for h in hits}))
                errors.append(f'{tag}: {status}{detail}')
                continue
            section, start, end = hits[0]
            ids_found, how = chunk_ids_for(section, start, end)
            e['matched_chunk_ids'] = ids_found
            e['match_type'] = how
            e['resolved_section'] = section['title']
        aligned.append(q)
    for q in aligned:
        q['chunk_run_id'] = run_id()
    if errors:
        print(f'{len(errors)} problem(s):')
        for line in errors:
            print('  -', line)
    ok = [q for q in aligned if all('matched_chunk_ids' in e for e in q['evidence'])]
    if not errors:
        Path(out_path).write_text('\n'.join(json.dumps(q, ensure_ascii=False) for q in ok) + '\n', encoding='utf-8')
        by_type = defaultdict(int)
        for q in ok:
            by_type[q['type']] += 1
        print(f'OK: {len(ok)} questions aligned to run {run_id()} -> {out_path}', dict(by_type))
        spans = sum(e['match_type'] == 'spans_chunks' for q in ok for e in q['evidence'])
        if spans:
            print(f'note: {spans} quote(s) straddle a chunk boundary; consider a shorter quote inside one chunk')
    return 1 if errors else 0


def grep(pattern, paper_id, limit):
    regex = re.compile(pattern, re.IGNORECASE)
    shown = 0
    for c in load_chunks():
        if paper_id and c['paper_id'] != paper_id:
            continue
        match = regex.search(c['text'])
        if not match:
            continue
        lo, hi = max(0, match.start() - 120), min(len(c['text']), match.end() + 120)
        snippet = ' '.join(c['text'][lo:hi].split())
        print(f"{c['paper_id']} | {c['section_title']} | {c['chunk_id']}\n    ...{snippet}...")
        shown += 1
        if shown >= limit:
            print(f'(stopped at {limit} results; narrow the pattern or use --paper)')
            break
    if not shown:
        print('no match')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('papers')
    g = sub.add_parser('grep')
    g.add_argument('pattern')
    g.add_argument('--paper')
    g.add_argument('--limit', type=int, default=20)
    a = sub.add_parser('align')
    a.add_argument('questions')
    a.add_argument('--out')
    args = parser.parse_args(argv)
    if args.command == 'papers':
        counts = defaultdict(int)
        for c in load_chunks():
            counts[c['paper_id']] += 1
        for pid, doc in load_papers().items():
            print(f"{pid}\t{doc['citation']}\t{doc['title']}\t{counts[pid]} chunks")
        return 0
    if args.command == 'grep':
        grep(args.pattern, args.paper, args.limit)
        return 0
    out = args.out or str(Path(args.questions).with_suffix('.aligned.jsonl'))
    return align(args.questions, out)


if __name__ == '__main__':
    sys.exit(main())
