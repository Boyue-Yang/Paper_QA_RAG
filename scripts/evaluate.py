"""Run the three arms on the evaluation set and score them.

  python -m scripts.evaluate run   --arm baseline --split dev [--limit 5]
  python -m scripts.evaluate run   --arm verify   --split dev      # verify once, drop failures (no revision)
  python -m scripts.evaluate run   --arm rules    --split dev      # fixed rules: verify and revise, up to 2 rounds
  python -m scripts.evaluate run   --arm agent    --split dev      # the model decides every next action
  python -m scripts.evaluate score --split dev                     # all arms found in data/runs, table to stdout
  python -m scripts.evaluate score --split dev --no-judge          # retrieval and abstention only, no model calls

Metrics, per question and then averaged (answerable questions and unanswerable ones are reported separately):
  retrieval  hit@8, key-point reach@8 (share of key points with at least one gold chunk retrieved), evidence
             recall@8, MRR and nDCG@8 over gold chunks (chunk level, from `matched_chunk_ids`), and the same reach
             over the whole retrieved pool (the agent retrieves more than 8 chunks)
  citation   cited_kp_reach (key points whose gold chunk is cited by a kept claim);
             citation_correct = share of kept claims whose citations exist and either include a gold chunk or
             are judged to support the claim
  support    supported_rate / unsupported_rate: share of kept claims the evaluation judge rates supported / not
  complete   completeness: share of key points the brief covers (partial counts half), judged from the claims
  abstain    unanswerable: abstained (answerable=false); answerable: false_abstain

The evaluation judge is the `judge` role (a different model from the pipeline's `verifier`).
"""
import argparse
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scripts.adaptive_agent import run_adaptive
from scripts.agent import run_agent
from scripts.llm import LLM, LLMError
from scripts.rag import Claim, answer
from scripts.retrieval import Retriever
from scripts.verify import verify_claims

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'data' / 'runs'
ARMS = {'baseline': None, 'verify': 0, 'rules': 2, 'agent': 'adaptive'}
# arm -> how it runs: None = plain RAG; an int = fixed-rule pipeline with that many repair rounds (no adaptive
# search); 'adaptive' = the model-driven agent that chooses its own actions
K = 8

COVER_SYSTEM = """You grade how completely a short research brief answers a question. You get the question, a list of key points that a good answer should contain, and the brief (a list of claims).

For each key point decide, from the brief alone:
- "covered": the brief states the key point (wording may differ, the content must match, including numbers).
- "partial": the brief states part of it, or is vague about it.
- "missing": the brief does not state it, or contradicts it.

Reply with json only, in this shape:
{"points": [{"point_id": "k1", "coverage": "covered"}]}"""


# ---------- data ----------

def default_questions():
    for path in (ROOT / 'data' / 'eval' / 'questions.aligned.jsonl', ROOT / 'questions.aligned.jsonl'):
        if path.exists():
            return path
    raise SystemExit('questions.aligned.jsonl not found; pass --questions')


def load_questions(path, split=None):
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if split in (None, 'all', r['split'])]    # 'all' = dev + test merged


def gold_chunks(question):
    return {c for e in question['evidence'] for c in e['matched_chunk_ids']}


# ---------- retrieval metrics (pure) ----------

def evidence_hits(question, retrieved):
    """Per evidence item: does any of its matched chunks appear in `retrieved`?"""
    got = set(retrieved)
    return {e['evidence_id']: any(c in got for c in e['matched_chunk_ids']) for e in question['evidence']}


def key_point_reach(question, chunk_ids):
    """Share of key points with at least one of their evidence items matched in `chunk_ids`."""
    hits = evidence_hits(question, chunk_ids)
    points = question['key_points']
    if not points:
        return None
    return sum(any(hits.get(e, False) for e in p['evidence_ids']) for p in points) / len(points)


def retrieval_metrics(question, retrieved, k=K):
    gold = gold_chunks(question)
    top = list(retrieved[:k])
    hits = evidence_hits(question, top)
    rank = next((i for i, c in enumerate(retrieved, 1) if c in gold), None)
    gains = [1.0 if c in gold else 0.0 for c in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {f'hit@{k}': float(any(hits.values())),
            f'kp_reach@{k}': key_point_reach(question, top),
            f'evid_recall@{k}': sum(hits.values()) / len(hits),
            'mrr': 1 / rank if rank else 0.0,
            f'ndcg@{k}': dcg / ideal if ideal else 0.0,
            'pool_kp_reach': key_point_reach(question, list(retrieved)),
            'pool_size': len(retrieved)}


# ---------- claim-level scoring ----------

def to_claims(brief):
    return [Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__}) for c in brief['claims']]


def judge_completeness(question, brief, llm):
    """{point_id: 'covered'|'partial'|'missing'}; a failed judge call returns None."""
    if not brief['claims']:
        return {p['point_id']: 'missing' for p in question['key_points']}
    points = '\n'.join(f"{p['point_id']}: {p['text']}" for p in question['key_points'])
    claims = '\n'.join(f"- {c['text']}" for c in brief['claims'])
    user = f"Question: {question['question']}\n\nKey points:\n{points}\n\nBrief:\n{claims}"
    try:
        result = llm.chat(COVER_SYSTEM, user, json_mode=True)
    except LLMError:
        return None
    rows = result.parsed.get('points') if isinstance(result.parsed, dict) else None
    found = {r.get('point_id'): str(r.get('coverage', '')).lower() for r in rows or [] if isinstance(r, dict)}
    return {p['point_id']: found.get(p['point_id'], 'missing') if found.get(p['point_id']) in {'covered', 'partial'}
            else 'missing' for p in question['key_points']}


def score_question(question, record, retriever=None, judge=None):
    """One score dict for one run record. `judge` None skips every model-judged metric."""
    brief = record['brief']
    score = {'question_id': question['question_id'], 'type': question['type'], 'arm': record['arm']}
    if brief is None:
        return {**score, 'error': record.get('error', 'no brief')}
    tokens = brief.get('usage', {}).get('total_tokens')
    score.update(abstained=not brief['answerable'], n_claims=len(brief['claims']), tokens=tokens,
                 seconds=record.get('seconds'))
    retrieved = [h['chunk_id'] for h in brief['retrieved']]
    claims = to_claims(brief)
    gold = gold_chunks(question)
    cited = [c for claim in claims for c in claim.cites]
    if question['type'] == 'unanswerable':
        score['false_support'] = None
    else:
        score.update(retrieval_metrics(question, retrieved))
        score['cited_kp_reach'] = key_point_reach(question, cited)
        score['false_abstain'] = score['abstained']
    verdicts = {}
    if judge and claims:
        verdicts = verify_claims(claims, retriever, judge)
    if claims:
        supported = [verdicts[c.claim_id]['verdict'] == 'supported' for c in claims] if verdicts else None
        score['supported_rate'] = sum(supported) / len(claims) if supported else None
        score['unsupported_rate'] = 1 - score['supported_rate'] if supported else None
        structure_ok = [bool(c.cites) and not c.invalid_cites and all(retriever.lookup_citation(x) for x in c.cites)
                        for c in claims] if retriever else None
        if supported is not None and structure_ok is not None:
            score['citation_correct'] = sum(ok and (bool(set(c.cites) & gold) or s)
                                            for c, ok, s in zip(claims, structure_ok, supported)) / len(claims)
    if question['type'] != 'unanswerable' and judge:
        coverage = judge_completeness(question, brief, judge)
        if coverage is not None:
            weight = {'covered': 1.0, 'partial': 0.5, 'missing': 0.0}
            score['completeness'] = sum(weight[v] for v in coverage.values()) / len(coverage)
            score['coverage'] = coverage
    return score


# ---------- aggregation ----------

GROUPS = {'answerable': ('single_fact', 'multi_paper', 'comparison'), 'single_fact': ('single_fact',),
          'multi_paper': ('multi_paper',), 'comparison': ('comparison',), 'unanswerable': ('unanswerable',)}
METRICS = ['hit@8', 'kp_reach@8', 'evid_recall@8', 'mrr', 'ndcg@8', 'pool_kp_reach', 'cited_kp_reach',
           'citation_correct', 'supported_rate', 'unsupported_rate', 'completeness', 'false_abstain', 'abstained',
           'n_claims', 'tokens', 'seconds']


def summarize(scores):
    """{group: {metric: mean or None, 'n': count}}; questions that errored are counted under 'errors'."""
    out = {}
    for group, types in GROUPS.items():
        rows = [s for s in scores if s['type'] in types]
        ok = [s for s in rows if 'error' not in s]
        entry = {'n': len(rows), 'errors': len(rows) - len(ok)}
        for metric in METRICS:
            values = [s[metric] for s in ok if s.get(metric) is not None]
            entry[metric] = sum(values) / len(values) if values else None
        out[group] = entry
    return out


def format_table(summaries, group):
    arms = list(summaries)
    metrics = [m for m in METRICS if any(summaries[a][group].get(m) is not None for a in arms)]
    lines = [f'[{group}]  n=' + ', '.join(f"{a}:{summaries[a][group]['n']}(err {summaries[a][group]['errors']})"
                                          for a in arms),
             f"{'metric':<18}" + ''.join(f'{a:>10}' for a in arms)]
    for metric in metrics:
        cells = []
        for arm in arms:
            value = summaries[arm][group].get(metric)
            cells.append(f"{'-':>10}" if value is None else f'{value:>10.1f}' if metric in {'tokens', 'seconds'}
                         else f'{value:>10.3f}')
        lines.append(f'{metric:<18}' + ''.join(cells))
    return '\n'.join(lines)


# ---------- running ----------

class LockedRetriever:
    """Serialises GPU encoding so several questions can run their LLM calls in parallel."""

    def __init__(self, retriever):
        self._inner, self._lock = retriever, threading.Lock()

    def retrieve(self, *args, **kwargs):
        with self._lock:
            return self._inner.retrieve(*args, **kwargs)

    def lookup_citation(self, chunk_id):
        return self._inner.lookup_citation(chunk_id)

    def papers(self):
        return self._inner.papers()


def run_one(arm, question, retriever, llms):
    started = time.time()
    rounds = ARMS[arm]
    if rounds is None:
        brief = answer(question['question'], retriever, llms['drafter'], K)
    elif rounds == 'adaptive':
        brief = run_adaptive(question['question'], retriever, llms, k=K, workers=3)
    else:
        brief = run_agent(question['question'], retriever, llms, max_rounds=rounds, k=K, workers=3)
    return asdict(brief), round(time.time() - started, 1)


def run_path(split, arm):
    return RUNS / f'{split}_{arm}.jsonl'


def load_records(path):
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {r['question_id']: r for r in rows}


def cmd_run(args):
    questions = load_questions(args.questions or default_questions(), args.split)
    if args.limit:
        questions = questions[:args.limit]
    RUNS.mkdir(parents=True, exist_ok=True)
    path = run_path(args.split, args.arm)
    done = {qid for qid, r in load_records(path).items() if r.get('brief')}
    todo = [q for q in questions if q['question_id'] not in done]
    print(f'{args.arm}/{args.split}: {len(done)} done, {len(todo)} to run')
    retriever = LockedRetriever(Retriever())
    llms = {name: LLM(name) for name in ('drafter', 'verifier', 'controller')}
    write_lock = threading.Lock()

    def work(question):
        record = {'question_id': question['question_id'], 'arm': args.arm}
        try:
            record['brief'], record['seconds'] = run_one(args.arm, question, retriever, llms)
        except Exception as error:    # keep the batch going; failed questions are retried on the next run
            record.update(brief=None, error=f'{type(error).__name__}: {error}')
        with write_lock, open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(question['question_id'], 'error: ' + record['error'] if record.get('error') else
              f"{record['seconds']}s, {len(record['brief']['claims'])} claims, answerable={record['brief']['answerable']}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    return 0


def cmd_score(args):
    questions = {q['question_id']: q for q in load_questions(args.questions or default_questions(), args.split)}
    retriever = Retriever() if not args.no_judge else None
    judge = LLM('judge') if not args.no_judge else None
    summaries = {}
    for arm in (args.arms or ARMS):
        records = load_records(run_path(args.split, arm))
        records = {qid: r for qid, r in records.items() if qid in questions}
        if not records:
            continue
        # a retried question appears twice in the file; load_records keeps the last record
        scores = [score_question(questions[qid], r, retriever, judge) for qid, r in sorted(records.items())]
        RUNS.mkdir(parents=True, exist_ok=True)
        with open(RUNS / f'{args.split}_{arm}.scores.jsonl', 'w', encoding='utf-8') as handle:
            for score in scores:
                handle.write(json.dumps(score, ensure_ascii=False) + '\n')
        summaries[arm] = summarize(scores)
    if not summaries:
        print('no runs found in', RUNS)
        return 1
    for group in GROUPS:
        if not any(s[group]['n'] for s in summaries.values()):
            continue
        print(format_table(summaries, group), end='\n\n')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--arm', choices=list(ARMS), required=True)
    run.add_argument('--split', default='dev')
    run.add_argument('--limit', type=int)
    run.add_argument('--workers', type=int, default=3)
    run.add_argument('--questions')
    score = sub.add_parser('score')
    score.add_argument('--split', default='dev')
    score.add_argument('--arms', nargs='*', choices=list(ARMS))
    score.add_argument('--no-judge', action='store_true')
    score.add_argument('--questions')
    args = parser.parse_args(argv)
    return cmd_run(args) if args.command == 'run' else cmd_score(args)


if __name__ == '__main__':
    sys.exit(main())
