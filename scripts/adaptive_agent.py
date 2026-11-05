"""Adaptive agent: a language model decides, one step at a time, what to do next.

  python -m scripts.adaptive_agent "How do layer normalization and batch normalization differ?"
  python -m scripts.adaptive_agent "..." --steps 12 --out brief.json

Unlike scripts/agent.py (fixed rules decide the next action), here the `controller` model sees the current
state (evidence found so far, claims and their verdicts, its own notes, the budget) and picks one action:

  search  retrieve new passages with a query it writes itself, optionally inside chosen papers
  draft   write claims from passages that have not been used yet (or from passages it names)
  verify  fact-check claims against the passages they cite
  revise  repair a claim that failed (retrieve again with the claim, rewrite or remove it)
  drop    remove a claim (failed, off-topic or duplicated)
  finish  stop, saying whether the question is answerable and what could not be found

The model chooses the order, the queries, the papers to look in and when to stop. Code only enforces what must
not depend on the model: an initial retrieval, a step budget, claims are verified before the brief is
written, and only claims judged `supported` reach the final brief. Every decision, with the model's reason and
the result it saw, is kept in `Brief.trace`.
"""
import argparse
import json
import re
import sys
from dataclasses import asdict

from scripts.agent import add_usage, as_hit, revise_claim
from scripts.llm import LLM, LLMError
from scripts.rag import Brief, draft_claims, to_markdown
from scripts.retrieval import Retriever
from scripts.verify import verify_claims

MAX_STEPS = 12
K_SEED = 8             # the automatic first retrieval, same size as the baseline's
K_SEARCH = 6           # new passages returned by one search action
MAX_DRAFT_EVIDENCE = 14
DUPLICATE_OVERLAP = 0.8
MAX_ERRORS_IN_A_ROW = 3
RECENT_DECISIONS = 8

CONTROLLER_SYSTEM = """You are the controller of a research agent. You answer a question using ONLY a fixed collection of academic papers, by choosing one action at a time. A separate fact-checker verifies every claim against the passages it cites, and only claims judged "supported" reach the user.

Actions (args in brackets are optional):
- search: {"query": "text", ["papers": ["arxiv_xxxx.xxxxx"]]} retrieve new passages for a query you write. Passages are named E1, E2, ... Use "papers" to look inside specific papers from the collection list.
- draft: {["evidence": ["E1", "E4"]], ["focus": "the sub-question to answer"]} write claims from the evidence. By default it uses every passage not drafted yet (marked * in the pool).
- verify: {["claims": ["c1", "c2"]]} fact-check claims. By default all claims not yet checked.
- revise: {"claim": "c3"} for a claim that failed verification: retrieve again using the claim itself, then rewrite it to what the sources really say, or remove it.
- drop: {"claim": "c3"} remove a claim (failed, off-topic, or a duplicate of another claim).
- finish: {"answerable": true or false, "not_found": "what could not be found, or empty"} stop.

Guidance:
- You choose the order and when to stop. Plan in "notes": list the parts of the question and their status. Your notes are shown back to you at the next step.
- If the question compares things or involves several papers or parts, search for each part separately; use "papers" to look inside a particular paper.
- If the passages do not answer something, try again with a differently worded query or inside a specific paper before giving up. Give up when targeted searches keep finding nothing relevant.
- Verify claims before you finish; fix or drop those that fail. Drop duplicates.
- If the key fact the question asks for is not in the collection (it is not reported, or the question's premise does not match the papers), finish with answerable=false and say what is missing. Do not pad the answer with loosely related claims. If only part of the question can be answered, finish with answerable=true and describe the missing part in not_found.
- Use only the passages; never outside knowledge. Be efficient: do not repeat an action that already gave its result.

Reply with json only, in this shape:
{"thought": "one or two sentences: why this action", "action": "search", "args": {"query": "..."}, "notes": "your running plan and status"}"""


class ActionError(Exception):
    """The controller asked for something invalid; the message goes back to it as the result."""


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [str(value)]
    return [str(v) for v in value] if isinstance(value, list) else []


def snippet(text, width=140):
    return ' '.join(text.split())[:width]


def tokens(text):
    return set(re.findall(r'[a-z0-9.]+', text.lower()))


def overlap(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


class State:
    """Everything the controller can see and change."""

    def __init__(self, question, retriever, drafter, verifier, workers=3):
        self.question, self.retriever, self.drafter, self.verifier, self.workers = (
            question, retriever, drafter, verifier, workers)
        self.pool = {}          # chunk_id -> hit, in the order found
        self.handle = {}        # chunk_id -> "E3"
        self.chunk_of = {}      # "E3" -> chunk_id
        self.drafted = set()    # chunk ids already used for drafting
        self.claims = []
        self.claim_count = 0
        self.notes = ''
        self.usage = {}
        self.log = []
        self.last = ''
        self.papers = retriever.papers()

    # ---- evidence pool ----
    def add_hit(self, hit):
        chunk_id = hit['chunk_id']
        if chunk_id not in self.pool:
            self.pool[chunk_id] = hit
            self.handle[chunk_id] = f'E{len(self.pool)}'
            self.chunk_of[self.handle[chunk_id]] = chunk_id
        return self.handle[chunk_id]

    def register(self, chunk_id):
        """Handle for any chunk, adding it to the pool if a repair step brought it in."""
        if chunk_id not in self.pool:
            info = self.retriever.lookup_citation(chunk_id)
            if not info:
                return chunk_id
            self.add_hit({**as_hit(info), 'score': None, 'paper_id': info.get('paper_id')})
        self.drafted.add(chunk_id)
        return self.handle[chunk_id]

    def search(self, query, papers, k):
        hits = self.retriever.retrieve(query, k + len(self.pool), papers or None)
        new = [h for h in hits if h['chunk_id'] not in self.pool][:k]
        if not new:
            return 'No new passages: everything that matched is already in the pool.'
        lines = [f'{len(new)} new passages:']
        for hit in new:
            lines.append(f"  {self.add_hit(hit)} {hit['paper_id']} | {hit['section_title'][:50]} | "
                         f"score {hit['score']:.2f} | {snippet(hit['text'])}")
        return '\n'.join(lines)

    # ---- claims ----
    def add_claims(self, claims):
        lines = []
        for claim in claims:
            self.claim_count += 1
            claim.claim_id = f'c{self.claim_count}'
            twin = next((c for c in self.claims if overlap(claim.text, c.text) >= DUPLICATE_OVERLAP), None)
            cites = ', '.join(self.register(c) for c in claim.cites) or 'no citation'
            lines.append(f"  {claim.claim_id}: {claim.text} [cites {cites}]"
                         + (f' (looks like a duplicate of {twin.claim_id})' if twin else ''))
            self.claims.append(claim)
        return lines

    def pick(self, ids):
        by_id = {c.claim_id: c for c in self.claims}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ActionError(f'unknown claim ids {unknown}; current claims: {list(by_id)}')
        return [by_id[i] for i in ids]

    # ---- actions ----
    def act_search(self, args, k=K_SEARCH):
        query = args.get('query')
        if not isinstance(query, str) or not query.strip():
            raise ActionError('search needs a "query" string')
        papers = as_list(args.get('papers'))
        unknown = [p for p in papers if p not in self.papers]
        if unknown:
            raise ActionError(f'unknown paper ids {unknown}; use ids from the collection list')
        return self.search(query.strip(), papers, k)

    def act_draft(self, args):
        refs = as_list(args.get('evidence'))
        if refs:
            unknown = [r for r in refs if r not in self.chunk_of]
            if unknown:
                raise ActionError(f'unknown evidence ids {unknown}')
            ids = [self.chunk_of[r] for r in refs]
        else:
            ids = [c for c in self.pool if c not in self.drafted]
            if not ids:
                raise ActionError('no undrafted evidence: search first, or name evidence ids to draft from')
        ids = ids[:MAX_DRAFT_EVIDENCE]
        focus = args.get('focus')
        focus = focus.strip() if isinstance(focus, str) and focus.strip() else self.question
        claims, data, usage, warnings = draft_claims(focus, [self.pool[c] for c in ids], self.drafter)
        add_usage(self.usage, usage)
        self.drafted.update(ids)
        lines = self.add_claims(claims)
        if not lines:
            return f"The drafter wrote no claims from {len(ids)} passages. Its note: {data.get('not_found') or '(none)'}"
        note = f"\n  drafter's note on what is missing: {data['not_found']}" if data.get('not_found') else ''
        return f'{len(lines)} new claims (not verified yet):\n' + '\n'.join(lines) + note

    def act_verify(self, args):
        ids = as_list(args.get('claims')) or [c.claim_id for c in self.claims if c.verdict is None]
        todo = self.pick(ids)
        if not todo:
            raise ActionError('there is nothing to verify')
        verdicts = verify_claims(todo, self.retriever, self.verifier, self.workers)
        lines = []
        for claim in todo:
            claim.verdict = verdicts[claim.claim_id]
            add_usage(self.usage, claim.verdict['usage'])
            verdict = claim.verdict['verdict']
            lines.append(f'  {claim.claim_id}: {verdict}' + ('' if verdict == 'supported'
                                                           else f" - {claim.verdict['reason'][:160]}"))
        return 'Verification:\n' + '\n'.join(lines)

    def act_revise(self, args):
        ids = as_list(args.get('claim'))
        if len(ids) != 1:
            raise ActionError('revise needs one "claim" id')
        claim = self.pick(ids)[0]
        if claim.verdict is None:
            raise ActionError(f'{claim.claim_id} has not been verified yet; verify it first')
        if claim.verdict['verdict'] == 'supported':
            raise ActionError(f'{claim.claim_id} is already supported')
        new, action, usage = revise_claim(claim, claim.verdict, self.retriever, self.drafter)
        add_usage(self.usage, usage)
        self.claims.remove(claim)
        if new is None:
            return f'{claim.claim_id} removed: no passage supports any version of it.'
        lines = self.add_claims([new])
        return f'{claim.claim_id} rewritten (not verified yet):\n' + '\n'.join(lines)

    def act_drop(self, args):
        ids = as_list(args.get('claim'))
        if len(ids) != 1:
            raise ActionError('drop needs one "claim" id')
        claim = self.pick(ids)[0]
        self.claims.remove(claim)
        return f'{claim.claim_id} dropped.'

    # ---- what the controller sees ----
    def render(self, step, max_steps):
        lines = [f'Question: {self.question}', '', f'Your notes from the previous step: {self.notes or "(none yet)"}', '',
                 f'Evidence pool ({len(self.pool)} passages; * = not drafted yet):']
        for chunk_id, hit in self.pool.items():
            lines.append(f"  {self.handle[chunk_id]}{'' if chunk_id in self.drafted else '*'} "
                         f"{hit['paper_id']} | {hit['section_title'][:50]}")
        lines += ['', f'Claims ({len(self.claims)}):'] if self.claims else ['', 'Claims: none yet']
        for claim in self.claims:
            status = 'not verified' if claim.verdict is None else claim.verdict['verdict']
            lines.append(f'  {claim.claim_id} [{status}] {claim.text}')
        lines += ['', 'Your previous decisions:']
        for record in self.log[-RECENT_DECISIONS:]:
            lines.append(f"  step {record['step']}: {record['action']} {json.dumps(record['args'], ensure_ascii=False)[:150]}")
        lines += ['', 'Result of your last action:', self.last, '',
                  f'Step {step} of {max_steps}.' + (' This is your LAST step: you must finish now.' if step == max_steps else '')]
        return '\n'.join(lines)


HANDLERS = {'search': State.act_search, 'draft': State.act_draft, 'verify': State.act_verify,
            'revise': State.act_revise, 'drop': State.act_drop}


def collection_list(papers):
    return '\n'.join(f"{pid}: {info['title']}" for pid, info in sorted(papers.items()))


def run_adaptive(question, retriever, llms, max_steps=MAX_STEPS, k=K_SEED, workers=3):
    controller = llms['controller']
    state = State(question, retriever, llms['drafter'], llms['verifier'], workers)
    system = CONTROLLER_SYSTEM + '\n\nCollection (paper id: title):\n' + collection_list(state.papers)
    warnings = []

    state.last = state.act_search({'query': question}, k)       # automatic first retrieval, same size as the baseline's
    state.log.append({'step': 0, 'action': 'search', 'args': {'query': question},
                      'thought': 'initial retrieval (automatic)', 'result': state.last})

    finish, errors = None, 0
    for step in range(1, max_steps + 1):
        try:
            result = controller.chat(system, state.render(step, max_steps), json_mode=True)
        except LLMError as error:
            warnings.append(f'controller call failed at step {step}: {error}')
            break
        add_usage(state.usage, result.usage)
        data = result.parsed if isinstance(result.parsed, dict) else {}
        action = str(data.get('action') or '').strip().lower()
        args = data.get('args') if isinstance(data.get('args'), dict) else {}
        if isinstance(data.get('notes'), str):
            state.notes = data['notes'][:1500]
        record = {'step': step, 'action': action, 'args': args, 'thought': str(data.get('thought') or '')[:500]}
        if action == 'finish':
            record['result'] = 'finished'
            state.log.append(record)
            finish = args
            break
        try:
            if action not in HANDLERS:
                raise ActionError(f'unknown action {action!r}; choose one of {sorted(HANDLERS) + ["finish"]}')
            observation = HANDLERS[action](state, args)
            errors = 0
        except ActionError as error:
            observation = f'ERROR: {error}'
            errors += 1
        record['result'] = observation[:2500]
        state.log.append(record)
        state.last = observation
        if errors >= MAX_ERRORS_IN_A_ROW:
            warnings.append('stopped after repeated invalid actions')
            break
    if finish is None:
        warnings.append('the agent did not finish by itself; the brief was closed by the step budget')
        finish = {}

    pending = [c for c in state.claims if c.verdict is None]
    if pending:    # verification cannot be skipped: nothing unchecked reaches the brief
        observation = state.act_verify({'claims': [c.claim_id for c in pending]})
        state.log.append({'step': len(state.log), 'action': 'verify', 'args': {'claims': [c.claim_id for c in pending]},
                          'thought': 'automatic: claims must be verified before the brief is written', 'result': observation[:2500]})
    supported = [c for c in state.claims if c.verdict['verdict'] == 'supported']
    flag = finish.get('answerable')
    answerable = bool(supported) and flag is not False
    not_found = str(finish.get('not_found') or '').strip()
    if not answerable and not not_found:
        not_found = 'No supported answer found in the collection.'
    if len(supported) < len(state.claims):
        warnings.append(f'{len(state.claims) - len(supported)} unsupported claims were left out of the brief')
    return Brief(question=question, answerable=answerable, claims=supported if answerable else [], not_found=not_found,
                 retrieved=[{'chunk_id': c, 'score': h.get('score'), 'paper_id': h.get('paper_id'),
                             'section_title': h['section_title']} for c, h in state.pool.items()],
                 usage=state.usage, warnings=warnings, trace=state.log)


def format_decisions(trace, width=300):
    lines = []
    for record in trace:
        lines.append(f"step {record['step']} | {record['action']} {json.dumps(record['args'], ensure_ascii=False)[:200]}")
        if record.get('thought'):
            lines.append(f"    thought: {record['thought']}")
        lines.append('    result: ' + record['result'][:width].replace('\n', '\n            '))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('question')
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    parser.add_argument('--out', help='also save the full brief and decision log as json')
    args = parser.parse_args(argv)
    retriever = Retriever()
    llms = {name: LLM(name) for name in ('controller', 'drafter', 'verifier')}
    brief = run_adaptive(args.question, retriever, llms, args.steps)
    print('--- decisions ---\n' + format_decisions(brief.trace) + '\n')
    print(to_markdown(brief, retriever))
    if brief.warnings:
        print('\nwarnings:', brief.warnings)
    print('usage:', brief.usage)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(asdict(brief), handle, ensure_ascii=False, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
