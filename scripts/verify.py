"""Citation verification: is each claim supported by the chunks it cites?

Three checks, cheapest first:
  1. structure (no model): the claim has citations and every cited chunk exists
  2. numbers (no model): digits in the claim that do not appear in the cited text are flagged to the judge
  3. judge (LLM, only sees the claim and its cited chunks): supported / partial / unsupported / contradicted,
     with a quote copied from the excerpts. The quote is checked by program; a "supported" verdict whose quote
     is not really in the cited text is downgraded to unsupported.

Only "supported" counts as passing. A judge call that fails is "unverified", never a pass.
"""
import re
from concurrent.futures import ThreadPoolExecutor

from scripts.eval_set import normalize
from scripts.llm import LLMError

VERDICTS = {'supported', 'partial', 'unsupported', 'contradicted'}
MAX_JUDGE_CHUNKS = 4

JUDGE_SYSTEM = """You are a strict fact-checker for a research brief. You get one claim and the excerpts it cites.

Decide whether the claim is supported by the excerpts alone. Ignore anything you know from elsewhere.
- "supported": every part of the claim is directly stated by, or a plain restatement of, the excerpts.
- "partial": some of the claim is stated, but part is missing, stronger, broader or more specific than the excerpts.
- "unsupported": the excerpts do not say it.
- "contradicted": the excerpts say something that conflicts with the claim.

Copy, word for word, the one passage from the excerpts that most directly bears on the claim into "quote" (at most 60 words; empty if there is no relevant passage). Do not alter it.

Reply with json only, in this shape:
{"verdict": "supported", "quote": "exact passage copied from an excerpt", "reason": "one short sentence"}"""

_NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?')


def numbers_in(text):
    return {float(n) for n in _NUMBER.findall(re.sub(r'(?<=\d),(?=\d{3})', '', text))}


def structure_problem(claim, retriever):
    """A reason string if the citations are unusable, else None."""
    if claim.invalid_cites:
        return f'cites excerpts that were never shown: {claim.invalid_cites}'
    if not claim.cites:
        return 'has no citation'
    missing = [c for c in claim.cites if retriever.lookup_citation(c) is None]
    return f'cites chunks that do not exist: {missing}' if missing else None


def judge_claim(claim, cited, llm):
    """`cited` is a list of lookup_citation() dicts for the claim's chunks. Returns a verdict dict."""
    missing_numbers = sorted(numbers_in(claim.text) - numbers_in(' '.join(c['text'] for c in cited)))
    blocks = [f"[E{i}] {c['citation']}, \"{c['paper_title']}\", section: {' > '.join(c['section_path'])}\n{c['text']}"
              for i, c in enumerate(cited[:MAX_JUDGE_CHUNKS], 1)]
    user = f'Claim: {claim.text}\n\n'
    if missing_numbers:
        user += f'Note: the claim contains numbers {missing_numbers} that do not appear anywhere in the excerpts.\n\n'
    user += 'Excerpts:\n\n' + '\n\n'.join(blocks)
    base = {'checked_by': 'llm', 'numbers_missing': missing_numbers, 'usage': {}}
    try:
        result = llm.chat(JUDGE_SYSTEM, user, json_mode=True)
    except LLMError as error:
        return {**base, 'verdict': 'unverified', 'raw_verdict': None, 'reason': f'judge call failed: {error}',
                'quote': '', 'quote_found': False}
    data = result.parsed if isinstance(result.parsed, dict) else {}
    raw = str(data.get('verdict', '')).strip().lower()
    quote = str(data.get('quote') or '').strip()
    reason = str(data.get('reason') or '').strip()
    verdict = raw if raw in VERDICTS else 'unsupported'
    if raw not in VERDICTS:
        reason = f'judge returned an invalid verdict {raw!r}. ' + reason
    haystack = normalize(' '.join(c['text'] for c in cited))[0]
    quote_found = bool(quote) and normalize(quote)[0] in haystack
    if verdict in {'supported', 'partial'} and not quote_found:
        reason = 'judge gave no verbatim quote from the cited chunks. ' + reason
        verdict = 'unsupported'
    return {**base, 'verdict': verdict, 'raw_verdict': raw, 'reason': reason, 'quote': quote,
            'quote_found': quote_found, 'usage': result.usage}


def verify_claim(claim, retriever, llm):
    problem = structure_problem(claim, retriever)
    if problem:
        return {'verdict': 'unsupported', 'raw_verdict': None, 'reason': 'claim ' + problem, 'quote': '',
                'quote_found': False, 'numbers_missing': [], 'checked_by': 'structure', 'usage': {}}
    return judge_claim(claim, [retriever.lookup_citation(c) for c in claim.cites], llm)


def verify_claims(claims, retriever, llm, workers=6):
    """{claim_id: verdict dict}; judge calls run in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda c: verify_claim(c, retriever, llm), claims))
    return {c.claim_id: v for c, v in zip(claims, verdicts)}
