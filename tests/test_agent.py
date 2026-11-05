"""Verification and the repair loop, checked with a scripted retriever and scripted LLMs (no network)."""
import unittest
from types import SimpleNamespace

from scripts.agent import run_agent
from scripts.llm import LLMError
from scripts.rag import Claim
from scripts.verify import JUDGE_SYSTEM, judge_claim, numbers_in, verify_claim

TEXTS = {
    'chunk_1': 'The base model reaches 28.4 BLEU on the WMT14 English-to-German task.',
    'chunk_2': 'Layer normalization computes its statistics over a single training case.',
    'chunk_3': 'A dropout rate of 0.1 was used for the base model.',
    'chunk_4': 'Batch normalization depends on the mini-batch size.',
}


def info(chunk_id):
    n = chunk_id.split('_')[1]
    return {'chunk_id': chunk_id, 'paper_id': f'p{n}', 'paper_title': f'Paper {n}', 'citation': f'A{n} (2020)',
            'section_path': [f'S{n}'], 'source_url': 'https://example.invalid', 'text': TEXTS[chunk_id]}


class FakeRetriever:
    def __init__(self, by_query=None, default=('chunk_1', 'chunk_2')):
        self.by_query = by_query or {}
        self.default = default
        self.queries = []

    def retrieve(self, query, k=8, paper_ids=None):
        self.queries.append(query)
        ids = next((v for key, v in self.by_query.items() if key in query), self.default)
        return [{**info(i), 'score': 0.5, 'section_title': f"S{i.split('_')[1]}"} for i in ids][:k]

    def lookup_citation(self, chunk_id):
        return info(chunk_id) if chunk_id in TEXTS else None


class ScriptedLLM:
    """handler(system, user) -> parsed json; records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def chat(self, system, user, json_mode=False):
        self.calls.append((system, user))
        return SimpleNamespace(parsed=self.handler(system, user), usage={'total_tokens': 1, 'prompt_tokens': 1,
                                                                        'completion_tokens': 0})


def judge(system, user):
    claim = user.split('\n')[0]
    if '28.4' in claim:
        return {'verdict': 'supported', 'quote': 'reaches 28.4 BLEU on the WMT14', 'reason': 'stated'}
    if 'dropout' in claim.lower():
        return {'verdict': 'supported', 'quote': 'A dropout rate of 0.1 was used', 'reason': 'stated'}
    if 'Batch normalization' in claim:
        return {'verdict': 'supported', 'quote': 'Batch normalization depends on the mini-batch size', 'reason': 'stated'}
    return {'verdict': 'unsupported', 'quote': '', 'reason': 'not in the excerpts'}


def make_llms(draft, revise=None, plan=None):
    def drafter(system, user):
        if system.startswith('You prepare a question'):
            return plan(user) if plan else {'sub_questions': [user.split('Question: ', 1)[1]]}
        if system.startswith('You repair one claim'):
            return revise(user)
        return draft(system, user)
    return {'drafter': ScriptedLLM(drafter), 'verifier': ScriptedLLM(judge)}


class VerifyTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(numbers_in('rose from 1,000 to 2.5 and 28.4%, GPT-3'), {1000.0, 2.5, 28.4, 3.0})

    def test_claim_without_citation_fails_without_calling_the_judge(self):
        llm = ScriptedLLM(judge)
        verdict = verify_claim(Claim('c1', 'Something.', []), FakeRetriever(), llm)
        self.assertEqual((verdict['verdict'], verdict['checked_by']), ('unsupported', 'structure'))
        verdict = verify_claim(Claim('c2', 'Something.', ['chunk_1'], ['C9']), FakeRetriever(), llm)
        self.assertIn('never shown', verdict['reason'])
        verdict = verify_claim(Claim('c3', 'Something.', ['chunk_404']), FakeRetriever(), llm)
        self.assertIn('do not exist', verdict['reason'])
        self.assertEqual(llm.calls, [])

    def test_supported_needs_a_quote_that_is_really_in_the_cited_chunks(self):
        cited = [info('chunk_1')]
        good = judge_claim(Claim('c1', 'It reaches 28.4 BLEU.', ['chunk_1']), cited, ScriptedLLM(judge))
        self.assertEqual((good['verdict'], good['quote_found']), ('supported', True))
        invented = ScriptedLLM(lambda s, u: {'verdict': 'supported', 'quote': 'a sentence the paper never wrote',
                                             'reason': 'trust me'})
        bad = judge_claim(Claim('c1', 'It reaches 28.4 BLEU.', ['chunk_1']), cited, invented)
        self.assertEqual((bad['verdict'], bad['raw_verdict']), ('unsupported', 'supported'))

    def test_missing_numbers_are_flagged_to_the_judge_and_bad_verdicts_do_not_pass(self):
        llm = ScriptedLLM(lambda s, u: {'verdict': 'definitely', 'quote': '', 'reason': ''})
        result = judge_claim(Claim('c1', 'It reaches 31.2 BLEU.', ['chunk_1']), [info('chunk_1')], llm)
        self.assertEqual(result['numbers_missing'], [31.2])
        self.assertIn('31.2', llm.calls[0][1])
        self.assertEqual(result['verdict'], 'unsupported')

    def test_failed_judge_call_is_unverified_not_a_pass(self):
        def boom(system, user):
            raise LLMError('down')
        result = judge_claim(Claim('c1', 'x', ['chunk_1']), [info('chunk_1')], ScriptedLLM(boom))
        self.assertEqual(result['verdict'], 'unverified')
        self.assertTrue(JUDGE_SYSTEM)


def draft_two_claims(system, user):
    return {'answerable': True, 'not_found': '', 'claims': [
        {'text': 'The base model reaches 28.4 BLEU.', 'cites': ['C1']},
        {'text': 'The base model reaches 30 BLEU.', 'cites': ['C1']}]}


class AgentTests(unittest.TestCase):
    def test_failing_claim_is_repaired_and_reverified(self):
        llms = make_llms(draft_two_claims, revise=lambda user: {
            'action': 'rewrite', 'text': 'The base model reaches 28.4 BLEU on WMT14 English-to-German.', 'cites': ['C1']})
        brief = run_agent('How good is the base model?', FakeRetriever(), llms, max_rounds=2)
        self.assertEqual(len(brief.claims), 2)
        self.assertTrue(all(c.verdict['verdict'] == 'supported' for c in brief.claims))
        self.assertEqual(brief.claims[1].revised_from, 'c2')
        self.assertEqual(len({c.claim_id for c in brief.claims}), 2)
        self.assertEqual([r['round'] for r in brief.trace], [0, 1])
        self.assertEqual(brief.trace[0]['actions'][0]['action'], 'rewrite')
        self.assertIn('30 BLEU', brief.trace[0]['claims'][1]['text'])  # the failed wording stays in the trace

    def test_repair_uses_the_claim_text_as_the_retrieval_query(self):
        retriever = FakeRetriever()
        llms = make_llms(draft_two_claims, revise=lambda user: {'action': 'drop'})
        run_agent('How good is the base model?', retriever, llms, max_rounds=1)
        self.assertIn('The base model reaches 30 BLEU.', retriever.queries)

    def test_zero_rounds_verifies_once_and_drops_failures(self):
        llms = make_llms(draft_two_claims)
        brief = run_agent('How good is the base model?', FakeRetriever(), llms, max_rounds=0)
        self.assertEqual([c.text for c in brief.claims], ['The base model reaches 28.4 BLEU.'])
        self.assertEqual(len(brief.trace), 1)
        self.assertEqual(brief.trace[0]['dropped'], ['c2'])
        self.assertTrue(brief.answerable)

    def test_unfixable_claim_is_dropped_after_the_round_budget(self):
        llms = make_llms(draft_two_claims, revise=lambda user: {'action': 'rewrite', 'text': 'It reaches 30 BLEU.',
                                                               'cites': ['C1']})
        brief = run_agent('How good is the base model?', FakeRetriever(), llms, max_rounds=2)
        self.assertEqual([c.text for c in brief.claims], ['The base model reaches 28.4 BLEU.'])
        self.assertEqual(len(brief.trace), 3)

    def test_uncovered_sub_question_is_refilled_with_unseen_chunks(self):
        def plan(user):
            return {'sub_questions': ['What BLEU does the base model get?', 'Does batch normalization depend on batch size?']}

        def draft(system, user):
            if 'Sub-questions:' in user:
                return {'answerable': True, 'not_found': '', 'claims': [
                    {'text': 'The base model reaches 28.4 BLEU.', 'cites': ['C1'], 'sub': 's1'}]}
            return {'answerable': True, 'not_found': '', 'claims': [
                {'text': 'Batch normalization depends on the mini-batch size.', 'cites': ['C1']}]}

        # the evidence for s2 (chunk_4) ranks below the first few results, so only the refill's wider search finds it
        retriever = FakeRetriever({'BLEU': ('chunk_1', 'chunk_3'), 'batch': ('chunk_2', 'chunk_3', 'chunk_1', 'chunk_4')},
                                  default=('chunk_1', 'chunk_3'))
        brief = run_agent('BLEU and batch norm?', retriever, make_llms(draft, plan=plan), max_rounds=2)
        self.assertEqual({c.sub_question for c in brief.claims}, {'s1', 's2'})
        self.assertEqual(brief.trace[0]['uncovered'], ['s2'])
        self.assertEqual(brief.trace[0]['actions'][0]['action'], 'refill')
        self.assertEqual(brief.trace[1]['uncovered'], [])
        self.assertEqual(brief.not_found, '')

    def test_abstention_when_nothing_can_be_supported(self):
        llms = make_llms(lambda s, u: {'answerable': False, 'claims': [], 'not_found': 'The excerpts say nothing.'})
        brief = run_agent('What did GPT-4 cost?', FakeRetriever(default=('chunk_1', 'chunk_2')), llms, max_rounds=2)
        self.assertFalse(brief.answerable)
        self.assertEqual(brief.claims, [])
        self.assertEqual(llms['verifier'].calls, [])
        self.assertIn('The excerpts say nothing.', brief.not_found)

    def test_unfixable_sub_question_is_reported_as_not_found(self):
        def plan(user):
            return {'sub_questions': ['What BLEU does the base model get?', 'What is the GPT-4 training cost?']}

        def draft(system, user):
            if 'Sub-questions:' in user:
                return {'answerable': True, 'not_found': '', 'claims': [
                    {'text': 'The base model reaches 28.4 BLEU.', 'cites': ['C1'], 'sub': 's1'}]}
            return {'answerable': False, 'claims': [], 'not_found': ''}

        brief = run_agent('two things', FakeRetriever(default=('chunk_1', 'chunk_3')), make_llms(draft, plan=plan),
                          max_rounds=1)
        self.assertTrue(brief.answerable)
        self.assertIn('GPT-4 training cost', brief.not_found)


if __name__ == '__main__':
    unittest.main()
