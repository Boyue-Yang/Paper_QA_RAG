"""The model-driven agent, checked with a scripted controller (no network)."""
import unittest

from scripts.adaptive_agent import format_decisions, run_adaptive
from scripts.llm import LLMError
from tests.test_agent import FakeRetriever, ScriptedLLM, draft_two_claims, make_llms


class Retriever(FakeRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paper_filters = []

    def retrieve(self, query, k=8, paper_ids=None):
        self.paper_filters.append(paper_ids)
        return super().retrieve(query, k)

    def papers(self):
        return {f'p{n}': {'citation': f'A{n} (2020)', 'title': f'Paper {n}'} for n in range(1, 5)}


def step(action, thought='because', notes=None, **args):
    return {'thought': thought, 'action': action, 'args': args, **({'notes': notes} if notes else {})}


def build(*steps, revise=None, draft=draft_two_claims):
    llms = make_llms(draft, revise=revise)
    queue = iter(steps)
    llms['controller'] = ScriptedLLM(lambda system, user: next(queue))
    return llms


REWRITE = lambda user: {'action': 'rewrite', 'text': 'The base model reaches 28.4 BLEU on WMT14 English-to-German.',
                        'cites': ['C1']}


class AdaptiveTests(unittest.TestCase):
    def test_the_controller_picks_every_action_and_each_decision_is_recorded(self):
        llms = build(step('draft'), step('verify'), step('revise', claim='c2'), step('verify'),
                     step('finish', answerable=True, not_found=''), revise=REWRITE)
        brief = run_adaptive('How good is the base model?', Retriever(), llms)
        self.assertEqual([r['action'] for r in brief.trace], ['search', 'draft', 'verify', 'revise', 'verify', 'finish'])
        self.assertEqual(brief.trace[0]['thought'], 'initial retrieval (automatic)')
        self.assertTrue(all('thought' in r and 'result' in r for r in brief.trace))
        self.assertIn('unsupported', brief.trace[2]['result'])
        self.assertTrue(brief.answerable)
        self.assertEqual({c.verdict['verdict'] for c in brief.claims}, {'supported'})
        self.assertEqual(len(brief.claims), 2)

    def test_a_rewrite_that_repeats_an_existing_claim_is_flagged_to_the_controller(self):
        llms = build(step('draft'), step('verify'), step('revise', claim='c2'),
                     step('finish', answerable=True), revise=REWRITE)
        brief = run_adaptive('How good is the base model?', Retriever(), llms)
        self.assertIn('duplicate of c1', brief.trace[3]['result'])

    def test_verification_cannot_be_skipped(self):
        llms = build(step('draft'), step('finish', answerable=True))
        brief = run_adaptive('How good is the base model?', Retriever(), llms)
        self.assertEqual(brief.trace[-1]['action'], 'verify')
        self.assertIn('automatic', brief.trace[-1]['thought'])
        self.assertEqual([c.text for c in brief.claims], ['The base model reaches 28.4 BLEU.'])
        self.assertTrue(any('unsupported claims' in w for w in brief.warnings))

    def test_searching_inside_chosen_papers_and_unknown_papers(self):
        retriever = Retriever()
        llms = build(step('search', query='dropout rate', papers=['p3']),
                     step('search', query='x', papers=['nope']), step('finish', answerable=False))
        brief = run_adaptive('q', retriever, llms)
        self.assertIn(['p3'], retriever.paper_filters)
        self.assertTrue(brief.trace[2]['result'].startswith('ERROR: unknown paper ids'))

    def test_invalid_actions_come_back_as_errors_and_three_in_a_row_stop_the_run(self):
        llms = build(step('fly'), step('fly'), step('fly'), step('finish'))
        brief = run_adaptive('q', Retriever(), llms)
        self.assertEqual([r['result'][:6] for r in brief.trace[1:]], ['ERROR:'] * 3)
        self.assertTrue(any('repeated invalid actions' in w for w in brief.warnings))
        self.assertFalse(brief.answerable)

    def test_the_agent_can_declare_a_question_unanswerable_even_if_some_claims_pass(self):
        llms = build(step('draft'), step('verify'),
                     step('finish', answerable=False, not_found='The premise does not match the papers.'))
        brief = run_adaptive('q', Retriever(), llms)
        self.assertFalse(brief.answerable)
        self.assertEqual(brief.claims, [])
        self.assertEqual(brief.not_found, 'The premise does not match the papers.')

    def test_a_revise_needs_a_verified_claim_and_draft_needs_fresh_evidence(self):
        llms = build(step('draft'), step('revise', claim='c2'), step('draft'), step('finish', answerable=True))
        brief = run_adaptive('q', Retriever(), llms)
        self.assertIn('verify it first', brief.trace[2]['result'])
        self.assertIn('no undrafted evidence', brief.trace[3]['result'])

    def test_the_step_budget_closes_a_run_that_never_finishes(self):
        llms = make_llms(draft_two_claims)
        llms['controller'] = ScriptedLLM(lambda system, user: step('search', query='again'))
        brief = run_adaptive('q', Retriever(), llms, max_steps=3)
        self.assertEqual(len(brief.trace), 4)    # the automatic search plus three decisions
        self.assertTrue(any('did not finish by itself' in w for w in brief.warnings))

    def test_controller_failure_still_returns_a_brief(self):
        def boom(system, user):
            raise LLMError('down')
        llms = make_llms(draft_two_claims)
        llms['controller'] = ScriptedLLM(boom)
        brief = run_adaptive('q', Retriever(), llms)
        self.assertFalse(brief.answerable)
        self.assertTrue(any('controller call failed' in w for w in brief.warnings))

    def test_notes_are_shown_back_at_the_next_step_and_the_log_formats(self):
        llms = build(step('search', query='x', notes='PLAN-X: part 1 open'), step('finish', answerable=False))
        brief = run_adaptive('q', Retriever(), llms)
        self.assertIn('PLAN-X', llms['controller'].calls[1][1])
        self.assertIn('step 1 | search', format_decisions(brief.trace))


if __name__ == '__main__':
    unittest.main()
