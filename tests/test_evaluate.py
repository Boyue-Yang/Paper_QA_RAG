"""Metric code checked on tiny hand-made questions and scripted judges (no network, no GPU)."""
import unittest
from types import SimpleNamespace

from scripts.evaluate import (evidence_hits, key_point_reach, retrieval_metrics, score_question, summarize)

QUESTION = {
    'question_id': 'q1', 'type': 'multi_paper', 'question': 'Q?',
    'evidence': [{'evidence_id': 'e1', 'matched_chunk_ids': ['a']},
                 {'evidence_id': 'e2', 'matched_chunk_ids': ['b', 'c']},
                 {'evidence_id': 'e3', 'matched_chunk_ids': ['d']}],
    'key_points': [{'point_id': 'k1', 'text': 'first', 'evidence_ids': ['e1', 'e2']},
                   {'point_id': 'k2', 'text': 'second', 'evidence_ids': ['e3']}],
}
TEXTS = {'a': 'Alpha is large.', 'b': 'Beta is small.', 'x': 'Unrelated text.'}


class FakeRetriever:
    def lookup_citation(self, chunk_id):
        if chunk_id not in TEXTS:
            return None
        return {'chunk_id': chunk_id, 'paper_id': 'p', 'paper_title': 'P', 'citation': 'A (2020)',
                'section_path': ['S'], 'source_url': '', 'text': TEXTS[chunk_id]}


class ScriptedJudge:
    def __init__(self, handler):
        self.handler = handler

    def chat(self, system, user, json_mode=False):
        return SimpleNamespace(parsed=self.handler(system, user), usage={})


def judge_handler(system, user):
    if system.startswith('You grade how completely'):
        return {'points': [{'point_id': 'k1', 'coverage': 'covered'}, {'point_id': 'k2', 'coverage': 'partial'}]}
    claim = user.split('\n')[0]
    quote = 'Alpha is large.' if 'Alpha' in claim else 'Unrelated text.'
    return {'verdict': 'supported' if 'Alpha' in claim else 'unsupported', 'quote': quote, 'reason': ''}


def record(claims, retrieved, answerable=True):
    brief = {'question': 'Q?', 'answerable': answerable, 'not_found': '', 'usage': {'total_tokens': 10},
             'retrieved': [{'chunk_id': c} for c in retrieved],
             'claims': [{'claim_id': f'c{i}', 'text': t, 'cites': cites, 'invalid_cites': []}
                        for i, (t, cites) in enumerate(claims, 1)]}
    return {'question_id': 'q1', 'arm': 'rules', 'brief': brief, 'seconds': 1.0}


class RetrievalTests(unittest.TestCase):
    def test_evidence_hit_needs_any_matched_chunk(self):
        self.assertEqual(evidence_hits(QUESTION, ['c']), {'e1': False, 'e2': True, 'e3': False})

    def test_key_point_reach_counts_a_point_once_if_any_evidence_hits(self):
        self.assertEqual(key_point_reach(QUESTION, ['c']), 0.5)
        self.assertEqual(key_point_reach(QUESTION, ['a', 'b', 'd']), 1.0)
        self.assertEqual(key_point_reach(QUESTION, ['x']), 0.0)
        self.assertIsNone(key_point_reach({**QUESTION, 'key_points': []}, ['a']))

    def test_metrics_at_k_and_over_the_pool(self):
        m = retrieval_metrics(QUESTION, ['x', 'a'] + ['y'] * 6 + ['d'], k=8)
        self.assertEqual(m['hit@8'], 1.0)
        self.assertEqual(m['mrr'], 0.5)
        self.assertAlmostEqual(m['evid_recall@8'], 1 / 3)
        self.assertEqual(m['kp_reach@8'], 0.5)
        self.assertEqual(m['pool_kp_reach'], 1.0)    # 'd' sits at rank 9, outside @8 but inside the pool
        self.assertEqual(m['pool_size'], 9)
        perfect = retrieval_metrics(QUESTION, ['a', 'b', 'c', 'd'], k=8)
        self.assertAlmostEqual(perfect['ndcg@8'], 1.0)


class ScoreTests(unittest.TestCase):
    def test_claim_scores_and_completeness(self):
        rec = record([('Alpha is large.', ['a']), ('Something else.', ['x'])], ['a', 'x'])
        score = score_question(QUESTION, rec, FakeRetriever(), ScriptedJudge(judge_handler))
        self.assertEqual(score['supported_rate'], 0.5)
        self.assertEqual(score['unsupported_rate'], 0.5)
        self.assertEqual(score['citation_correct'], 0.5)    # claim 2 cites a real chunk but not gold, and fails the judge
        self.assertEqual(score['cited_kp_reach'], 0.5)      # only 'a' (k1) is cited
        self.assertEqual(score['completeness'], 0.75)       # covered 1.0 + partial 0.5, averaged

    def test_gold_citation_counts_even_if_the_judge_disagrees(self):
        rec = record([('Beta claim worded loosely.', ['b'])], ['b'])
        score = score_question(QUESTION, rec, FakeRetriever(), ScriptedJudge(judge_handler))
        self.assertEqual(score['supported_rate'], 0.0)
        self.assertEqual(score['citation_correct'], 1.0)

    def test_nonexistent_citation_is_never_correct(self):
        rec = record([('Alpha is large.', ['ghost'])], ['a'])
        score = score_question(QUESTION, rec, FakeRetriever(), ScriptedJudge(judge_handler))
        self.assertEqual(score['citation_correct'], 0.0)

    def test_no_judge_still_gives_retrieval_and_abstention(self):
        rec = record([], ['a'], answerable=False)
        score = score_question(QUESTION, rec)
        self.assertTrue(score['abstained'])
        self.assertTrue(score['false_abstain'])
        self.assertNotIn('supported_rate', score)
        self.assertEqual(score['hit@8'], 1.0)

    def test_unanswerable_question_scores_abstention_only(self):
        question = {'question_id': 'q9', 'type': 'unanswerable', 'question': 'Q?', 'evidence': [], 'key_points': []}
        score = score_question(question, record([], ['a'], answerable=False))
        self.assertTrue(score['abstained'])
        self.assertNotIn('hit@8', score)

    def test_summarize_groups_by_type_and_ignores_errors(self):
        scores = [{'type': 'single_fact', 'abstained': False, 'hit@8': 1.0},
                  {'type': 'multi_paper', 'abstained': False, 'hit@8': 0.0},
                  {'type': 'unanswerable', 'abstained': True},
                  {'type': 'comparison', 'error': 'boom'}]
        out = summarize(scores)
        self.assertEqual(out['answerable']['hit@8'], 0.5)
        self.assertEqual((out['answerable']['n'], out['answerable']['errors']), (3, 1))
        self.assertEqual(out['unanswerable']['abstained'], 1.0)
        self.assertIsNone(out['unanswerable']['hit@8'])


if __name__ == '__main__':
    unittest.main()
