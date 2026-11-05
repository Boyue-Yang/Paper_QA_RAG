"""Baseline RAG plumbing checked with a fake retriever and fake LLM: handles, parsing and rendering."""
import unittest
from types import SimpleNamespace

from scripts.rag import answer, format_context, parse_draft, to_markdown

HITS = [{'chunk_id': f'chunk_{i}', 'score': 0.9 - i / 10, 'paper_id': f'p{i}', 'paper_title': f'Paper {i}',
         'citation': f'Author{i} (2020)', 'section_title': f'S{i}', 'text': f'text of excerpt {i}'} for i in range(1, 4)]


class FakeRetriever:
    def __init__(self):
        self.asked = None

    def retrieve(self, query, k):
        self.asked = (query, k)
        return HITS[:k]

    def lookup_citation(self, chunk_id):
        n = chunk_id.split('_')[1]
        return {'citation': f'Author{n} (2020)', 'paper_title': f'Paper {n}', 'section_path': [f'S{n}'],
                'source_url': f'https://example.invalid/{n}'}


class FakeLLM:
    def __init__(self, parsed):
        self.parsed = parsed
        self.prompt = None

    def chat(self, system, user, json_mode=False):
        self.prompt = (system, user, json_mode)
        return SimpleNamespace(parsed=self.parsed, usage={'total_tokens': 7})


HANDLES = {'C1': 'chunk_1', 'C2': 'chunk_2', 'C3': 'chunk_3'}


class RagTests(unittest.TestCase):
    def test_context_labels_each_excerpt_with_a_handle(self):
        text = format_context(HITS)
        self.assertIn('[C1] Author1 (2020), "Paper 1", section: S1\ntext of excerpt 1', text)
        self.assertIn('[C3]', text)

    def test_handles_map_to_chunk_ids_and_unknown_handles_are_recorded(self):
        claims, warnings = parse_draft({'claims': [
            {'text': 'A is true.', 'cites': ['C2', '[C1]', 'C2']},
            {'text': 'B is true.', 'cites': ['C9']},
            {'text': 'C is true.', 'cites': 'C3'},
            {'text': '  '}, 'junk']}, HANDLES)
        self.assertEqual([c.cites for c in claims], [['chunk_2', 'chunk_1'], [], ['chunk_3']])
        self.assertEqual(claims[1].invalid_cites, ['C9'])
        self.assertEqual([c.claim_id for c in claims], ['c1', 'c2', 'c3'])
        self.assertEqual(len(warnings), 2)

    def test_missing_claims_list_does_not_raise(self):
        self.assertEqual(parse_draft({'oops': 1}, HANDLES), ([], ['reply had no "claims" list']))
        self.assertEqual(parse_draft(None, HANDLES)[0], [])

    def test_answer_end_to_end(self):
        retriever = FakeRetriever()
        llm = FakeLLM({'answerable': True, 'claims': [{'text': 'X holds.', 'cites': ['C1']}], 'not_found': 'Y'})
        brief = answer('What holds?', retriever, llm, k=2)
        self.assertEqual(retriever.asked, ('What holds?', 2))
        self.assertTrue(llm.prompt[2])  # json mode on
        self.assertIn('Question: What holds?', llm.prompt[1])
        self.assertNotIn('[C3]', llm.prompt[1])  # only the k=2 excerpts were shown
        self.assertEqual(brief.claims[0].cites, ['chunk_1'])
        self.assertEqual([r['chunk_id'] for r in brief.retrieved], ['chunk_1', 'chunk_2'])
        self.assertNotIn('text', brief.retrieved[0])
        self.assertTrue(brief.answerable)
        self.assertEqual(brief.usage, {'total_tokens': 7})

    def test_answerable_without_claims_is_not_answerable(self):
        brief = answer('q', FakeRetriever(), FakeLLM({'answerable': True, 'claims': [], 'not_found': ''}))
        self.assertFalse(brief.answerable)
        self.assertTrue(brief.warnings)

    def test_refusal_renders_without_sources(self):
        retriever = FakeRetriever()
        brief = answer('q', retriever, FakeLLM({'answerable': False, 'claims': [], 'not_found': 'nothing relevant'}))
        markdown = to_markdown(brief, retriever)
        self.assertIn('No supported answer', markdown)
        self.assertIn('**Not found:** nothing relevant', markdown)
        self.assertNotIn('## Sources', markdown)

    def test_markdown_numbers_sources_in_order_of_first_citation(self):
        retriever = FakeRetriever()
        brief = answer('q', retriever, FakeLLM({'answerable': True, 'not_found': '', 'claims': [
            {'text': 'First.', 'cites': ['C3']}, {'text': 'Second.', 'cites': ['C1', 'C3']}, {'text': 'Third.'}]}))
        markdown = to_markdown(brief, retriever)
        self.assertIn('- First. [1]', markdown)
        self.assertIn('- Second. [2][1]', markdown)
        self.assertIn('- Third.  [no citation]', markdown)
        self.assertIn('1. Author3 (2020), *Paper 3*, S3 (https://example.invalid/3)', markdown)


if __name__ == '__main__':
    unittest.main()
