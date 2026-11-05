"""Retriever logic checked with a cheap hashed bag-of-words encoder, so no model download is needed."""
import json
import re
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from scripts.retrieval import Retriever, current_run, embed_text, read_chunks

DIM = 512


def fake_encoder(text):
    vector = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r'[a-z0-9]+', text.lower()):
        vector[zlib.crc32(word.encode()) % DIM] += 1.0
    return vector


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_id, path = current_run()
        cls.chunks = read_chunks(path)
        cls.folder = tempfile.TemporaryDirectory()
        index = Path(cls.folder.name)
        vectors = np.stack([fake_encoder(embed_text(c)) for c in cls.chunks])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
        np.save(index / 'embeddings.npy', vectors)
        (index / 'chunk_ids.json').write_text(json.dumps([c['chunk_id'] for c in cls.chunks]), encoding='utf-8')
        (index / 'meta.json').write_text(json.dumps({'run_id': run_id, 'partial': False}), encoding='utf-8')
        cls.retriever = Retriever(index, encoder=fake_encoder)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_chunk_text_retrieves_itself(self):
        chunk = self.chunks[1234]
        hits = self.retriever.retrieve(embed_text(chunk), k=3)
        self.assertEqual(hits[0]['chunk_id'], chunk['chunk_id'])
        self.assertEqual([h['score'] for h in hits], sorted((h['score'] for h in hits), reverse=True))

    def test_paper_filter_restricts_results(self):
        chunk = self.chunks[10]
        other = next(c['paper_id'] for c in self.chunks if c['paper_id'] != chunk['paper_id'])
        hits = self.retriever.retrieve(chunk['text'], k=5, paper_ids=[other])
        self.assertEqual(len(hits), 5)
        self.assertTrue(all(h['paper_id'] == other for h in hits))

    def test_k_larger_than_available_and_unknown_paper(self):
        self.assertEqual(self.retriever.retrieve('attention', k=3, paper_ids=['no_such_paper']), [])
        self.assertEqual(len(self.retriever.retrieve('attention', k=10**6, paper_ids=[self.chunks[0]['paper_id']])),
                         sum(c['paper_id'] == self.chunks[0]['paper_id'] for c in self.chunks))

    def test_lookup_citation(self):
        chunk = self.chunks[42]
        info = self.retriever.lookup_citation(chunk['chunk_id'])
        self.assertEqual(info['text'], chunk['text'])
        self.assertEqual(info['source_url'], chunk['source_url'])
        self.assertTrue(info['element_ids'])
        self.assertIsNone(self.retriever.lookup_citation('chunk_doesnotexist'))

    def test_every_chunk_has_a_citation_even_if_the_frozen_file_left_it_empty(self):
        self.assertTrue(all(c['citation'].strip() for c in self.retriever.chunks.values()))
        empty_in_file = next(c for c in self.chunks if not c['citation'].strip())
        info = self.retriever.lookup_citation(empty_in_file['chunk_id'])
        self.assertTrue(info['citation'].strip())

    def test_embed_text_adds_title_and_section_but_keeps_text(self):
        chunk = self.chunks[0]
        text = embed_text(chunk)
        self.assertTrue(text.startswith(chunk['paper_title'] + ' | '))
        self.assertTrue(text.endswith('\n' + chunk['text']))


if __name__ == '__main__':
    unittest.main()
