"""Dense retrieval over the frozen chunk run: build the bge-m3 index, then retrieve and look up citations.

  python -m scripts.retrieval build
  python -m scripts.retrieval query "why use multiple attention heads?" -k 5

The index is a normalized embedding matrix plus the chunk ids in row order. Chunk text, titles and
citation fields stay in chunks.jsonl and are joined by chunk_id.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK_ROOT = ROOT / 'data' / 'processed' / 'corpus_78'
INDEX_ROOT = ROOT / 'data' / 'index' / 'corpus_78'
MODEL_NAME = 'BAAI/bge-m3'
MAX_SEQ_LENGTH = 1024  # chunks are <=512 tokens; the title/section prefix adds a few dozen
EMBED_FORMAT = 'v1: "{paper_title} | {section_path joined by >}\\n{text}"'


def embed_text(chunk):
    """What the model reads for a chunk. The stored chunk text is never changed."""
    return f"{chunk['paper_title']} | {' > '.join(chunk['section_path'])}\n{chunk['text']}"


def current_run(chunk_root=CHUNK_ROOT):
    latest = json.loads((chunk_root / 'latest_run.json').read_text(encoding='utf-8'))
    return latest['run_id'], chunk_root / 'chunks' / latest['run_id'] / 'chunks.jsonl'


def read_chunks(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model(device=None):
    from sentence_transformers import SentenceTransformer
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache = ROOT / '.cache' / 'hf'
    model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=str(cache))
    model.max_seq_length = MAX_SEQ_LENGTH
    if device == 'cuda':
        model.half()
    return model


def build_index(batch_size=32, device=None, limit=None):
    run_id, chunks_path = current_run()
    chunks = read_chunks(chunks_path)
    if limit:
        chunks = chunks[:limit]
    model = load_model(device)
    texts = [embed_text(c) for c in chunks]
    lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)['input_ids']]
    too_long = sum(n > MAX_SEQ_LENGTH for n in lengths)
    if too_long:
        raise ValueError(f'{too_long} embedding inputs exceed {MAX_SEQ_LENGTH} tokens and would be truncated')
    started = time.time()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    out = INDEX_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'embeddings.npy', vectors)
    (out / 'chunk_ids.json').write_text(json.dumps([c['chunk_id'] for c in chunks]), encoding='utf-8')
    meta = {'model': MODEL_NAME, 'run_id': run_id, 'embed_format': EMBED_FORMAT, 'normalized': True,
            'count': len(chunks), 'dim': int(vectors.shape[1]), 'max_embed_tokens': max(lengths),
            'device': str(model.device), 'seconds': round(time.time() - started, 1),
            'chunks_sha256': hashlib.sha256(Path(chunks_path).read_bytes()).hexdigest(),
            'partial': bool(limit)}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out, meta


class Retriever:
    """retrieve() and lookup_citation(): the two tools the agent uses."""

    def __init__(self, index_dir=None, encoder=None):
        run_id, chunks_path = current_run()
        self.index_dir = Path(index_dir) if index_dir else INDEX_ROOT / run_id
        self.meta = json.loads((self.index_dir / 'meta.json').read_text(encoding='utf-8'))
        if self.meta['run_id'] != run_id and index_dir is None:
            raise RuntimeError(f"index was built for run {self.meta['run_id']}, current run is {run_id}")
        if self.meta.get('partial') and index_dir is None:
            raise RuntimeError('index is a partial smoke-test build; run `python -m scripts.retrieval build`')
        self.vectors = np.load(self.index_dir / 'embeddings.npy')
        self.chunk_ids = json.loads((self.index_dir / 'chunk_ids.json').read_text(encoding='utf-8'))
        if len(self.chunk_ids) != len(self.vectors):
            raise RuntimeError('chunk_ids.json and embeddings.npy disagree on row count')
        self.chunks = {c['chunk_id']: c for c in read_chunks(chunks_path)}
        self._fill_missing_citations()
        self.paper_of_row = np.array([self.chunks[i]['paper_id'] for i in self.chunk_ids])
        self._encoder = encoder
        self._model = None

    def _fill_missing_citations(self):
        """Some papers have no citation in the frozen chunk files; fill them in memory from paper_meta.json."""
        path = ROOT / 'data' / 'corpus_78' / 'paper_meta.json'
        meta = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for chunk in self.chunks.values():
            if not (chunk.get('citation') or '').strip():
                chunk['citation'] = meta.get(chunk['paper_id'], {}).get('citation') or chunk['arxiv_id']

    def papers(self):
        """{paper_id: {'citation', 'title'}} for every paper in the index; the agent uses it to pick papers to search."""
        found = {}
        for chunk in self.chunks.values():
            found.setdefault(chunk['paper_id'], {'citation': chunk['citation'], 'title': chunk['paper_title']})
        return found

    def encode(self, query):
        if self._encoder:
            vector = np.asarray(self._encoder(query), dtype=np.float32)
        else:
            if self._model is None:
                self._model = load_model()
            vector = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector / (np.linalg.norm(vector) or 1.0)

    def retrieve(self, query, k=8, paper_ids=None):
        """Top-k chunks by cosine similarity, best first. `paper_ids` limits the search to those papers."""
        scores = self.vectors @ self.encode(query)
        if paper_ids is not None:
            scores = np.where(np.isin(self.paper_of_row, list(paper_ids)), scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results = []
        for row in top:
            chunk = self.chunks[self.chunk_ids[row]]
            results.append({'chunk_id': chunk['chunk_id'], 'score': float(scores[row]),
                            'paper_id': chunk['paper_id'], 'paper_title': chunk['paper_title'],
                            'citation': chunk['citation'],
                            'section_title': chunk['section_title'], 'text': chunk['text']})
        return results

    def lookup_citation(self, chunk_id):
        """Source of truth for a cited chunk, or None if the id does not exist."""
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None
        return {'chunk_id': chunk_id, 'paper_id': chunk['paper_id'], 'arxiv_id': chunk['arxiv_id'],
                'paper_title': chunk['paper_title'], 'citation': chunk['citation'], 'version': chunk['version'],
                'section_path': chunk['section_path'], 'source_url': chunk['source_url'],
                'element_ids': [s['element_id'] for s in chunk['source_spans']],
                'previous_chunk_id': chunk['previous_chunk_id'], 'next_chunk_id': chunk['next_chunk_id'],
                'text': chunk['text']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--batch-size', type=int, default=32)
    b.add_argument('--device')
    b.add_argument('--limit', type=int, help='embed only the first N chunks (smoke test)')
    q = sub.add_parser('query')
    q.add_argument('text')
    q.add_argument('-k', type=int, default=5)
    q.add_argument('--paper', action='append')
    args = parser.parse_args(argv)
    if args.command == 'build':
        out, meta = build_index(args.batch_size, args.device, args.limit)
        print(f'index written to {out}')
        print(json.dumps(meta, indent=2))
        return 0
    retriever = Retriever()
    for rank, hit in enumerate(retriever.retrieve(args.text, args.k, args.paper), 1):
        snippet = ' '.join(hit['text'].split())[:220]
        print(f"{rank}. {hit['score']:.3f}  {hit['paper_id']} | {hit['section_title']} | {hit['chunk_id']}\n    {snippet}...")
    return 0


if __name__ == '__main__':
    sys.exit(main())
