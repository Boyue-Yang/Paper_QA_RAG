"""Generate notebooks/retrieval_embedding_demo.ipynb; all retrieval logic lives in scripts/retrieval.py.

  python -m scripts.build_retrieval_notebook          # write the notebook
  python -m scripts.build_retrieval_notebook --run    # also execute it so the outputs are saved in the file
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'retrieval_embedding_demo.ipynb'
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：向量化与检索演示

这个 notebook 演示第一阶段（A. retrieve）里"文本变成向量、再用向量找 chunk"的全过程。
所有逻辑在 `scripts/retrieval.py`，这里只调用它，并把每一步的中间结果摆出来。

内容：
1. 一个 chunk 在送进模型之前长什么样（`embed_text`）
2. 向量存在哪里：`embeddings.npy`、`chunk_ids.json`、`meta.json`
3. 存下来的向量和重新编码的结果是否一致
4. 手工做一遍检索（query 向量 × 矩阵），并和 `Retriever.retrieve` 对照
5. 几个示例查询
6. 限定论文范围、`lookup_citation`
7. 在评测集 dev 上看检索的真实表现，以及一个失败例子

运行前：选择项目 `.venv` 的 Python 内核。索引已经建好，不需要重建（重建命令：`python -m scripts.retrieval build`，约 4 分钟）。
第 4 步起会加载 bge-m3 模型（有 GPU 时用 GPU，约十几秒）。
''')

code('''
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / 'scripts').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import retrieval as rt

print('项目根目录:', ROOT)
''')

markdown('''
## 1. 一个 chunk 送进模型之前长什么样

chunk 文件是冻结的分块结果（5755 个）。编码时不是只用正文，而是把"论文标题 | 章节路径"拼在正文前面，
这样模型能看到这段话属于哪篇论文的哪一节。拼接只发生在编码那一刻，存储的 chunk 原文没有被改动。
''')

code('''
run_id, chunks_path = rt.current_run()
chunks = rt.read_chunks(chunks_path)
by_id = {c['chunk_id']: c for c in chunks}
print('分块 run id:', run_id)
print('chunk 总数 :', len(chunks))

chunk = chunks[100]
print()
print('一个 chunk 的字段:', list(chunk))
print()
for key in ('chunk_id', 'paper_id', 'paper_title', 'section_path', 'section_title'):
    print(f'{key:14}', chunk[key])
''')

code('''
text_for_model = rt.embed_text(chunk)
print(text_for_model[:700])
print('...')
print()
print('正文字符数:', len(chunk['text']), '| 送进模型的字符数:', len(text_for_model))
''')

markdown('''
## 2. 向量存在哪里

索引目录名就是分块 run id，所以换了分块就不会误用旧索引。三个文件：

- `embeddings.npy`：N×1024 的矩阵，**第 i 行就是第 i 个 chunk 的向量**，矩阵里没有 id
- `chunk_ids.json`：长度 N 的列表，第 i 项是第 i 行对应的 chunk_id（靠下标对应）
- `meta.json`：模型、维度、是否归一化、拼接格式、chunk 文件哈希
''')

code('''
index_dir = rt.INDEX_ROOT / run_id
for path in sorted(index_dir.iterdir()):
    print(f'{path.name:18}{path.stat().st_size / 1e6:8.2f} MB')

print()
print(json.dumps(json.loads((index_dir / 'meta.json').read_text(encoding='utf-8')), indent=2, ensure_ascii=False))
''')

code('''
vectors = np.load(index_dir / 'embeddings.npy')
chunk_ids = json.loads((index_dir / 'chunk_ids.json').read_text(encoding='utf-8'))

print('矩阵形状:', vectors.shape, vectors.dtype)
print('chunk_ids 长度:', len(chunk_ids), '| 前 3 个:', chunk_ids[:3])

norms = np.linalg.norm(vectors, axis=1)
print(f'每行向量的长度: 最小 {norms.min():.6f}, 最大 {norms.max():.6f}  (约等于 1，说明已归一化；微小偏差来自 fp16)')

row = 100
print()
print(f'第 {row} 行 -> chunk_id {chunk_ids[row]} -> {by_id[chunk_ids[row]]["paper_title"]} / {by_id[chunk_ids[row]]["section_title"]}')
print('向量前 8 个数:', np.round(vectors[row][:8], 4))
''')

markdown('''
因为每个向量长度都是 1，**余弦相似度 = 点积**，所以检索只需要一次矩阵乘法，不需要向量数据库。
''')

markdown('''
## 3. 存下来的向量和重新编码一致吗

随机抽 10 个 chunk，用同一个模型重新编码，和存储的向量算点积。应该非常接近 1
（GPU 上用 fp16，所以不会是精确的 1.0000）。
''')

code('''
retriever = rt.Retriever()
model = retriever._model or rt.load_model()
retriever._model = model
print('模型所在设备:', model.device)

rng = np.random.default_rng(0)
rows = rng.choice(len(chunks), 10, replace=False)
fresh = model.encode([rt.embed_text(by_id[chunk_ids[r]]) for r in rows], normalize_embeddings=True, convert_to_numpy=True)
same = (fresh * vectors[rows]).sum(axis=1)
print('重新编码与存储向量的相似度:', np.round(same, 4))
''')

markdown('''
## 4. 手工做一遍检索

步骤：
1. query 用同一个模型编码成 1024 维向量（同样归一化）
2. 向量矩阵 × query 向量，得到 5755 个相似度
3. 取最大的 k 个，行号换成 chunk_id，再取出正文
''')

code('''
query = 'Why does the Transformer use multi-head attention?'

q = retriever.encode(query)
print('query 向量形状:', q.shape, '| 长度:', round(float(np.linalg.norm(q)), 4))

scores = vectors @ q
print('相似度个数:', scores.shape, f'| 最高 {scores.max():.3f} | 中位数 {np.median(scores):.3f} | 最低 {scores.min():.3f}')

top_rows = np.argsort(-scores)[:5]
for rank, r in enumerate(top_rows, 1):
    c = by_id[chunk_ids[r]]
    print(f'{rank}. {scores[r]:.3f}  {c["paper_title"][:45]:45} | {c["section_title"][:35]}')
''')

code('''
manual = [chunk_ids[r] for r in top_rows]
official = [h['chunk_id'] for h in retriever.retrieve(query, k=5)]
print('手工结果与 Retriever.retrieve 一致:', manual == official)
''')

markdown('''
## 5. 几个示例查询

`retrieve` 返回的每一项含：`chunk_id`、`score`、`paper_id`、`paper_title`、`citation`、`section_title`、`text`。
下面换几种问法，只看分数、论文、章节和正文开头。
''')

code('''
def show(query, k=5, **kwargs):
    print('Q:', query)
    for rank, h in enumerate(retriever.retrieve(query, k, **kwargs), 1):
        print(f'  {rank}. {h["score"]:.3f} {h["paper_title"][:40]:40} | {h["section_title"][:28]:28} | {h["text"][:70]!r}')
    print()

show('What learning rate schedule does the original Transformer use?')
show('How does layer normalization differ from batch normalization?')
show('Which paper introduced low-rank adaptation for fine-tuning large language models?')
''')

markdown('''
最后一个是问"哪篇论文"，检索结果的论文标题可以直接回答；前两个则要看具体 chunk 里有没有所问的事实。

**无关问题**也会返回 top-k，只是分数更低。相似度没有一个通用的阈值能区分"有答案"和"没答案"，
所以后面是否拒答靠起草和验证，而不是靠分数。
''')

code('''
show('How many tokens was Mixtral 8x7B pretrained on?')
show('What is the best recipe for chocolate cake?')
''')

markdown('''
## 6. 限定论文范围、查引用信息

`retrieve(..., paper_ids=[...])` 只在指定论文里找。`lookup_citation(chunk_id)` 是第二个工具，
按 chunk_id 取回引用信息，以及相邻 chunk 的 id；id 不存在时返回 `None`，验证器靠这一点发现编造的引用。
''')

code('''
papers = sorted({c['paper_id'] for c in chunks})
print('论文数:', len(papers), '| 例如:', papers[:3])

target = by_id[chunk_ids[top_rows[0]]]['paper_id']
print('只在', target, '里检索:')
show(query, k=3, paper_ids=[target])
''')

code('''
info = retriever.lookup_citation(chunk_ids[top_rows[0]])
print({k: (v if k != 'text' else v[:80] + '...') for k, v in info.items()})
print()
print('不存在的 id ->', retriever.lookup_citation('chunk_does_not_exist'))
''')

markdown('''
## 7. 在评测集 dev 上看检索的真实表现

评测题的金标是原文里的逐字引文，已经对齐到 chunk id（`matched_chunk_ids`）。
"要点可达率" = 每个要点只要有一条证据所在的 chunk 被检索到就算命中，再对要点取平均。
只用检索，不调用大模型。
''')

code('''
from scripts.evaluate import default_questions, gold_chunks, key_point_reach, load_questions

questions = [q for q in load_questions(default_questions(), 'dev') if q['type'] != 'unanswerable']
print('dev 可回答题:', len(questions))

reach = {}
ranks = {}
for q in questions:
    got = [h['chunk_id'] for h in retriever.retrieve(q['question'], 20)]
    gold = gold_chunks(q)
    ranks[q['question_id']] = next((i for i, c in enumerate(got, 1) if c in gold), None)
    for k in (5, 8, 20):
        reach.setdefault((q['type'], k), []).append(key_point_reach(q, got[:k]))

print()
print(f'{"题型":14}{"@5":>8}{"@8":>8}{"@20":>8}')
for t in ('single_fact', 'multi_paper', 'comparison'):
    print(f'{t:14}' + ''.join(f'{np.mean(reach[(t, k)]):8.2f}' for k in (5, 8, 20)))
''')

markdown('''
单事实题检索基本够用，多论文和对比题在 top-8 里覆盖不全。这正是后面把问题拆成子问题、
再补检索的动机。
''')

code('''
worst = min(questions, key=lambda q: (key_point_reach(q, [h["chunk_id"] for h in retriever.retrieve(q["question"], 8)]) or 0, q["question_id"]))
print(worst['question_id'], worst['type'])
print('问题:', worst['question'])
print('金标证据:')
for e in worst['evidence'][:3]:
    print('  -', e['paper_id'], '|', e['section_title'], '|', e['quote'][:90])

hits = retriever.retrieve(worst['question'], 20)
gold = gold_chunks(worst)
print()
print('top-8 检索到的:')
for rank, h in enumerate(hits[:8], 1):
    print(f'  {rank}. {h["score"]:.3f} {"[金标]" if h["chunk_id"] in gold else "      "} {h["paper_title"][:38]:38} | {h["section_title"][:28]}')
first = next((i for i, h in enumerate(hits, 1) if h['chunk_id'] in gold), None)
print('第一个金标 chunk 的排名 (top-20 内):', first)
''')

markdown('''
## 小结

- 向量存成 `embeddings.npy`（按行号对应）+ `chunk_ids.json`（行号 → chunk_id）
- 编码用 bge-m3，输入是"标题 | 章节路径 + 正文"，向量已归一化，余弦相似度就是点积
- 检索 = query 编码 → 矩阵乘法 → 取 top-k → 按 chunk_id 取正文
- 已知限制：只用稠密向量，对换了说法的问题、对比题、多论文题召回不全；相似度分数不能当拒答依据
''')

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--run' in argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
