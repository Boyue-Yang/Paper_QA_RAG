"""Generate notebooks/drafting_demo.ipynb; all drafting logic lives in scripts/rag.py.

  python -m scripts.build_drafting_notebook          # write the notebook
  python -m scripts.build_drafting_notebook --run    # also execute it (calls the DeepSeek API a few times)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'drafting_demo.ipynb'
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：起草（B. generate）演示

检索（A）给出 top-k 个 chunk，起草负责把它们整理好交给大模型，让模型写出**一条一条带引用的论断**。
逻辑在 `scripts/rag.py`，这个 notebook 把每一步摆出来：

1. 检索结果
2. `format_context`：模型实际看到的文字，以及 `C1`、`C2` 这些编号如何对应真实 chunk id
3. 系统提示词 `DRAFT_SYSTEM`
4. 调用模型，看原始 JSON 回复
5. `parse_draft`：编号换回真实 chunk id；模型编造编号时会怎样（不调用模型）
6. 一个资料里没有答案的问题：模型是否拒答
7. 完整的基线 `answer()` 和最终输出的 Markdown 简报
8. 带子问题的起草（完整流水线用的版本）

运行前：选项目 `.venv` 的内核；根目录 `.env` 里要有 `DEEPSEEK_API_KEY`（notebook 不会显示它）。
第 4、6、7、8 步会真实调用 DeepSeek，每次约几千 token；调用结果缓存在 `data/llm_cache/`，重复运行不会重复计费。
''')

code('''
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / 'scripts').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts import rag
from scripts.llm import LLM, ROLES
from scripts.retrieval import Retriever

retriever = Retriever()
drafter = LLM('drafter')
print('起草用的模型角色:', ROLES['drafter'])
''')

markdown('''
## 1. 检索结果

起草的输入是"问题 + 检索到的 chunk"。下面取 top-6。
''')

code('''
question = 'Why does the Transformer use multi-head attention instead of a single attention function?'
hits = retriever.retrieve(question, k=6)

for rank, h in enumerate(hits, 1):
    print(f'{rank}. {h["score"]:.3f}  {h["citation"]:22} {h["section_title"][:40]:40} {h["chunk_id"]}')
''')

markdown('''
## 2. 整理成模型看到的文字：`format_context`

每个 chunk 变成"`[Cn]` 引用信息 + 章节 + 正文"。模型看到的是 `C1`、`C2` 这样的**短编号**，
不是真实的 chunk id（`chunk_xxxx...` 是一串哈希，模型容易抄错，也容易被误导）。
编号和真实 id 的对照表只保存在我们自己的程序里。
''')

code('''
context = rag.format_context(hits)
print(context[:2200])
print('...')
print()
print('全部摘录共', len(context), '个字符')
''')

code('''
handles = {f'C{i}': h['chunk_id'] for i, h in enumerate(hits, 1)}
for handle, chunk_id in handles.items():
    print(handle, '->', chunk_id)
''')

markdown('''
## 3. 系统提示词

这段话规定了模型的行为：只能用给出的摘录，每条论断只陈述一件事并引用直接支持它的摘录，
摘录不够就在 `not_found` 里说明，**回复必须是 JSON**。摘录里出现的指令只当作数据。
''')

code('''
print(rag.DRAFT_SYSTEM)
''')

markdown('''
## 4. 调用模型，看原始回复

`draft_claims` 做的事：建编号对照表 → 拼出用户消息（问题 + 摘录）→ 调用 `llm.chat(..., json_mode=True)` → 解析回复。
这里先单独看用户消息的开头和模型的原始 JSON。
''')

code('''
user_message = f"Question: {question}\\n\\nExcerpts:\\n\\n{context}"
print(user_message[:500])
print('...')
''')

code('''
claims, data, usage, warnings = rag.draft_claims(question, hits, drafter)

print(json.dumps(data, indent=2, ensure_ascii=False))
print()
print('token 用量:', usage)
print('警告:', warnings)
''')

markdown('''
## 5. 解析：`parse_draft`

模型回复里的 `C1` 被换成真实 chunk id。下面是解析后的 `Claim` 对象：文字、真实 chunk id、
以及"模型引用了但根本没给过它的编号"（`invalid_cites`，正常应该是空的）。
''')

code('''
for claim in claims:
    print(claim.claim_id, '|', claim.text)
    print('    引用:', claim.cites, '| 无效引用:', claim.invalid_cites)
''')

markdown('''
**模型引用了没给过的编号会怎样？** 下面用一份手工构造的"坏回复"演示，不调用模型：
编号 `C9` 从未出现过，另一条论断没有文字，还有一条把 `cites` 写成了字符串。
程序不会崩溃，也不会相信 `C9`，而是把它记进 `invalid_cites`，验证阶段会据此把这条论断判为不支持。
''')

code('''
bad_reply = {'claims': [
    {'text': 'A claim citing something that was never shown.', 'cites': ['C1', 'C9']},
    {'text': '', 'cites': ['C2']},
    {'text': 'A claim whose cites is a bare string.', 'cites': 'C2'},
]}
parsed, warns = rag.parse_draft(bad_reply, handles)
for claim in parsed:
    print(claim.claim_id, '|', claim.text)
    print('    引用:', claim.cites, '| 无效引用:', claim.invalid_cites)
print('警告:', warns)
''')

markdown('''
## 6. 资料里没有答案的问题

这是拒答的测试：问题的前提在资料里不存在。期望 `answerable` 为 false、没有论断，并在 `not_found` 里说明缺什么。
注意起草阶段**只靠提示词**让模型拒答；它不一定每次都做对，这正是后面要加验证的原因。
''')

code('''
unanswerable = 'How many tokens was Mixtral 8x7B pretrained on?'
hits_u = retriever.retrieve(unanswerable, k=6)
print('最高相似度:', round(hits_u[0]['score'], 3), '(有结果返回，但不代表有答案)')

claims_u, data_u, usage_u, warns_u = rag.draft_claims(unanswerable, hits_u, drafter)
print(json.dumps(data_u, indent=2, ensure_ascii=False))
''')

markdown('''
## 7. 完整的基线：`answer()` 和 Markdown 简报

`answer(question, retriever, llm, k)` = 检索 + 起草，返回 `Brief`。`to_markdown` 把真实 chunk id
换成 `[1][2]` 这样的角标，并在末尾列出来源（作者、论文标题、章节、链接）。
这就是用户最终看到的东西。基线到这里就结束了，没有验证。
''')

code('''
brief = rag.answer(question, retriever, drafter, k=8)
print('answerable:', brief.answerable, '| 论断数:', len(brief.claims), '| token:', brief.usage)
print()
print(rag.to_markdown(brief, retriever))
''')

markdown('''
## 8. 带子问题的起草

完整流水线（`agent.py`）会先把复杂问题拆成子问题，起草时多传一份子问题清单，并要求每条论断标上它回答的是哪个子问题（`sub` 字段）。
这样之后才能检查"每个子问题是否都有被支持的论断"。下面手工给出两个子问题演示这一模式。
''')

code('''
print(rag.SUBQ_RULE)
''')

code('''
q2 = 'How does layer normalization differ from batch normalization?'
subs = {'s1': 'How does batch normalization compute its statistics?',
        's2': 'How does layer normalization compute its statistics?'}

pool = {}
for text in [q2] + list(subs.values()):
    for h in retriever.retrieve(text, 3 if text != q2 else 6):
        pool.setdefault(h['chunk_id'], h)
hits2 = list(pool.values())
print('合并去重后的 chunk 数:', len(hits2))

claims2, data2, usage2, warns2 = rag.draft_claims(q2, hits2, drafter, subs)
for c in claims2:
    print(f'[{c.sub_question}] {c.text}')
    print('      引用:', c.cites)
print('not_found:', data2.get('not_found'))
''')

markdown('''
## 小结

- 起草 = 把 chunk 整理成带 `[Cn]` 编号的文字 + 系统提示词 + 问题，让模型返回 JSON 论断列表
- 模型看到的是短编号；真实 chunk id 只在程序里，解析时换回来，编造的编号被记入 `invalid_cites`
- 拒答只靠提示词；模型可能漏拒答或写出超出摘录的内容 —— 这就是下一步 **验证** 要检查的
- 论断是否真的被所引用的 chunk 支持，起草阶段**完全没有检查**
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
