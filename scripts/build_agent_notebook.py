"""Generate notebooks/agent_demo.ipynb; all control logic lives in scripts/agent.py.

  python -m scripts.build_agent_notebook          # write the notebook
  python -m scripts.build_agent_notebook --run    # also execute it (several DeepSeek calls, a few minutes)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'agent_demo.ipynb'
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：完整流水线（验证 → 修订 → 完成）演示

基线 RAG 到"起草"就结束了。完整流水线在后面加了一个循环：

```
拆子问题 → 检索 → 起草 → 验证每条论断 ─┬─ 全部通过且每个子问题都有论断 → 完成
                                      └─ 否则 → 修订 → 再验证（最多 N 轮）→ 完成
```

控制这个循环的是**普通的 Python 规则**（`scripts/agent.py` 的 `run_agent`），不是大模型自己决定下一步。
大模型只在四个地方被调用：拆子问题、起草、改写一条论断、判官验证。

控制器按失败的**类型**选动作：

| 情况 | 动作 |
|---|---|
| 一条论断没通过验证 | **修订**：用这条论断当查询再检索一次，让模型改写成原文真正说的，或者删掉 |
| 一个子问题没有任何通过的论断 | **补充起草**：检索之前没看过的 chunk，只针对这个子问题重新起草 |
| 轮数用完 | 停止。最终简报只保留通过验证的论断；没找到的写进"未找到" |

本 notebook 的内容：
1. 拆子问题：简单问题和复合问题分别怎样
2. 检索并合并去重
3. 一次完整运行，并展示每一轮的记录（trace）。为了稳定地看到修订，第一次起草会被**故意注入两条错误论断**
4. 同一个问题分别用"验证后直接删除（rounds=0）"和"完整修订（rounds=2）"跑，对比
5. 子问题没有答案时：补充起草，最后写进"未找到"

运行前：选项目 `.venv` 的内核，`.env` 里要有 `DEEPSEEK_API_KEY`。会真实调用 DeepSeek 十几到几十次，整个 notebook 需要几分钟；调用结果有缓存。
''')

code('''
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / 'scripts').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts import agent, rag
from scripts.llm import LLM
from scripts.retrieval import Retriever

retriever = Retriever()
llms = {'drafter': LLM('drafter'), 'verifier': LLM('verifier')}
print('常量:', {k: getattr(agent, k) for k in ('K_SUB', 'K_REPAIR', 'K_REFILL', 'MAX_SUBQUESTIONS')})
''')

markdown('''
## 1. 拆子问题：`agent.plan`

提示词规定：问题问了好几件事、要比较、或者要从不同论文找事实，就拆成最多 4 个能独立查找的子问题；
只是一个集中的问题，原样返回。拆子问题的目的是**让检索和覆盖检查有抓手**。
''')

code('''
for q in ['Why does the Transformer use multi-head attention instead of a single attention function?',
          'How does layer normalization differ from batch normalization, and which one does the Transformer use?',
          'What BLEU does the base Transformer reach on English-to-German, and how many tokens was Mixtral 8x7B pretrained on?']:
    subs, usage = agent.plan(q, llms['drafter'])
    print('Q:', q)
    for sid, text in subs.items():
        print(f'   {sid}: {text}')
    print()
''')

markdown('''
## 2. 检索：`agent.gather`

先用原问题取 top-8，再对每个子问题各取 3 个，合并去重（只有一个子问题时就只有主检索）。
这样复合问题的每个部分都能保证检索到一些相关 chunk，而不是被主问题的向量"平均"掉。
''')

code('''
q2 = 'How does layer normalization differ from batch normalization, and which one does the Transformer use?'
subs2, _ = agent.plan(q2, llms['drafter'])
main_only = [h['chunk_id'] for h in retriever.retrieve(q2, 8)]
pool = agent.gather(q2, subs2, retriever, 8)
print('只用原问题检索:', len(main_only), '个 chunk')
print('加上子问题检索、合并去重后:', len(pool), '个 chunk，其中新增', len({h['chunk_id'] for h in pool} - set(main_only)), '个')
for h in pool:
    mark = ' ' if h['chunk_id'] in main_only else '+'
    print(f'  {mark} {h["citation"]:24} {h["section_title"][:40]}')
''')

markdown('''
## 3. 一次完整运行，以及每一轮的记录

为了稳定地看到"修订"，我写了一个包装器 `InjectingLLM`：在**第一次起草**的回复里插入两条错误论断，其余完全不动。

- 错误 1：把真实数字 0.9 BLEU 改成 **1.9 BLEU**（真实论断被改了一个数字，改写后应该能救回）
- 错误 2：一个**完全编造**的句子（原文里什么都没有，应该被删掉）

这只用于演示，不会影响 `scripts/` 里的任何代码。
''')

code('''
class InjectingLLM:
    """Wraps the real drafter. On the first drafting call only, appends wrong claims to the reply."""

    def __init__(self, inner, bad_claims):
        self.inner, self.bad_claims, self.done = inner, bad_claims, False

    def chat(self, system, user, json_mode=False):
        result = self.inner.chat(system, user, json_mode)
        if not self.done and system.startswith('You write short research briefs') and isinstance(result.parsed, dict):
            self.done = True
            result.parsed['answerable'] = True
            result.parsed.setdefault('claims', []).extend(self.bad_claims)
        return result


BAD = [{'text': 'Single-head attention is 1.9 BLEU worse than the best setting.', 'cites': ['C1']},
       {'text': 'Multi-head attention was invented by Geoffrey Hinton in 2012.', 'cites': ['C2']}]


def show_trace(brief):
    for record in brief.trace:
        print(f'—— 第 {record["round"]} 轮验证 ——  未覆盖的子问题: {record["uncovered"] or "无"}')
        for c in record['claims']:
            print(f'  [{c["verdict"]:11}] {c["claim_id"]}: {c["text"][:95]}')
        for a in record['actions']:
            if a['action'] == 'rewrite':
                print(f'  -> 修订 {a["claim_id"]}: 改写为 {a["new_text"][:90]!r}')
            elif a['action'] == 'drop':
                print(f'  -> 修订 {a["claim_id"]}: 删除', a.get('why', ''))
            else:
                print(f'  -> 补充起草 {a["sub_question"]}: 新检索 {len(a["retrieved"])} 个 chunk，得到 {a.get("claims", 0)} 条论断')
        if record.get('dropped'):
            print('  最终丢弃的论断:', record['dropped'])
''')

code('''
question = 'Why does the Transformer use multi-head attention instead of a single attention function?'
inj = {'drafter': InjectingLLM(llms['drafter'], BAD), 'verifier': llms['verifier']}
brief = agent.run_agent(question, retriever, inj, max_rounds=2)

show_trace(brief)
''')

markdown('''
**读法**：第 0 轮是第一次验证，两条注入的错误论断应该被判为未通过（其余原本的论断通过）。
然后对每条没通过的论断做一次修订：补检索 → 改写或删除。第 1 轮是修订后的再验证。
编造的那条应该被删除；改了数字的那条如果检索到了含 0.9 BLEU 的 chunk，会被改写成正确的数字并通过。
循环在"全部通过、且每个子问题都有论断"时提前结束。
''')

code('''
print(rag.to_markdown(brief, retriever))
print()
print('answerable:', brief.answerable, '| 论断数:', len(brief.claims), '| 总 token:', brief.usage.get('total_tokens'))
print('被改写过的论断:', [(c.claim_id, '来自', c.revised_from) for c in brief.claims if c.revised_from])
''')

markdown('''
最终简报里**只有通过验证的论断**。失败的原始措辞没有进简报，但保存在 `brief.trace` 里，可以回头查每一轮发生了什么。
''')

markdown('''
## 4. 验证后直接删除 vs 完整修订

同一个问题、同样注入的两条错误论断，分别用两种设置跑：

- `max_rounds=0`：验证一次，没通过的**直接删掉**，不修订（实验里的 `verify` 组）
- `max_rounds=2`：验证后修订，再验证，最多两轮（实验里的 `full` 组）

差别应该体现在：修订能把"数字改错"那类**可救的**论断救回来，而直接删除会少一条信息。
''')

markdown('''
**注意看实际结果**：完整修订多保留了一条论断（"单头注意力比最佳设置差 0.9 BLEU"），
但这个事实**已经在第 3 条论断里写过了**。也就是说这次修订救回来的是一条**重复**的论断，并没有增加信息。
这是当前设计的一个缺口：控制器不检查改写后的论断是否和已有论断重复。
所以"完整修订比验证后删除多保留了几条论断"不能直接等同于"更完整"，评测里要看的是要点覆盖（完整性）和不支持论断率，而不是论断条数。
''')

code('''
results = {}
for rounds in (0, 2):
    inj = {'drafter': InjectingLLM(llms['drafter'], BAD), 'verifier': llms['verifier']}
    results[rounds] = agent.run_agent(question, retriever, inj, max_rounds=rounds)

for rounds, b in results.items():
    print(f'=== max_rounds={rounds}: 保留 {len(b.claims)} 条论断，验证了 {len(b.trace)} 次 ===')
    for c in b.claims:
        print('  -', c.text[:120], '(改写)' if c.revised_from else '')
    print()
''')

markdown('''
## 5. 子问题没有答案：补充起草，最后写进"未找到"

问题的第一部分资料里有答案，第二部分（Mixtral 预训练 token 数）这批论文里没有。期望：
第二个子问题没有任何论断被支持 → 控制器对它做**补充起草**（检索之前没看过的 chunk）→ 仍然没有 → 最终简报写明"未找到"，**不编造**。
''')

code('''
q3 = 'What BLEU does the base Transformer reach on English-to-German, and how many tokens was Mixtral 8x7B pretrained on?'
brief3 = agent.run_agent(q3, retriever, llms, max_rounds=2)

show_trace(brief3)
print()
print(rag.to_markdown(brief3, retriever))
''')

markdown('''
## 小结

- 流水线 = 拆子问题 → 检索 → 起草 → **验证** → 修订/补充起草 → 再验证 → 完成，由 Python 规则控制，模型只做具体的小任务
- 控制器按失败类型选动作：论断没通过 → 用论断本身补检索，改写或删除；子问题没覆盖 → 检索没看过的 chunk 重新起草
- 轮数有上限（默认 2）；最终简报只含通过验证的论断，没找到的写进"未找到"
- 每一轮都留有记录（`brief.trace`），失败的原始措辞也保存在里面，方便分析和评测
- 局限：验证看不出"漏写"；覆盖检查依赖起草时模型给论断打的子问题标签；总成本比基线高
''')

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--run' in argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
