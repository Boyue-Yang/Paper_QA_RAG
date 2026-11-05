"""Generate notebooks/adaptive_agent_demo.ipynb; all agent logic lives in scripts/adaptive_agent.py.

  python -m scripts.build_adaptive_notebook          # write the notebook
  python -m scripts.build_adaptive_notebook --run    # also execute it (many DeepSeek calls, several minutes)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'adaptive_agent_demo.ipynb'
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：自适应 agent（由模型决定每一步）演示

`agent.py` 的固定规则版（评测里叫 `rules`）是"我写死规则，根据观察选下一步"。
`adaptive_agent.py`（评测里叫 `agent`）是**让模型自己选下一步**：每一步它看到当前状态，从六个动作里选一个，写出理由，
代码执行后把结果还给它，再选下一步，直到它选择结束或步数用完。

## 已实现的内容

| 部分 | 做法 |
|---|---|
| 控制器 | `controller` 角色（deepseek-flash，开思考），每一步输出 JSON：`thought`（理由）、`action`、`args`、`notes`（它自己的计划笔记，下一步会还给它） |
| 动作 | `search`（自己写查询，可限定在某几篇论文里找）、`draft`（从没用过的证据写论断）、`verify`、`revise`、`drop`、`finish`（声明能否回答、缺什么） |
| 状态 | 证据池（编号 E1、E2…，标出哪些还没拿去起草）、论断列表和判决、它的笔记、最近的决定、上一步的结果、剩余步数 |
| 代码保证（模型改变不了） | ① 自动先做一轮检索（和基线一样取 8 个）；② **步数上限**（`MAX_STEPS = 12`）；③ 论断**必须先验证**，没验证的论断不进简报（模型忘了验证，代码替它验证）；④ 只有 `supported` 的论断输出；⑤ 无效动作会作为错误结果还给模型，连续 3 次无效就收尾；⑥ 控制器调用失败也会收尾，不会崩溃 |
| 记录 | 每一步的理由、动作、参数、结果都记在 `Brief.trace` 里 |

内容：
1. 控制器的提示词
2. 控制器每一步实际看到的文字
3. 一道多论文题，逐步展示每个决定
4. 一道不可回答题
5. 同一道不可回答题用固定规则版（`rules`）跑一遍作对比
6. 步数上限：把预算压到 3 步会怎样
7. 护栏：模型发无效动作、试图跳过验证时会怎样

运行前：选项目 `.venv` 的内核，`.env` 里要有 `DEEPSEEK_API_KEY`。第 3、4、5、6、7 步会真实调用 DeepSeek，整个 notebook 需要几分钟。
''')

code('''
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
while not (ROOT / 'scripts').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts import adaptive_agent as aa
from scripts import agent as rules_agent
from scripts import rag
from scripts.llm import LLM, ROLES
from scripts.retrieval import Retriever

retriever = Retriever()
llms = {name: LLM(name) for name in ('controller', 'drafter', 'verifier')}
print('控制器模型角色:', ROLES['controller'])
print('默认步数上限 MAX_STEPS =', aa.MAX_STEPS, '| 首轮检索 K_SEED =', aa.K_SEED, '| 每次 search 新增 K_SEARCH =', aa.K_SEARCH)


def short(brief):
    """One line per decision: step, action, args, and the model's reason."""
    for r in brief.trace:
        print(f"{r['step']:>2} {r['action']:7} {json.dumps(r['args'], ensure_ascii=False)[:110]}")
        if r.get('thought'):
            print('     因为:', r['thought'][:230])
''')

markdown('''
## 1. 控制器的提示词

先看模型被告知了什么。重点：动作菜单、"自己决定顺序和何时停止"、"问题的关键事实不在集合里就声明不可回答、不要用相关但不回答问题的论断凑数"。
提示词末尾还会附上 78 篇论文的"编号: 标题"清单，这样它才知道可以指定去哪篇论文里找。
''')

code('''
print(aa.CONTROLLER_SYSTEM)
print()
print('--- 附在后面的论文清单（前 4 行，共', len(retriever.papers()), '行）---')
print('\\n'.join(aa.collection_list(retriever.papers()).splitlines()[:4]))
''')

markdown('''
## 2. 控制器每一步实际看到什么

下面构造一个刚做完"自动首轮检索"的状态，打印 `state.render(...)`，这就是第 1 步发给控制器的用户消息。
证据池里每个 chunk 只有编号、论文、章节；论断列表现在是空的；上一步的结果是首轮检索返回的段落摘要。
''')

code('''
question = 'What bit widths do LLM.int8() and GPTQ each quantize to, and does each one make inference faster?'
state = aa.State(question, retriever, llms['drafter'], llms['verifier'])
state.last = state.act_search({'query': question}, aa.K_SEED)
state.log.append({'step': 0, 'action': 'search', 'args': {'query': question}, 'thought': 'initial retrieval (automatic)', 'result': state.last})
print(state.render(1, aa.MAX_STEPS))
''')

markdown('''
## 3. 一道多论文题：逐步看每个决定

这道题要从两篇不同的论文里找答案（LLM.int8() 和 GPTQ），而且问了两件事（位宽、是否加速）。
下面的运行**每一步的动作、参数、理由、结果**都显示出来；理由是模型自己写的。
''')

code('''
brief = aa.run_adaptive(question, retriever, llms)
print(aa.format_decisions(brief.trace, width=420))
''')

markdown('''
**读法**：
- `step 0` 是代码自动做的首轮检索，之后每一步都是模型的选择。
- 注意它的 `search` 查询是**它自己写的**，不一定等于原问题；`papers` 参数表示它决定只在某篇论文里找。
- `draft` 的结果里论断是"未验证"的；`verify` 才给出判决；没通过的论断由它决定 `revise` 还是 `drop`。
- 最后一步是 `finish`，它自己声明能否回答、缺什么。
''')

code('''
print('动作统计:', dict(Counter(r['action'] for r in brief.trace)))
print('一共', len(brief.trace) - 1, '个决定（不含自动首轮检索），上限', aa.MAX_STEPS)
print('用了', brief.usage.get('total_tokens'), 'token | 警告:', brief.warnings)
print()
print(rag.to_markdown(brief, retriever))
''')

markdown('''
## 4. 一道不可回答题

问题的前提在资料里不存在（Mixtral 论文没有给出预训练的 token 数）。要看的是：它搜了几次、怎么换查询、
什么时候放弃、最后怎么回答。下面只显示每一步的动作和理由，结果在 `brief.trace` 里。
''')

code('''
unanswerable = 'How many tokens was Mixtral 8x7B pretrained on?'
brief_u = aa.run_adaptive(unanswerable, retriever, llms)
short(brief_u)
print()
print('answerable =', brief_u.answerable, '| 决定数:', len(brief_u.trace) - 1, '| token:', brief_u.usage.get('total_tokens'))
print('not_found:', brief_u.not_found[:400])
''')

markdown('''
## 5. 对比：同一道题用固定规则版

固定规则版（`rules`）的结束条件是"全部论断通过、子问题都有论断"。
它没有"这些论断是真的但没回答问题"这个判断，只看有没有被支持的论断。下面看它在同一道题上的结果。
''')

code('''
brief_r = rules_agent.run_agent(unanswerable, retriever, llms, max_rounds=2)
print('rules: answerable =', brief_r.answerable, '| 论断数:', len(brief_r.claims))
for c in brief_r.claims:
    print('  -', c.text)
print('not_found:', brief_r.not_found[:300])
print()
print('agent: answerable =', brief_u.answerable, '| 论断数:', len(brief_u.claims))
''')

markdown('''
**注意看实际结果**：上次运行时，固定规则版在这道题上**也拒答了**，因为起草模型自己就没有写出论断（拒答的第一道关起了作用）。
所以在这道题上，agent 并没有显示出比 `rules` 更好的结果，两者答案相同，agent 只是多走了几步搜索。
两者的差别只会出现在另一种情形：起草模型写出了"真实、能通过验证、但没有回答问题"的论断。固定规则版会因为"有通过验证的论断"而当作可回答，
agent 则可以自己判断"这些论断没回答问题"而拒答。这次没有触发这种情形，所以这里只能说明设计上的区别，不能当作 agent 更好的证据。
评测里的不可回答题（每个 split 6 道）才是检验这一点的地方。
''')

markdown('''
## 6. 步数上限

把预算压到 3 步。到最后一步时，提示里会告诉模型"这是最后一步，必须结束"，模型一般会听从并给出收尾（这一步它也可以顺便说明哪些没来得及找）。
如果它仍然不结束，代码才会强制收尾：先验证还没验证的论断，再只输出通过的论断，并在 `warnings` 里写明"是被步数预算收尾的"。
下面的运行里模型自己在最后一步选择了 `finish`，所以 `warnings` 是空的；代码强制收尾的路径由单元测试覆盖。
注意预算少意味着答案可能不完整：看它在 `not_found` 里说了什么没找到。
''')

code('''
tight = aa.run_adaptive(question, retriever, llms, max_steps=3)
short(tight)
print()
print('警告:', tight.warnings)
print('answerable =', tight.answerable, '| 输出的论断数:', len(tight.claims), '| 全部 supported:',
      all(c.verdict['verdict'] == 'supported' for c in tight.claims))
''')

markdown('''
## 7. 护栏：模型发无效动作、试图跳过验证

用一个**脚本化的控制器**（不是真模型）模拟两种出错：先发一个不存在的动作 `fly`，再在没验证的情况下直接 `finish`。
起草和验证仍然用真实模型。期望：无效动作变成一条 ERROR 还给模型；没验证的论断在输出前被代码自动验证。
''')

code('''
class ScriptedController:
    def __init__(self, *steps):
        self.steps = iter(steps)

    def chat(self, system, user, json_mode=False):
        return SimpleNamespace(parsed=next(self.steps), usage={})


script = ScriptedController(
    {'thought': 'try something invalid', 'action': 'fly', 'args': {}},
    {'thought': 'draft from the first retrieval', 'action': 'draft', 'args': {}},
    {'thought': 'finish without verifying', 'action': 'finish', 'args': {'answerable': True, 'not_found': ''}},
)
guarded = aa.run_adaptive('How does layer normalization differ from batch normalization?', retriever,
                          {**llms, 'controller': script})
for r in guarded.trace:
    print(f"{r['step']:>2} {r['action']:7} 因为: {r['thought'][:70]:70} -> {r['result'][:90].splitlines()[0]}")
print()
print('输出的论断:', len(guarded.claims), '条，全部 supported:',
      all(c.verdict['verdict'] == 'supported' for c in guarded.claims))
''')

markdown('''
最后一行 `verify` 的理由是 `automatic: claims must be verified before the brief is written`：模型想跳过验证，代码没让它跳过。

## 小结

- agent = 模型每步自己选动作（搜什么、在哪篇论文里搜、起草、验证、修订、删除、结束），并写下理由；所有决定都记录在 `Brief.trace`
- 不由模型决定的只有：自动首轮检索、步数上限、验证不可跳过、只有 supported 的论断输出
- 和固定规则版的差别：查询由模型写、可指定论文、由模型判断何时放弃；设计上还能判断"论断是真的但没回答问题"（本次演示的对比里没有触发，两者都拒答了）
- 代价：路径不固定、同一个问题两次可能走法不同，调用次数和 token 都多；评测时靠 `agent` 对 `rules` 的差来衡量这份自主性是否有价值
''')

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--run' in argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=2400, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
