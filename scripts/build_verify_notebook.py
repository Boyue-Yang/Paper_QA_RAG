"""Generate notebooks/verification_demo.ipynb; all verification logic lives in scripts/verify.py.

  python -m scripts.build_verify_notebook          # write the notebook
  python -m scripts.build_verify_notebook --run    # also execute it (calls the DeepSeek API a few times)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks' / 'verification_demo.ipynb'
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：引用验证（C. citation verification）演示

起草（B）产出的论断只是草稿，没有任何检查。验证回答的问题是：**每条论断是否真的被它所引用的 chunk 支持？**
逻辑在 `scripts/verify.py`，这里把三层检查一层一层摆出来：

| 层 | 谁做 | 查什么 |
|---|---|---|
| 1. 结构 | 程序，不调用模型 | 论断有没有引用？引用的 chunk 存在吗？有没有引用从未给过模型的编号？ |
| 2. 数字 | 程序，不调用模型 | 论断里的数字有没有出现在所引用的原文里？有缺的就提醒判官 |
| 3. 判官 | 大模型 | 只看"论断 + 它引用的 chunk"，给出 supported / partial / unsupported / contradicted，并**逐字抄一段原文**作为依据 |

判官抄的原文还会被程序检查一遍：抄的那段话必须真的在所引用的 chunk 里，否则"supported"会被降级。
**只有 `supported` 算通过。**

本 notebook 用几条**故意构造**的论断来测试，看验证能不能抓出来。
运行前：选项目 `.venv` 的内核，根目录 `.env` 里要有 `DEEPSEEK_API_KEY`。第 4、6、7 步会真实调用 DeepSeek（验证器角色）。
''')

code('''
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
while not (ROOT / 'scripts').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts import rag, verify
from scripts.llm import LLM, LLMError, ROLES
from scripts.retrieval import Retriever

retriever = Retriever()
drafter = LLM('drafter')
verifier = LLM('verifier')
print('验证器用的模型角色:', ROLES['verifier'])
''')

markdown('''
## 1. 准备两个已知内容的 chunk

为了知道"正确答案是什么"，先从 Attention Is All You Need 里找出两个 chunk：
一个讲单头注意力比最佳设置差多少 BLEU（6.2 节），另一个讲"有效分辨率降低"（2 节）。
''')

code('''
paper = 'arxiv_1706.03762'
variants = next(c for c in retriever.chunks.values() if c['paper_id'] == paper and '0.9 BLEU worse' in c['text'])
background = next(c for c in retriever.chunks.values() if c['paper_id'] == paper and 'reduced effective resolution' in c['text'])

for name, c in (('变体实验', variants), ('背景', background)):
    print(name, '|', c['chunk_id'], '|', c['section_title'])
i = variants['text'].index('0.9 BLEU worse')
print()
print('变体实验 chunk 里的相关句子:')
print('  ...' + variants['text'][max(0, i - 160): i + 120].replace(chr(10), ' ') + '...')
''')

markdown('''
## 2. 第 1 层：结构检查（不调用模型）

不合格的引用直接判不支持，连判官都不用叫：没有引用、引用的 chunk 不存在、引用了从未给过模型的编号（`invalid_cites`）。
''')

code('''
Claim = rag.Claim
cases = {
    '没有引用': Claim('x1', 'Single-head attention is worse than multi-head attention.', []),
    '引用的 chunk 不存在': Claim('x2', 'Single-head attention is worse than multi-head attention.', ['chunk_does_not_exist']),
    '引用了从未给过的编号': Claim('x3', 'Single-head attention is worse than multi-head attention.', [variants['chunk_id']], ['C9']),
}
for name, claim in cases.items():
    print(f'{name:14} ->', verify.structure_problem(claim, retriever))
''')

markdown('''
## 3. 第 2 层：数字检查（不调用模型）

`numbers_in` 把论断里的数字都找出来。如果论断里有数字**不在所引用的原文里**，就在发给判官的消息里加一条提醒。
这是个确定性的检查，专门抓"改了数字"这类模型最容易犯、读起来又最不显眼的错。
''')

code('''
print(verify.numbers_in('Single-head attention is 0.9 BLEU worse; the base model reaches 28.4 BLEU, 1,000 steps, GPT-3'))
print()
cited_text = variants['text']
for claim_text in ['Single-head attention is 0.9 BLEU worse than the best setting.',
                   'Single-head attention is 1.9 BLEU worse than the best setting.']:
    missing = sorted(verify.numbers_in(claim_text) - verify.numbers_in(cited_text))
    print(f'{claim_text}\\n   原文里找不到的数字: {missing}')
''')

markdown('''
## 4. 第 3 层：大模型判官

判官的提示词：只看论断和引用的摘录，不用外部知识；按四档评判；**必须逐字抄一段最相关的原文放进 `quote`**。
''')

code('''
print(verify.JUDGE_SYSTEM)
''')

markdown('''
现在用 6 条论断测试。其中只有第一条是真的；其余分别是：改了数字、对的话引错了 chunk、说反了、说得比原文强、没有引用。
每条都经过完整的 `verify_claim`（结构检查 → 数字检查 → 判官）。
''')

code('''
vid, bid = variants['chunk_id'], background['chunk_id']
claims = [
    ('A 正确',        Claim('a', 'Single-head attention is 0.9 BLEU worse than the best setting.', [vid])),
    ('B 改了数字',    Claim('b', 'Single-head attention is 1.9 BLEU worse than the best setting.', [vid])),
    ('C 引错 chunk',  Claim('c', 'Single-head attention is 0.9 BLEU worse than the best setting.', [bid])),
    ('D 说反了',      Claim('d', 'Single-head attention is 0.9 BLEU better than the best setting.', [vid])),
    ('E 说得太强',    Claim('e', 'Multi-head attention outperforms single-head attention on every task and dataset.', [vid])),
    ('F 没有引用',    Claim('f', 'Single-head attention is 0.9 BLEU worse than the best setting.', [])),
]
verdicts = verify.verify_claims([c for _, c in claims], retriever, verifier)

for name, claim in claims:
    v = verdicts[claim.claim_id]
    print(f'{name:12} 判定: {v["verdict"]:12} (判官原判: {v["raw_verdict"]}, 由 {v["checked_by"]} 判)')
    print(f'             缺失数字: {v["numbers_missing"]} | 引文在原文里: {v["quote_found"]}')
    print(f'             理由: {v["reason"][:150]}')
''')

markdown('''
**读法**：只有 A 应该是 `supported`。B 的数字 1.9 在原文里不存在，C 的论断是对的但引的 chunk 没这句话，
D 与原文相反，E 超出原文，F 没有引用 —— 都不应该通过。判官是大模型，偶尔会和预期不同，上面的输出是这一次的真实结果。
''')

markdown('''
## 5. 为什么要检查判官的引文（不调用真实模型）

判官自己也可能出错，比如凭印象说"supported"，但抄的那段话其实原文里没有。程序把引文拿到所引用的 chunk 里逐字核对，
找不到就把 `supported` / `partial` 降级为 `unsupported`。下面用一个**假判官**演示：它永远回答 supported，但引文是编的。
''')

code('''
class FakeJudge:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def chat(self, system, user, json_mode=False):
        if self.error:
            raise self.error
        return SimpleNamespace(parsed=self.reply, usage={})


claim = Claim('z', 'Single-head attention is 0.9 BLEU worse than the best setting.', [vid])
cited = [retriever.lookup_citation(vid)]

lying = FakeJudge({'verdict': 'supported', 'quote': 'a sentence the paper never wrote', 'reason': 'trust me'})
r = verify.judge_claim(claim, cited, lying)
print('编造引文  ->', r['verdict'], '| 原判:', r['raw_verdict'], '| 引文在原文里:', r['quote_found'])

honest = FakeJudge({'verdict': 'supported', 'quote': 'single-head attention is 0.9 BLEU worse than the best setting', 'reason': 'stated'})
r = verify.judge_claim(claim, cited, honest)
print('真实引文  ->', r['verdict'], '| 引文在原文里:', r['quote_found'])

r = verify.judge_claim(claim, cited, FakeJudge(error=LLMError('network down')))
print('判官失败  ->', r['verdict'], '|', r['reason'])
''')

markdown('''
最后一行：判官调用失败时结果是 `unverified`，**不是通过**。宁可多修订一次，也不放行没验证过的论断。
''')

markdown('''
## 6. 对真实起草的结果做验证

把上一个 notebook 里的流程接起来：检索 → 起草 → 验证每条论断。
''')

code('''
question = 'Why does the Transformer use multi-head attention instead of a single attention function?'
brief = rag.answer(question, retriever, drafter, k=8)
verdicts = verify.verify_claims(brief.claims, retriever, verifier)

for claim in brief.claims:
    v = verdicts[claim.claim_id]
    print(f'[{v["verdict"]:11}] {claim.text[:110]}')
    print(f'              引文: {v["quote"][:110]!r}')
print()
print('通过:', sum(v['verdict'] == 'supported' for v in verdicts.values()), '/', len(brief.claims))
''')

markdown('''
## 7. 没通过的论断接下来怎么办：修订

验证只给出判决，不改论断。没通过的论断交给 `agent.py` 里的修订：用**论断本身当查询**再检索一次，
让模型把论断改写成"摘录真正说的范围"，或者删掉。下面拿第 4 步里改了数字的论断 B 试一次。
''')

code('''
from scripts.agent import revise_claim

claim_b = claims[1][1]
claim_b.verdict = verdicts_b = verify.verify_claim(claim_b, retriever, verifier)
print('原论断 :', claim_b.text)
print('判决   :', verdicts_b['verdict'], '|', verdicts_b['reason'][:120])

new_claim, action, usage = revise_claim(claim_b, verdicts_b, retriever, drafter)
print()
print('动作   :', action['action'], '| 补检索到的 chunk 数:', len(action['retrieved']))
print('改写后 :', new_claim.text if new_claim else '(删除)')
if new_claim:
    again = verify.verify_claim(new_claim, retriever, verifier)
    print('再验证 :', again['verdict'])
''')

markdown('''
## 小结

- 验证 = 结构检查 + 数字检查 + 判官（附逐字引文，且引文会被程序核对）；只有 `supported` 算通过
- 判官失败、引文对不上，都按"未通过"处理，宁严勿松
- 验证**只判断论断对不对，不判断答案全不全**：漏掉的内容验证看不出来
- 判官是大模型，也会错；评测时需要用一小批人工标注来校准
- 没通过的论断由修订处理（补检索 + 改写或删除），再回到验证，直到全部通过或用完轮数 —— 这是 `agent.py` 的控制循环
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
