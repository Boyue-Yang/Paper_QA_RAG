"""Generate the user-facing notebook; all chunking logic lives in chunk_papers.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown('''
# Paper QA：章节内递归切分（arXiv HTML）

输入是固定版本的 78 篇论文的 arXiv HTML 快照（`data/corpus_78`，由 `corpus_80` 去掉 Adam 和 PPO 两篇得到，
这两篇的 HTML 只是嵌入 PDF 的空壳，没有正文）；输出是本地 JSONL chunks、解析缓存和质量报告。
Notebook 调用 `scripts/chunk_papers.py`，与命令行使用同一套实现。

在 VS Code/Jupyter 中选择项目 `.venv` 的 Python 内核，然后依次运行所有单元格。
如使用新环境，先安装 `requirements-chunking.txt` 中的依赖。

流程：校验 HTML 快照哈希 → 按标签提取章节/段落 → 合并过短小节 → 小节内递归切分 → 来源与长度校验 → 保存。
默认：正文最多 512 tokens，重叠最多 64 tokens；排除参考文献和致谢，保留摘要与附录。
章节、参考文献和阅读顺序来自 HTML 标签，不再靠字体和版面猜测。
''')

code('''
from pathlib import Path
import dataclasses
import importlib.util
import json
import sys
from html import escape
from IPython.display import HTML, display

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "scripts" / "chunk_papers.py").exists()
             and (p / "data" / "corpus_78" / "manifest.json").exists()), None)
if ROOT is None:
    raise RuntimeError("请从 Paper_QA_RAG 项目根目录或 notebooks 目录运行此 notebook。")
missing = [name for name in ["bs4", "lxml", "tokenizers", "huggingface_hub"] if importlib.util.find_spec(name) is None]
if missing:
    raise ModuleNotFoundError(f"缺少依赖 {missing}。请在当前内核运行 %pip install -r \\"{ROOT / 'requirements-chunking.txt'}\\"。")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.chunk_papers import ChunkingConfig, run_pipeline, read_jsonl, get_tokenizer, token_count
print("项目目录:", ROOT)
print("当前 Python:", sys.executable)
''')

markdown('''
## 配置

`max_tokens` 是 chunk 正文的 token 上限，`overlap_tokens` 是同一小节内相邻 chunks 的最大重叠。
`min_section_tokens=100`：短于 100 tokens 的小节会并入同一一级章节下相邻的小节（先向后并，末尾的向前并），
被并入的小节标题会作为一行标题写进正文，不会丢失。这样不会出现只有一两句话的孤立 chunk。
tokenizer 使用 embedding 模型 `BAAI/bge-m3`（XLM-R SentencePiece）的 `tokenizer.json`，
因此 token 数与 embedding 实际输入一致；该模型输入上限为 8192 tokens，512 的 chunk 另加 2 个特殊 token 仍远低于上限。
首次运行需联网下载 tokenizer 文件，之后缓存在项目 `.cache/hf/`。

将 `LIMIT` 改成 `3` 可先预览三篇，设为 `None` 处理全部 78 篇。
改变配置或选择不同论文子集会生成不同的结果目录；解析缓存可以复用。
''')

code('''
CONFIG = ChunkingConfig(
    max_tokens=512,
    overlap_tokens=64,
    tokenizer_name="BAAI/bge-m3",
    include_references=False,
    include_front_matter=False,
    source_format="html",
    min_section_tokens=100,
)
MANIFEST = ROOT / "data" / "corpus_78" / "manifest.json"
OUTPUT_ROOT = ROOT / "data" / "processed" / "corpus_78"
LIMIT = None
ONLY_PAPERS = None  # 例如 ["arxiv_1706.03762"]
FORCE_REPARSE = False
print(json.dumps(dataclasses.asdict(CONFIG), indent=2, ensure_ascii=False))
''')

markdown('''
## 运行并保存

章节来自 `<section>` 嵌套，参考文献来自 `ltx_bibliography`，公式保留 LaTeX（`$...$`），
表格按行展开为 `cell | cell`，图片只保留图注。转换错误元素（`ltx_ERROR`）会被跳过并计数。
同一小节中先尝试合并段落，超长段落再按句子、空白边界和 Unicode 字符边界细分。
正文始终取自规范化 section 文本的连续范围，切分阶段不会用 LLM 改写文本。
''')

code('''
def show_progress(message):
    entry = json.loads(message)
    if entry["paper"] == 1 or entry["paper"] % 10 == 0 or entry["paper"] == entry["total"] or entry["status"] == "failed":
        print(f"{entry['paper']}/{entry['total']}: {entry['paper_id']}  {entry['status']}  chunks={entry['chunks']}")

result = run_pipeline(
    root=ROOT, manifest_path=MANIFEST, output_root=OUTPUT_ROOT,
    config=CONFIG, limit=LIMIT, paper_ids=ONLY_PAPERS,
    force_reparse=FORCE_REPARSE, progress=show_progress,
)
RUN_DIR = Path(result["output_dir"])
assert result["failed_papers"] == 0, "有论文处理失败，请查看 quality_report.json 后重跑。"
''')

code('''
metrics = {
    "处理论文数": result["completed_papers"],
    "chunks 数": result["total_chunks"],
    "待抽查论文数": result["papers_needing_review"],
    "最小 tokens": result["token_statistics"]["min"],
    "中位 tokens": result["token_statistics"]["median"],
    "最大 tokens": result["token_statistics"]["max"],
    "少于 64 tokens 的 chunks": result["token_statistics"]["chunks_under_64_tokens"],
}
rows = "".join(f"<tr><td style='padding:6px 18px'>{escape(str(k))}</td><td style='padding:6px 18px'>{escape(str(v))}</td></tr>" for k,v in metrics.items())
display(HTML("<table>" + rows + "</table>"))
print("输出目录:", RUN_DIR)
''')

markdown('''
## 检查章节识别

以下展示首篇论文解析出的章节树和各节正文长度（合并短小节之前的原始解析）。`kind=references` 的文本仍保留在解析缓存中，
只是默认不进入检索 chunks。每个 chunk 的字符偏移都以对应 section 的规范化文本为基准。
''')

code('''
chunks = read_jsonl(RUN_DIR / "chunks.jsonl")
reports = read_jsonl(RUN_DIR / "papers.jsonl")
first = reports[0]
parsed = json.loads(Path(first["parsed_json"]).read_text(encoding="utf-8"))
for section in parsed["sections"][:35]:
    print("  " * max(0, section["level"] - 1) + section["title"],
          f"[{section['kind']}; chars={len(section['text'])}]")
print("\\n本篇合并掉的短小节数:", first["merged_section_count"])
''')

markdown('''
## 查看一个 chunk 及其来源

稳定的 `paragraph_id` 和 HTML 元素 ID（`element_id`）用于定位原文；`chunk_id` 还包含切分配置和字符范围。
评测集的证据以逐字摘录为准，再对齐到 chunk；不要只绑定 chunk ID。
''')

code('''
sample = next((c for c in chunks if "Scaled Dot-Product" in c["section_title"]), chunks[0])
print(json.dumps({k: sample[k] for k in [
    "chunk_id", "paper_id", "version", "section_path", "token_count", "content_types",
    "char_start", "char_end", "source_format", "source_url"
]}, ensure_ascii=False, indent=2))
print("\\n正文:\\n" + sample["text"])
print("\\n来源段落范围:\\n" + json.dumps(sample["source_spans"], ensure_ascii=False, indent=2))
''')

markdown('''
## 输出检查与质量报告

程序已逐篇验证 token 上限、段落/章节字符范围、正文覆盖、排除规则和 chunk ID 唯一性。
这些检查不能证明 HTML 转换在语义上完全正确；`latex_conversion_errors_in_body` 表示该论文的 HTML 含有 LaTeXML 转换错误，
应抽查公式和表格密集的部分。图像未被读取，表格只保留文字。
''')

code('''
encoding = get_tokenizer(CONFIG.tokenizer_name, ROOT)
assert len(chunks) == len({c["chunk_id"] for c in chunks})
assert all(0 < token_count(encoding, c["text"]) <= CONFIG.max_tokens for c in chunks)
assert all(c["source_spans"] and all(s["element_id"] for s in c["source_spans"]) for c in chunks)
if not CONFIG.include_references:
    assert all(c["section_kind"] != "references" for c in chunks)
quality = json.loads((RUN_DIR / "quality_report.json").read_text(encoding="utf-8"))
print("chunk 完整性检查通过。")
print("需要人工抽查的论文:")
for paper in quality["needs_review"]:
    print(paper["paper_id"], paper["quality_flags"], "错误元素数:", paper["latex_error_elements"])
if not quality["needs_review"]:
    print("本轮没有触发启发式质量标记；仍建议抽查。")
''')

markdown('''
## 本地目录

- `data/corpus_78/html/`：固定版本的 arXiv HTML 快照，`manifest.json` 记录 URL 和 SHA-256。
- `data/corpus_80/`：原始 80 篇清单和 PDF，保持不变。
- `data/processed/corpus_78/parsed/<parser_version>/`：每篇的章节/段落 JSON，以及方便抽查的 TXT。
- `data/processed/corpus_78/chunks/<run_id>/chunks.jsonl`：全部论文的 chunks，每行一条 JSON。
- 同目录下 `by_paper/`：按论文保存的 chunks；`summary.json`、`config.json`、`papers.jsonl` 和 `quality_report.json` 记录配置与质量。

后续 embedding 和 retrieval 直接读取 `chunks.jsonl`（也可读 `data/processed/corpus_78/latest_run.json` 找到最新一次）。
如果给正文加上论文标题、章节路径等前缀，需要按 embedding 模型的 tokenizer 重新计算完整输入长度。
此 notebook 不调用 embedding 或生成 API。
''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    'kernelspec': {'display_name': 'Python 3 (Paper RAG .venv)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
}
nbf.validate(notebook)
target = ROOT / 'notebooks/section_recursive_chunking.ipynb'
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(target)
