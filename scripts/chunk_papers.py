"""Local, section-aware recursive chunking for the versioned arXiv paper collection.

Default source is the pinned arXiv HTML of data/corpus_78 (see scripts/prepare_html_corpus.py);
--source-format pdf runs the older PDF layout parser on data/corpus_80.

Run from the project root:
    python scripts/chunk_papers.py
    python scripts/chunk_papers.py --max-tokens 512 --overlap-tokens 64 --limit 3

The notebook imports this module, so CLI and notebook use exactly the same code.
PDF text extraction and heading recognition are heuristic, not table/figure OCR.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pymupdf
import tiktoken

PARSER_VERSION = 'section-layout-v5'
HTML_PARSER_VERSION = 'arxiv-html-v1'
CHUNKER_VERSION = 'recursive-spans-v2'
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64
    tokenizer_name: str = 'BAAI/bge-m3'
    include_references: bool = False
    include_front_matter: bool = False
    source_format: str = 'html'
    min_section_tokens: int = 100

    def __post_init__(self):
        if self.source_format not in ('html', 'pdf'):
            raise ValueError("source_format must be 'html' or 'pdf'.")
        if self.min_section_tokens < 0:
            raise ValueError('min_section_tokens must not be negative.')
        if self.max_tokens < 16:
            raise ValueError('max_tokens must be at least 16.')
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError('Require 0 <= overlap_tokens < max_tokens.')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


class HFTokenizer:
    """Adapter giving a Hugging Face `tokenizers` object the tiktoken-style encode() used here.

    Special tokens (<s>, </s>) are not added: max_tokens counts chunk text only, and the
    embedding model adds its own two special tokens on top.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str, disallowed_special=()) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids


def _hf_tokenizer_file(name: str, root: Path) -> Path:
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    from huggingface_hub import hf_hub_download
    cache_dir = str(root / '.cache' / 'hf')
    try:  # Prefer the local copy so reruns work offline.
        return Path(hf_hub_download(name, 'tokenizer.json', cache_dir=cache_dir, local_files_only=True))
    except Exception:
        return Path(hf_hub_download(name, 'tokenizer.json', cache_dir=cache_dir))


def is_tiktoken_name(name: str) -> bool:
    return name in tiktoken.list_encoding_names()


def get_tokenizer(name: str, root: Path = DEFAULT_ROOT):
    """tiktoken encodings by name (e.g. cl100k_base), otherwise a Hugging Face repo id (e.g. BAAI/bge-m3)."""
    if is_tiktoken_name(name):
        # Keep downloaded encoding assets in this project rather than a global cache.
        os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(root / '.cache' / 'tiktoken'))
        return tiktoken.get_encoding(name)
    from tokenizers import Tokenizer
    return HFTokenizer(Tokenizer.from_file(str(_hf_tokenizer_file(name, root))))


def tokenizer_packages(name: str) -> tuple[str, ...]:
    return ('tiktoken',) if is_tiktoken_name(name) else ('tokenizers', 'huggingface_hub')


@functools.lru_cache(maxsize=None)
def tokenizer_backend(name: str, root: Path = DEFAULT_ROOT) -> str:
    """Identity of the tokenizer implementation, folded into chunk IDs and the run ID."""
    versions = ','.join(f'{pkg}=={importlib.metadata.version(pkg)}' for pkg in tokenizer_packages(name))
    if is_tiktoken_name(name):
        return versions
    return f'{versions};tokenizer.json:{file_sha256(_hf_tokenizer_file(name, root))[:12]}'


def token_count(encoding, text: str) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _safe_prefix_end(text: str, start: int, end: int, budget: int, encoding) -> int:
    """Character boundaries prevent splitting UTF-8 sequences or losing text."""
    low, high = start + 1, end
    best = start
    while low <= high:
        middle = (low + high) // 2
        if token_count(encoding, text[start:middle]) <= budget:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError('Token budget cannot hold a single Unicode character.')
    while best > start and token_count(encoding, text[start:best]) > budget:
        best -= 1
    return best


def _recursive_atoms(text: str, start: int, end: int, budget: int, encoding,
                     level: int = 0) -> list[tuple[int, int]]:
    if start >= end:
        return []
    if token_count(encoding, text[start:end]) <= budget:
        return [(start, end)]
    separators = [r'\n\s*\n', r'\n', r'(?<=[.!?。！？])\s+', r'\s+']
    if level < len(separators):
        ends = [start + match.end() for match in re.finditer(separators[level], text[start:end])]
        boundaries = [start] + [point for point in ends if start < point < end] + [end]
        if len(boundaries) > 2:
            result = []
            for left, right in zip(boundaries, boundaries[1:]):
                result.extend(_recursive_atoms(text, left, right, budget, encoding, level + 1))
            return result
        return _recursive_atoms(text, start, end, budget, encoding, level + 1)
    result = []
    while start < end:
        point = _safe_prefix_end(text, start, end, budget, encoding)
        result.append((start, point))
        start = point
    return result


def _overlap_start(text: str, start: int, end: int, budget: int, encoding) -> int:
    if not budget:
        return end
    low, high = start, end
    best = end
    while low <= high:
        middle = (low + high) // 2
        if token_count(encoding, text[middle:end]) <= budget:
            best, high = middle, middle - 1
        else:
            low = middle + 1
    # Prefer a word boundary rather than beginning overlap in the middle of a word.
    boundary = re.search(r'\s+', text[best:min(end, best + 32)])
    if boundary:
        best += boundary.end()
    while best < end and token_count(encoding, text[best:end]) > budget:
        best += 1
    return best


def recursive_chunk_spans(text: str, encoding, max_tokens: int = 512,
                          overlap_tokens: int = 64) -> list[tuple[int, int]]:
    """Return offsets into the unchanged section text, never decoded token slices.

    Paragraphs, lines, sentences, and words are split recursively only as needed.
    Overlap is an upper bound; it shrinks when necessary to fit the next atom.
    """
    if max_tokens < 16 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError('Invalid token/overlap settings.')
    atoms = _recursive_atoms(text, 0, len(text), max_tokens, encoding)
    result = []
    index, carry_start = 0, None
    while index < len(atoms):
        start, end = atoms[index]
        if carry_start is not None and token_count(encoding, text[carry_start:end]) <= max_tokens:
            start = carry_start
        while index + 1 < len(atoms):
            candidate_end = atoms[index + 1][1]
            if token_count(encoding, text[start:candidate_end]) > max_tokens:
                break
            index += 1
            end = candidate_end
        left, right = _trim_span(text, start, end)
        if left < right:
            result.append((left, right))
            carry_start = _overlap_start(text, left, right, overlap_tokens, encoding)
        else:
            carry_start = None
        index += 1
    return result


def _clean_line(text: str) -> str:
    # Preserve mathematical Unicode. Only fix common PDF ligatures and soft hyphens.
    for before, after in {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', '\u00ad': ''}.items():
        text = text.replace(before, after)
    return re.sub(r'[ \t]+', ' ', unicodedata.normalize('NFC', text)).strip()


def _heading_key(text: str) -> str:
    text = re.sub(r'^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)[.)]?\s+', '', text.strip())
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def _bbox_union(lines: list[dict]) -> list[float]:
    return [round(min(l['bbox'][0] for l in lines), 2), round(min(l['bbox'][1] for l in lines), 2),
            round(max(l['bbox'][2] for l in lines), 2), round(max(l['bbox'][3] for l in lines), 2)]


def _extract_page_units(page, page_number: int) -> list[dict]:
    content = page.get_text('dict', flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    links = [link for link in page.get_links() if link.get('kind') == pymupdf.LINK_GOTO
             and link.get('page', page_number - 1) != page_number - 1]
    units = []
    for block_index, block in enumerate(content['blocks']):
        if block.get('type') != 0:
            continue
        lines = []
        for line in block.get('lines', []):
            spans = line.get('spans', [])
            text = _clean_line(''.join(span.get('text', '') for span in spans))
            if not text:
                continue
            weighted = [(float(span['size']), max(1, len(span.get('text', '')))) for span in spans]
            size = sum(size * weight for size, weight in weighted) / max(1, sum(w for _, w in weighted))
            bold_chars = sum(max(1, len(span.get('text', ''))) for span in spans
                             if (span.get('flags', 0) & 16) or 'bold' in span.get('font', '').lower())
            line_rect = pymupdf.Rect(line['bbox'])
            linked_elsewhere = any((line_rect & link['from']).get_area() / max(1, line_rect.get_area()) >= .75 for link in links)
            lines.append({'text': text, 'bbox': [round(float(x), 2) for x in line['bbox']],
                'font_size': round(size, 2),
                'bold': bold_chars / max(1, sum(w for _, w in weighted)) >= .7,
                'linked_elsewhere': linked_elsewhere})
        # A section number and its title may be separate PDF lines on the same baseline.
        merged_lines = []
        for line in lines:
            previous = merged_lines[-1] if merged_lines else None
            if (previous and abs(previous['bbox'][1] - line['bbox'][1]) < 2
                    and previous['bbox'][2] <= line['bbox'][0] + 1):
                count_a, count_b = len(previous['text']), len(line['text'])
                previous.update(text=previous['text'] + ' ' + line['text'], bbox=_bbox_union([previous, line]),
                    font_size=round((previous['font_size'] * count_a + line['font_size'] * count_b) / (count_a + count_b), 2),
                    bold=(previous['bold'] * count_a + line['bold'] * count_b) / (count_a + count_b) >= .7,
                    linked_elsewhere=previous['linked_elsewhere'] and line['linked_elsewhere'])
            else:
                merged_lines.append(line)
        lines = merged_lines
        if lines:
            units.append({'page': page_number, 'pdf_block': block_index, 'lines': lines,
                          'bbox': _bbox_union(lines), 'page_width': page.rect.width, 'page_height': page.rect.height})
    return units


def _body_font(pages: list[list[dict]]) -> float:
    sizes = collections.Counter()
    for units in pages:
        for unit in units:
            for line in unit['lines']:
                if len(line['text']) >= 30:
                    sizes[round(line['font_size'] * 2) / 2] += len(line['text'])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _remove_margins(pages: list[list[dict]]) -> tuple[list[list[dict]], int]:
    occurrences = collections.defaultdict(set)
    for page in pages:
        for unit in page:
            if unit['bbox'][1] < .08 * unit['page_height'] or unit['bbox'][3] > .92 * unit['page_height']:
                key = ' '.join(line['text'] for line in unit['lines']).lower()
                if len(key) <= 160:
                    occurrences[key].add(unit['page'])
    repeated = {key for key, found in occurrences.items() if len(found) >= max(3, math.ceil(len(pages) * .3))}
    output, removed = [], 0
    for page in pages:
        kept = []
        for unit in page:
            text = ' '.join(line['text'] for line in unit['lines'])
            edge = unit['bbox'][1] < .08 * unit['page_height'] or unit['bbox'][3] > .92 * unit['page_height']
            if (edge and (text.lower() in repeated or re.fullmatch(r'\d{1,4}', text))) or re.match(r'^arXiv:\s*\d', text):
                removed += 1
            else:
                kept.append(unit)
        output.append(kept)
    return output, removed


def order_page_units(units: list[dict], body_font: float) -> tuple[list[dict], str]:
    """Read left column then right, within bands separated by full-width blocks."""
    if not units:
        return [], 'empty'
    width = units[0]['page_width']
    middle = width / 2
    body = [line for unit in units for line in unit['lines']
            if len(line['text']) >= 25 and .85 * body_font <= line['font_size'] <= 1.15 * body_font]
    left = sum(line['bbox'][2] < middle + 8 for line in body)
    right = sum(line['bbox'][0] > middle - 8 for line in body)
    crossing = sum(line['bbox'][0] < middle - 8 and line['bbox'][2] > middle + 8 for line in body)
    two_columns = left >= 6 and right >= 6 and crossing / max(1, len(body)) < .25
    if not two_columns:
        return sorted(units, key=lambda u: (u['bbox'][1], u['bbox'][0])), 'single_column'
    split_units = []
    for unit in units:
        group, category = [], None
        for line in unit['lines']:
            bbox = line['bbox']
            current = 'wide' if bbox[0] < middle - 8 and bbox[2] > middle + 8 else 'left' if (bbox[0] + bbox[2]) / 2 < middle else 'right'
            if group and category != current:
                split_units.append({**unit, 'lines': group, 'bbox': _bbox_union(group), 'column': category})
                group = []
            group.append(line)
            category = current
        if group:
            split_units.append({**unit, 'lines': group, 'bbox': _bbox_union(group), 'column': category})
    wide = sorted((u for u in split_units if u['column'] == 'wide'), key=lambda u: u['bbox'][1])
    remaining = [u for u in split_units if u['column'] != 'wide']
    ordered = []
    for separator in wide:
        above = [u for u in remaining if (u['bbox'][1] + u['bbox'][3]) / 2 < (separator['bbox'][1] + separator['bbox'][3]) / 2]
        remaining = [u for u in remaining if u not in above]
        ordered.extend(sorted(above, key=lambda u: (u['column'] != 'left', u['bbox'][1], u['bbox'][0])))
        ordered.append(separator)
    ordered.extend(sorted(remaining, key=lambda u: (u['column'] != 'left', u['bbox'][1], u['bbox'][0])))
    return ordered, 'two_columns_heuristic'


NUMBERED_HEADING = re.compile(r'^(?P<number>\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)[.)]?\s+(?P<title>[A-Z][^\n]{1,135})$')
KNOWN_HEADINGS = {
    'abstract', 'introduction', 'background', 'relatedwork', 'method', 'methods', 'methodology',
    'experiments', 'experimentalsetup', 'results', 'discussion', 'conclusion', 'conclusions',
    'limitations', 'references', 'bibliography', 'acknowledgments', 'acknowledgements',
    'appendix', 'appendices', 'supplementarymaterial', 'broaderimpact', 'ethicsstatement',
}


def detect_heading(line: dict, body_font: float, outline: dict, page_number: int) -> dict | None:
    text = line['text'].strip()
    if len(text) > 150 or len(text.split()) > 20:
        return None
    if text.endswith(':') or re.match(r'^(?:\d+[.)]?\s+)?(?:return|for|if|else|end|while|repeat|until|break|continue|assert|input|output)\b', text, re.I):
        return None
    key = _heading_key(text)
    emphasized = line['bold'] or line['font_size'] >= body_font * 1.06
    toc = outline.get((page_number, key))
    if toc and (emphasized or NUMBERED_HEADING.fullmatch(text) or text.isupper()):
        return {'title': text, 'level': toc['level'], 'method': 'pdf_outline_match'}
    numbered = NUMBERED_HEADING.fullmatch(text)
    if numbered and emphasized and not re.search(r'[.;,]$', numbered['title']):
        number = numbered['number']
        if number.split('.')[0].isdigit() and int(number.split('.')[0]) > 40:
            return None
        return {'title': text, 'level': number.count('.') + 1, 'method': 'number_and_font'}
    if (numbered and re.fullmatch(r'[A-Z](?:\.\d+)*', numbered['number'])
            and line['font_size'] >= body_font * .9 and len(text) <= 110
            and not re.search(r'[,;=]|\b(?:19|20)\d{2}\b|[.]$', numbered['title'])):
        return {'title': text, 'level': numbered['number'].count('.') + 1, 'method': 'appendix_heading_heuristic'}
    if key in KNOWN_HEADINGS and (emphasized or key in ('abstract', 'references', 'bibliography')):
        return {'title': text, 'level': 1, 'method': 'named_heading'}
    if (page_number > 1 and line['font_size'] >= body_font * 1.12 and (line['bold'] or text.isupper())
            and 2 <= len(text.split()) <= 12 and not re.search(r'[.;,]$', text)
            and not all(len(word) <= 1 for word in text.split())
            and re.fullmatch(r'[A-Z][A-Za-z0-9 /&()\x27’–-]+', text)):
        return {'title': text, 'level': 1, 'method': 'large_font_heading'}
    return None


def _section_kind(title: str) -> str:
    key = _heading_key(title)
    if key in ('references', 'bibliography'):
        return 'references'
    if key in ('acknowledgments', 'acknowledgements'):
        return 'acknowledgments'
    return 'body'


def parse_pdf_sections(pdf_path: Path, paper: dict, pdf_hash: str) -> dict:
    """Create normalized sections plus stable paragraph/page/bbox locators."""
    with pymupdf.open(pdf_path) as document:
        outline_rows = document.get_toc()
        outline = {(int(page), _heading_key(title)): {'level': int(level), 'title': title}
                   for level, title, page in outline_rows if page > 0}
        raw_pages = [_extract_page_units(page, i + 1) for i, page in enumerate(document)]
        font = _body_font(raw_pages)
        pages, removed = _remove_margins(raw_pages)
        page_stats, ordered_pages = [], []
        for page_number, units in enumerate(pages, 1):
            ordered, layout = order_page_units(units, font)
            raw_lines = [line for unit in ordered for line in unit['lines']]
            toc_regions = []
            for unit in ordered:
                for title_line in unit['lines']:
                    if re.fullmatch(r'(?:table\s+of\s+)?contents(?:\s+of\s+.*)?|appendix\s+contents', title_line['text'], re.I):
                        middle = unit['page_width'] / 2
                        if layout == 'two_columns_heuristic' and unit['bbox'][2] < middle + 8:
                            x0, x1 = 0, middle
                        elif layout == 'two_columns_heuristic' and unit['bbox'][0] > middle - 8:
                            x0, x1 = middle, unit['page_width']
                        else:
                            x0, x1 = 0, unit['page_width']
                        y0 = title_line['bbox'][1]
                        following = [line['bbox'][1] for line in raw_lines
                            if line['bbox'][1] > title_line['bbox'][3] and x0 <= line['bbox'][0] < x1
                            and line['font_size'] >= max(font * 1.1, title_line['font_size'] * 1.08)
                            and (NUMBERED_HEADING.fullmatch(line['text']) or _heading_key(line['text']) in KNOWN_HEADINGS)]
                        toc_regions.append([x0, y0, x1, min(following) if following else unit['page_height']])
            likely_toc = bool(toc_regions) or sum(line.get('linked_elsewhere', False) for line in raw_lines) >= 8
            filtered = []
            for unit in ordered:
                kept = [line for line in unit['lines'] if not (likely_toc and line.get('linked_elsewhere')) and not any(
                    x0 <= (line['bbox'][0] + line['bbox'][2]) / 2 < x1 and y0 <= line['bbox'][1] < y1
                    for x0, y0, x1, y1 in toc_regions)]
                if kept:
                    filtered.append({**unit, 'lines': kept, 'bbox': _bbox_union(kept)})
            lines = [line['text'] for unit in filtered for line in unit['lines']]
            page_stats.append({'page': page_number, 'layout': layout,
                'extracted_characters': sum(len(text) for text in lines), 'excluded_toc_regions': toc_regions})
            ordered_pages.append(filtered)

    sections, stack = [], []

    def new_section(title: str, level: int, method: str, page: int, kind: str | None = None):
        while stack and stack[-1]['level'] >= level:
            stack.pop()
        section = {'section_id': f"{paper['paper_id']}_{paper['selected_version']}_s{len(sections):04d}",
            'title': title, 'level': level, 'heading_method': method, 'heading_page': page,
            'path': [s['title'] for s in stack] + [title],
            'kind': kind or ('references' if stack and stack[-1]['kind'] == 'references' else _section_kind(title)),
            'paragraphs': []}
        sections.append(section)
        if level:
            stack.append(section)
        return section

    current = new_section('Front matter', 0, 'synthetic_front_matter', 1, 'front_matter')
    for page_number, units in enumerate(ordered_pages, 1):
        for unit in units:
            fragment = []

            def flush():
                if not fragment:
                    return
                text = ' '.join(line['text'] for line in fragment)
                bbox = _bbox_union(fragment)
                block_id = 'p_' + digest_json([pdf_hash, page_number, unit['pdf_block'], bbox, text])[:20]
                current['paragraphs'].append({'paragraph_id': block_id, 'page': page_number,
                    'bbox': bbox, 'pdf_block': unit['pdf_block'], 'text': text,
                    'raw_lines': [line['text'] for line in fragment],
                    'kind': 'caption' if re.match(r'^(?:Table|Figure|Fig\.|Algorithm)\s+\d+', text) else 'text'})
                fragment.clear()

            index = 0
            while index < len(unit['lines']):
                line = unit['lines'][index]
                heading = detect_heading(line, font, outline, page_number)
                consumed = 1
                # Join a wrapped heading only when the complete title matches a PDF bookmark.
                if not heading or heading['method'] != 'pdf_outline_match':
                    for count in (2, 3):
                        sample = unit['lines'][index:index + count]
                        if len(sample) != count:
                            continue
                        combined = {**line, 'text': ' '.join(l['text'] for l in sample)}
                        if (page_number, _heading_key(combined['text'])) in outline:
                            heading = detect_heading(combined, font, outline, page_number)
                            if heading:
                                heading['title'] = outline[(page_number, _heading_key(combined['text']))]['title']
                            consumed = count
                            break
                if heading:
                    numbered = NUMBERED_HEADING.fullmatch(heading['title'])
                    # A first-page title starting with "A ..." is not an appendix heading.
                    title_like = (page_number == 1 and numbered and numbered['number'] == 'A'
                                  and line['bbox'][1] < .35 * unit['page_height']
                                  and heading['method'] != 'pdf_outline_match')
                    theorem_like = numbered and re.match(r'^(?:Theorem|Lemma|Proposition|Definition|Assumption|Corollary|Proof|Equation)\b', numbered['title'])
                    if title_like or theorem_like:
                        heading = None
                if heading:
                    flush()
                    if numbered:
                        heading['level'] = numbered['number'].count('.') + 1
                    current = new_section(heading['title'], heading['level'], heading['method'], page_number)
                else:
                    fragment.extend(unit['lines'][index:index + consumed])
                index += consumed
            flush()

    real_headings = [s for s in sections if s['heading_method'] != 'synthetic_front_matter']
    flags = []
    # Some reports have a prose abstract without an "Abstract" heading. Preserve
    # substantial pre-heading prose rather than dropping it with author metadata.
    if real_headings and not any(_heading_key(s['title']) == 'abstract' for s in sections):
        front = sections[0]
        prose = [p for p in front['paragraphs'] if len(p['text'].split()) >= 30
                 and len(p['text']) >= 200 and '@' not in p['text']
                 and not re.match(r'^(?:keywords?|key words)\b', p['text'], re.I)]
        if prose:
            front['paragraphs'] = [p for p in front['paragraphs'] if p not in prose]
            sections.insert(1, {'section_id': front['section_id'] + '_prose',
                'title': 'Unlabeled introductory text', 'level': 1,
                'heading_method': 'retained_front_matter_prose', 'heading_page': prose[0]['page'],
                'path': ['Unlabeled introductory text'], 'kind': 'body', 'paragraphs': prose})
    if not real_headings:
        sections[0].update(title='Document (unsectioned)', path=['Document (unsectioned)'], kind='body')
        flags.append('unsectioned_fallback')
    elif len(real_headings) < 2:
        flags.append('very_few_section_headings')
    if len(real_headings) > max(50, len(pages) * 5):
        flags.append('unusually_many_headings')
    sparse = [p['page'] for p in page_stats if p['extracted_characters'] < 40]
    if len(sparse) / max(1, len(pages)) > .25:
        flags.append('many_sparse_pages_check_ocr')
    recovered = []
    for section in sections:
        recovered.append(section)
        if section['kind'] != 'references':
            continue
        floating_pages = sorted({p['page'] for p in section['paragraphs'] if p['kind'] == 'caption'})
        for page in floating_pages:
            title = f'Floating material after references (page {page})'
            paragraphs = [p for p in section['paragraphs'] if p['page'] == page]
            section['paragraphs'] = [p for p in section['paragraphs'] if p['page'] != page]
            recovered.append({'section_id': section['section_id'] + f'_float_p{page}',
                'title': title, 'level': section['level'] + 1, 'heading_method': 'float_page_recovery',
                'heading_page': page, 'path': section['path'] + [title],
                'kind': 'floating_content', 'paragraphs': paragraphs})
        if floating_pages and 'floating_content_recovered_needs_review' not in flags:
            flags.append('floating_content_recovered_needs_review')
    sections = recovered
    for section in sections:
        position, parts = 0, []
        for paragraph in section['paragraphs']:
            if parts:
                parts.append('\n\n')
                position += 2
            paragraph['char_start'] = position
            parts.append(paragraph['text'])
            position += len(paragraph['text'])
            paragraph['char_end'] = position
        section['text'] = ''.join(parts)
    return {'parser_version': PARSER_VERSION, 'parser_backend_version': pymupdf.VersionBind, 'paper_id': paper['paper_id'],
        'version': paper['selected_version'], 'title': paper['title'], 'pdf_sha256': pdf_hash,
        'source_pdf': str(pdf_path), 'page_count': len(pages), 'body_font_size': font,
        'outline_entry_count': len(outline_rows), 'removed_margin_blocks': removed,
        'pages': page_stats, 'quality_flags': flags, 'sections': sections}


HTML_SKIP = {'ltx_ERROR', 'ltx_missing', 'ltx_rdf', 'ltx_pagination', 'ltx_picture'}
HTML_ERRORS = {'ltx_ERROR', 'ltx_missing'}
HTML_SECTION_LEVELS = {'ltx_section': 1, 'ltx_subsection': 2, 'ltx_subsubsection': 3, 'ltx_appendix': 1}
HTML_RUN_IN = {'ltx_paragraph', 'ltx_subparagraph'}
HTML_HEADINGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')


def html_backend_version() -> str:
    return ','.join(f'{pkg}=={importlib.metadata.version(pkg)}' for pkg in ('beautifulsoup4', 'lxml'))


def _classes(tag) -> set:
    return set(tag.get('class') or [])


def _normalize_text(text: str) -> str:
    return _clean_line(re.sub(r'\s+', ' ', text))


def _inline(tag, ctx: dict, notes: list) -> str:
    """Text of an element in reading order; math becomes its LaTeX and footnotes are set aside."""
    from bs4 import Comment, NavigableString
    parts = []

    def walk(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            parts.append(str(node))
            return
        classes = _classes(node)
        if classes & HTML_SKIP or node.name in ('script', 'style', 'img', 'svg'):
            ctx['errors'] += bool(classes & HTML_ERRORS)
            return
        if 'ltx_note' in classes:
            content = node.find(class_='ltx_note_content')
            note = _normalize_text(_inline(content, ctx, []) if content else '')
            if note:
                notes.append(note)
            return
        if node.name == 'math':
            alt = (node.get('alttext') or node.get_text()).strip()
            if alt:
                parts.append('$' + alt + '$')
            return
        if node.name == 'br':
            parts.append(' ')
            return
        for child in node.children:
            walk(child)

    walk(tag)
    return ''.join(parts)


def _html_title(section_tag, ctx: dict) -> str:
    for child in section_tag.find_all(True, recursive=False):
        if child.name in HTML_HEADINGS and 'ltx_title' in _classes(child):
            return _normalize_text(_inline(child, ctx, []))
    return ''


def _table_text(table, ctx: dict) -> str:
    """Rows of 'cell | cell'. Empty cells and column spans keep their column position.

    LaTeXML emits either <table><tr><td> or, inside minipages, <span class=ltx_tabular><span class=ltx_tr>.
    """
    def own(row):
        owner = row.find_parent(lambda t: 'ltx_tabular' in _classes(t) or t.name == 'table')
        return owner is table
    rows = []
    for row in [r for r in table.find_all(lambda t: t.name == 'tr' or 'ltx_tr' in _classes(t)) if own(r)]:
        cells = []
        for cell in row.find_all(lambda t: t.name in ('th', 'td') or _classes(t) & {'ltx_td', 'ltx_th'}, recursive=False):
            cells.append(_normalize_text(_inline(cell, ctx, [])))
            try:
                cells.extend([''] * (int(cell.get('colspan', 1)) - 1))
            except ValueError:
                pass
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            rows.append(' | '.join(cells))
    return '\n'.join(rows)


def _assign_offsets(section: dict) -> None:
    """Join paragraphs with blank lines and record each paragraph's range in section['text']."""
    position, parts = 0, []
    for paragraph in section['paragraphs']:
        if parts:
            parts.append('\n\n')
            position += 2
        paragraph['char_start'] = position
        parts.append(paragraph['text'])
        position += len(paragraph['text'])
        paragraph['char_end'] = position
    section['text'] = ''.join(parts)


def parse_html_sections(html_path: Path, paper: dict, html_hash: str) -> dict:
    """Section/paragraph structure from arXiv's LaTeXML HTML, shaped like parse_pdf_sections.

    Headings, the bibliography and reading order come from the markup, not from layout guesses.
    Figures contribute captions only; tables become 'cell | cell' rows; math becomes $LaTeX$.
    Run-in paragraph headings stay inside their section as heading paragraphs.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8', errors='replace'), 'lxml')
    article = soup.find('article')
    if article is None:
        raise ValueError('No <article> element: not a LaTeXML document.')
    ctx = {'errors': 0}
    sections, ordinal = [], [0]
    prefix = f"{paper['paper_id']}_{paper['selected_version']}"

    def new_section(title, level, path, kind, html_id, method='html_section'):
        section = {'section_id': f'{prefix}_s{len(sections):04d}', 'title': title, 'level': level,
            'heading_method': method, 'heading_page': None, 'html_id': html_id, 'path': path,
            'kind': kind, 'paragraphs': []}
        sections.append(section)
        return section

    def current(state):
        # After a sub-section ends, trailing paragraphs of the parent open a continuation section.
        if state['current'] is None:
            state['current'] = new_section(state['title'], state['level'], state['path'], state['kind'],
                                           state['id'], 'html_section_continued')
        return state['current']

    def add(state, text, kind, element_id):
        text = text.strip()
        if not text:
            return
        ordinal[0] += 1
        current(state)['paragraphs'].append({
            'paragraph_id': 'p_' + digest_json([html_hash, ordinal[0], element_id, text])[:20],
            'page': None, 'bbox': None, 'element_id': element_id, 'text': text, 'kind': kind})

    def add_inline(state, tag, kind, element_id):
        notes = []
        add(state, _normalize_text(_inline(tag, ctx, notes)), kind, element_id)
        for note in notes:
            add(state, 'Footnote: ' + note, 'footnote', element_id)

    def figure(state, tag, element_id):
        for child in tag.find_all(True, recursive=False):
            classes = _classes(child)
            if child.name == 'figcaption':
                add_inline(state, child, 'caption', child.get('id') or element_id)
            elif child.name == 'figure':
                figure(state, child, child.get('id') or element_id)
            elif 'ltx_tabular' in classes:
                add(state, _table_text(child, ctx), 'table', child.get('id') or element_id)
            elif 'ltx_listing' in classes:
                lines = child.find_all(class_='ltx_listingline') or [child]
                add(state, '\n'.join(_normalize_text(_inline(line, ctx, [])) for line in lines), 'listing',
                    child.get('id') or element_id)
            elif not classes & HTML_SKIP and child.name in ('div', 'span', 'p'):
                tables = child.find_all(class_='ltx_tabular')
                for table in tables:
                    if table.find_parent(class_='ltx_tabular') is None:
                        add(state, _table_text(table, ctx), 'table', table.get('id') or element_id)

    def content(state, parent, element_id):
        from bs4 import NavigableString
        for child in parent.children:
            if isinstance(child, NavigableString):
                add(state, _normalize_text(str(child)), 'text', element_id)
                continue
            if not hasattr(child, 'get'):
                continue
            classes, name = _classes(child), child.name
            here = child.get('id') or element_id
            if classes & HTML_SKIP or name in ('script', 'style', 'nav'):
                ctx['errors'] += bool(classes & HTML_ERRORS)
            elif name == 'section':
                if classes & (set(HTML_SECTION_LEVELS) | {'ltx_bibliography'}):
                    walk_section(child, state['path'], state['kind'])
                    state['current'] = None
                elif classes & HTML_RUN_IN:
                    add(state, _html_title(child, ctx), 'heading', here)
                    content(state, child, here)
                else:
                    content(state, child, here)
            elif name in HTML_HEADINGS and 'ltx_title' in classes:
                if parent.name != 'section':
                    add_inline(state, child, 'heading', here)
            elif name == 'p':
                add_inline(state, child, 'text', here)
            elif name == 'table' and classes & {'ltx_equation', 'ltx_equationgroup', 'ltx_eqn_table'}:
                maths = [(m.get('alttext') or m.get_text()).strip() for m in child.find_all('math')]
                add(state, _normalize_text(' '.join(m for m in maths if m)), 'equation', here)
            elif name == 'table' or 'ltx_tabular' in classes:
                add(state, _table_text(child, ctx), 'table', here)
            elif name in ('ul', 'ol', 'dl'):
                for item in child.find_all(['li', 'dd'], recursive=False):
                    add_inline(state, item, 'list_item', item.get('id') or here)
            elif name == 'figure':
                figure(state, child, here)
            elif 'ltx_listing' in classes or name == 'pre':
                lines = child.find_all(class_='ltx_listingline') or [child]
                add(state, '\n'.join(_normalize_text(_inline(line, ctx, [])) for line in lines), 'listing', here)
            else:
                content(state, child, here)

    def walk_section(tag, parent_path, parent_kind):
        classes = _classes(tag)
        title = _html_title(tag, ctx) or 'Untitled section'
        level = next((HTML_SECTION_LEVELS[c] for c in classes if c in HTML_SECTION_LEVELS), 1)
        if 'ltx_bibliography' in classes:
            kind = 'references'
        elif 'ltx_acknowledgements' in classes:
            kind = 'acknowledgments'
        elif parent_kind == 'references':
            kind = 'references'
        else:
            kind = _section_kind(title)
        path = parent_path + [title]
        state = {'title': title, 'level': level, 'path': path, 'kind': kind, 'id': tag.get('id'),
                 'current': new_section(title, level, path, kind, tag.get('id'))}
        content(state, tag, tag.get('id'))

    abstract = article.find(class_='ltx_abstract')
    if abstract is not None:
        state = {'title': 'Abstract', 'level': 1, 'path': ['Abstract'], 'kind': 'body', 'id': abstract.get('id'),
                 'current': new_section('Abstract', 1, ['Abstract'], 'body', abstract.get('id'))}
        for child in abstract.find_all('p', recursive=False):
            add_inline(state, child, 'text', child.get('id') or abstract.get('id'))
    real = [0]

    def top_level(parent):
        for child in parent.find_all(True, recursive=False):
            classes = _classes(child)
            if child.name == 'section' and classes & {'ltx_part', 'ltx_chapter'}:
                top_level(child)  # \part/\chapter wrappers only group sections
            elif child.name == 'section' and classes & (set(HTML_SECTION_LEVELS) | {'ltx_bibliography'}):
                real[0] += 1
                walk_section(child, [], 'body')
            elif 'ltx_acknowledgements' in classes:
                walk_section(child, [], 'body')

    top_level(article)
    real = real[0]
    flags = []
    if not real:
        flags.append('no_sections_found')
    elif real < 3:
        flags.append('very_few_section_headings')
    if abstract is None:
        flags.append('no_abstract')
    if ctx['errors']:
        flags.append('latex_conversion_errors_in_body')
    for section in sections:
        _assign_offsets(section)
    if sum(len(s['text']) for s in sections if s['kind'] == 'body') < 5000:
        flags.append('very_little_body_text')
    return {'parser_version': HTML_PARSER_VERSION, 'parser_backend_version': html_backend_version(),
        'paper_id': paper['paper_id'], 'version': paper['selected_version'], 'title': paper['title'],
        'source_format': 'html', 'source_sha256': html_hash, 'source_url': paper['html']['url'],
        'source_path': paper['html']['local_path'], 'latex_error_elements': ctx['errors'],
        'quality_flags': flags, 'sections': sections}


def _common_prefix(first: list, second: list) -> list:
    shared = []
    for left, right in zip(first, second):
        if left != right:
            break
        shared.append(left)
    return shared or first[:1]


def _merge_two(first: dict, second: dict) -> dict:
    """Join two adjacent sections; the text stays a contiguous run of source paragraphs.

    The merged path is the shared ancestor path. A heading that is no longer in the path is
    written into the text as a heading paragraph rather than being lost.
    """
    path = _common_prefix(first['path'], second['path'])
    paragraphs = []
    for part in (first, second):
        if part['path'] != path and part.get('heading_method') != 'html_section_continued':
            paragraphs.append({'paragraph_id': 'p_' + digest_json(['heading', part['section_id'], part['title']])[:20],
                'page': None, 'bbox': None, 'element_id': part.get('html_id'), 'text': part['title'],
                'kind': 'heading'})
        paragraphs.extend(dict(p) for p in part['paragraphs'])
    merged = {**first, 'title': path[-1], 'path': path, 'level': min(first['level'], second['level']),
        'paragraphs': paragraphs,
        'merged_titles': first.get('merged_titles', [first['title']]) + second.get('merged_titles', [second['title']])}
    _assign_offsets(merged)
    return merged


def merge_short_sections(parsed: dict, encoding, min_tokens: int) -> dict:
    """Fold sections shorter than `min_tokens` into a neighbour under the same top-level section.

    A short section is merged forward into the next body section; a short one left over at the end
    of its top-level section is merged backward. References and acknowledgements are never merged.
    """
    if not min_tokens:
        return parsed

    def mergeable(a, b):
        return (a['kind'] == b['kind'] == 'body' and a['text'].strip() and b['text'].strip()
                and a['path'][0] == b['path'][0])

    def short(section):
        return token_count(encoding, section['text']) < min_tokens

    forward, last = [], None
    for section in parsed['sections']:
        if not section['text'].strip():
            forward.append(section)
        elif last is not None and mergeable(forward[last], section) and short(forward[last]):
            forward[last] = _merge_two(forward[last], section)
        else:
            forward.append(section)
            last = len(forward) - 1
    result, last = [], None
    for section in forward:
        if not section['text'].strip():
            result.append(section)
        elif last is not None and mergeable(result[last], section) and short(section):
            result[last] = _merge_two(result[last], section)
        else:
            result.append(section)
            last = len(result) - 1
    return {**parsed, 'sections': result, 'merged_section_count': len(parsed['sections']) - len(result)}


def should_index(section: dict, config: ChunkingConfig) -> bool:
    if section['kind'] == 'references' and not config.include_references:
        return False
    if section['kind'] == 'front_matter' and not config.include_front_matter:
        return False
    if section['kind'] == 'acknowledgments':
        return False
    return bool(section['text'].strip())


def chunk_document(parsed: dict, paper: dict, config: ChunkingConfig, encoding,
                   root: Path = DEFAULT_ROOT) -> list[dict]:
    settings_hash = digest_json({'config': dataclasses.asdict(config), 'chunker': CHUNKER_VERSION,
                                 'parser': parsed['parser_version'],
                                 'parser_backend': parsed.get('parser_backend_version'),
                                 'tokenizer_backend': tokenizer_backend(config.tokenizer_name, root)})[:12]
    chunks = []
    for section in parsed['sections']:
        if not should_index(section, config):
            continue
        spans = recursive_chunk_spans(section['text'], encoding, config.max_tokens, config.overlap_tokens)
        section_chunks = []
        for index, (start, end) in enumerate(spans):
            source_spans = []
            for paragraph in section['paragraphs']:
                left, right = max(start, paragraph['char_start']), min(end, paragraph['char_end'])
                if left < right:
                    source_spans.append({'paragraph_id': paragraph['paragraph_id'], 'page': paragraph['page'],
                        'bbox': paragraph['bbox'], 'element_id': paragraph.get('element_id'),
                        'paragraph_kind': paragraph.get('kind', 'text'), 'paragraph_char_start': left - paragraph['char_start'],
                        'paragraph_char_end': right - paragraph['char_start'],
                        'section_char_start': left, 'section_char_end': right})
            text = section['text'][start:end]
            source_sha = parsed.get('source_sha256') or parsed['pdf_sha256']
            chunk_id = 'chunk_' + digest_json([source_sha, section['section_id'], settings_hash, start, end])[:24]
            is_html = parsed.get('source_format') == 'html'
            record = {'chunk_id': chunk_id, 'paper_id': paper['paper_id'],
                'arxiv_id': paper['arxiv_id'], 'paper_title': paper['title'], 'citation': paper.get('citation'),
                'version': paper['selected_version'], 'source_format': 'html' if is_html else 'pdf',
                'source_url': parsed['source_url'] if is_html else paper['download']['versioned_pdf_url'],
                'source_path': parsed['source_path'] if is_html else paper['download']['local_path'],
                'source_sha256': source_sha,
                'section_id': section['section_id'], 'section_title': section['title'],
                'section_path': section['path'], 'section_kind': section['kind'],
                'merged_section_titles': section.get('merged_titles', []),
                'section_chunk_index': index, 'char_start': start, 'char_end': end,
                'offset_scope': 'normalized_section_text', 'text': text,
                'token_count': token_count(encoding, text), 'tokenizer': config.tokenizer_name,
                'page_numbers': sorted({s['page'] for s in source_spans if s['page'] is not None}),
                'content_types': sorted({s['paragraph_kind'] for s in source_spans}),
                'source_spans': source_spans,
                'parser_quality_flags': parsed['quality_flags'],
                'chunker_version': CHUNKER_VERSION, 'config_id': settings_hash}
            if not is_html:
                record.update(source_pdf=paper['download']['local_path'], source_pdf_sha256=source_sha)
            section_chunks.append(record)
        for index, chunk in enumerate(section_chunks):
            chunk['previous_chunk_id'] = section_chunks[index - 1]['chunk_id'] if index else None
            chunk['next_chunk_id'] = section_chunks[index + 1]['chunk_id'] if index + 1 < len(section_chunks) else None
        chunks.extend(section_chunks)
    return chunks


def validate_document_chunks(parsed: dict, chunks: list[dict], config: ChunkingConfig, encoding) -> None:
    sections = {s['section_id']: s for s in parsed['sections']}
    grouped = collections.defaultdict(list)
    ids = {c['chunk_id'] for c in chunks}
    if len(ids) != len(chunks):
        raise AssertionError('Duplicate chunk IDs.')
    for chunk in chunks:
        section = sections[chunk['section_id']]
        if chunk['text'] != section['text'][chunk['char_start']:chunk['char_end']]:
            raise AssertionError('Chunk differs from its normalized source span.')
        if not 0 < token_count(encoding, chunk['text']) <= config.max_tokens:
            raise AssertionError('Invalid token count.')
        if not chunk['source_spans']:
            raise AssertionError('Chunk has no source locator.')
        paragraphs = {p['paragraph_id']: p for p in section['paragraphs']}
        for locator in chunk['source_spans']:
            paragraph = paragraphs[locator['paragraph_id']]
            actual = paragraph['text'][locator['paragraph_char_start']:locator['paragraph_char_end']]
            expected = section['text'][locator['section_char_start']:locator['section_char_end']]
            if actual != expected:
                raise AssertionError('Paragraph and section offsets disagree.')
        grouped[chunk['section_id']].append(chunk)
    for section in parsed['sections']:
        if not should_index(section, config):
            if grouped.get(section['section_id']):
                raise AssertionError('Excluded section was indexed.')
            continue
        end = 0
        for chunk in grouped[section['section_id']]:
            if section['text'][end:chunk['char_start']].strip():
                raise AssertionError('Non-whitespace source text was lost between chunks.')
            if chunk['char_end'] <= end:
                raise AssertionError('Chunking failed to advance.')
            if chunk['char_start'] < end and token_count(encoding, section['text'][chunk['char_start']:end]) > config.overlap_tokens:
                raise AssertionError('Overlap exceeds configured budget.')
            end = chunk['char_end']
        if section['text'][end:].strip():
            raise AssertionError('Source text was lost at the end of a section.')


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def run_pipeline(root: Path = DEFAULT_ROOT, manifest_path: Path | None = None,
                 output_root: Path | None = None, config: ChunkingConfig | None = None,
                 limit: int | None = None, paper_ids: list[str] | None = None,
                 force_reparse: bool = False, progress: Callable[[str], None] | None = print) -> dict:
    """Parse, chunk, validate and save a reproducible local run.

    Token counts apply to `text`, without an additional title/section prefix.
    Offsets point into normalized section text, not file bytes or page strings.
    """
    started = time.monotonic()
    root = Path(root).resolve()
    config = config or ChunkingConfig()
    html = config.source_format == 'html'
    corpus = 'corpus_78' if html else 'corpus_80'
    manifest_path = Path(manifest_path or root / f'data/{corpus}/manifest.json').resolve()
    output_root = Path(output_root or root / f'data/processed/{corpus}').resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    papers = list(manifest['documents'])
    if paper_ids:
        requested = set(paper_ids)
        available = {p['paper_id'] for p in papers}
        if requested - available:
            raise ValueError('Unknown paper IDs: ' + ', '.join(sorted(requested - available)))
        papers = [p for p in papers if p['paper_id'] in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive.')
        papers = papers[:limit]
    if not papers:
        raise ValueError('No papers selected.')
    if len({p['paper_id'] for p in papers}) != len(papers):
        raise ValueError('Manifest contains duplicate paper IDs.')
    parser_version = HTML_PARSER_VERSION if html else PARSER_VERSION
    parser_backend = html_backend_version() if html else pymupdf.VersionBind
    source_key = 'html' if html else 'download'
    identity = {'config': dataclasses.asdict(config), 'parser': parser_version, 'chunker': CHUNKER_VERSION,
        'parser_backend': parser_backend, 'tokenizer_backend': tokenizer_backend(config.tokenizer_name, root),
        'sources': [(p['paper_id'], p['selected_version'], p[source_key]['sha256']) for p in papers]}
    run_id = f'section_recursive_{config.max_tokens}_{config.overlap_tokens}_{digest_json(identity)[:10]}'
    run_dir = output_root / 'chunks' / run_id
    cache_dir = output_root / 'parsed' / f"{parser_version}_{re.sub(r'[^A-Za-z0-9_.=-]+', '_', str(parser_backend))}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    encoding = get_tokenizer(config.tokenizer_name, root)
    hash_key = 'source_sha256' if html else 'pdf_sha256'
    all_chunks, reports = [], []
    for index, paper in enumerate(papers, 1):
        try:
            source = paper[source_key]
            source_path = (root / source['local_path']).resolve()
            if source_path.suffix.lower() != ('.html' if html else '.pdf'):
                raise ValueError(f'The manifest entry is not a {config.source_format} file.')
            source_hash = file_sha256(source_path)
            if source_hash != source['sha256']:
                raise ValueError('Source SHA-256 does not match the collection manifest.')
            stem = paper['paper_id'] + paper['selected_version']
            cache_path = cache_dir / (stem + '.json')
            parsed = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() and not force_reparse else None
            cache_used = bool(parsed and parsed.get(hash_key) == source_hash
                              and parsed.get('parser_version') == parser_version
                              and parsed.get('parser_backend_version') == parser_backend)
            if not cache_used:
                parsed = (parse_html_sections(source_path, paper, source_hash) if html
                          else parse_pdf_sections(source_path, paper, source_hash))
                if not html:
                    parsed['source_pdf'] = paper['download']['local_path']
                write_json(cache_path, parsed)
                with (cache_dir / (stem + '.txt')).open('w', encoding='utf-8', newline='\n') as stream:
                    for section in parsed['sections']:
                        stream.write(' > '.join(section['path']) + f" [{section['kind']}]\n" + section['text'] + '\n\n')
            prepared = merge_short_sections(parsed, encoding, config.min_section_tokens)
            chunks = chunk_document(prepared, paper, config, encoding, root)
            validate_document_chunks(prepared, chunks, config, encoding)
            if not chunks:
                raise ValueError('No indexable text. Inspect extraction and section recognition.')
            _write_jsonl(run_dir / 'by_paper' / (stem + '.jsonl'), chunks)
            all_chunks.extend(chunks)
            reports.append({'paper_id': paper['paper_id'], 'title': paper['title'], 'version': paper['selected_version'],
                'status': 'complete', 'source_format': config.source_format,
                'page_count': parsed.get('page_count'), 'section_count': len(parsed['sections']),
                'merged_section_count': prepared.get('merged_section_count', 0),
                'indexed_section_count': sum(should_index(s, config) for s in prepared['sections']),
                'excluded_section_titles': [s['title'] for s in prepared['sections'] if not should_index(s, config)],
                'chunk_count': len(chunks), 'quality_flags': parsed['quality_flags'], 'cache_used': cache_used,
                'latex_error_elements': parsed.get('latex_error_elements'),
                'parsed_json': str(cache_path),
                'heading_methods': dict(collections.Counter(s['heading_method'] for s in parsed['sections']))})
        except Exception as exc:
            reports.append({'paper_id': paper['paper_id'], 'title': paper['title'], 'status': 'failed',
                            'error': str(exc), 'quality_flags': ['processing_failed']})
        if progress:
            current = reports[-1]
            progress(json.dumps({'paper': index, 'total': len(papers), 'paper_id': paper['paper_id'],
                'status': current['status'], 'chunks': current.get('chunk_count', 0),
                'review_flags': current['quality_flags'], 'error': current.get('error')}, ensure_ascii=True))
    if len({c['chunk_id'] for c in all_chunks}) != len(all_chunks):
        raise AssertionError('Chunk IDs collide across documents.')
    counts = [c['token_count'] for c in all_chunks]
    errors = [r for r in reports if r['status'] == 'failed']
    review = [r for r in reports if r['quality_flags']]
    summary = {'run_id': run_id, 'status': 'complete' if not errors else 'partial_failure',
        'created_at_utc': utc_now(), 'collection_id': manifest['collection_id'],
        'input_manifest': str(manifest_path), 'config': dataclasses.asdict(config),
        'parser_version': parser_version, 'chunker_version': CHUNKER_VERSION,
        'packages': {name: importlib.metadata.version(name) for name in (
            ('beautifulsoup4', 'lxml') if html else ('PyMuPDF',)) + tokenizer_packages(config.tokenizer_name)},
        'selected_papers': len(papers), 'completed_papers': len(papers) - len(errors),
        'failed_papers': len(errors), 'papers_needing_review': len(review), 'total_chunks': len(all_chunks),
        'token_statistics': {'min': min(counts, default=0), 'max': max(counts, default=0),
            'mean': round(statistics.mean(counts), 2) if counts else 0,
            'median': statistics.median(counts) if counts else 0,
            'chunks_under_64_tokens': sum(n < 64 for n in counts)},
        'output_dir': str(run_dir), 'chunks_file': str(run_dir / 'chunks.jsonl'),
        'parsed_dir': str(cache_dir), 'elapsed_seconds': round(time.monotonic() - started, 2),
        'limitations': [
            ('Text comes from arXiv LaTeXML HTML; conversion errors in a paper show up as quality flags.' if html else
             'PDF heading and column detection are heuristic; inspect representative parsed documents.'),
            'Figures contribute captions only; images are not read. Tables are flattened to "cell | cell" rows.',
            f'Token counts use the {config.tokenizer_name} tokenizer; it must match the embedding model that reads these chunks.',
            'max_tokens applies to chunk text only; account for any added metadata prefix when embedding.',
        ]}
    _write_jsonl(run_dir / 'chunks.jsonl', all_chunks)
    _write_jsonl(run_dir / 'papers.jsonl', reports)
    write_json(run_dir / 'quality_report.json', {'status': summary['status'], 'needs_review': review,
        'failed_papers': errors, 'limitations': summary['limitations']})
    write_json(run_dir / 'summary.json', summary)
    write_json(run_dir / 'config.json', identity)
    write_json(output_root / 'latest_run.json', {'run_id': run_id, 'summary_path': str(run_dir / 'summary.json'),
        'chunks_file': str(run_dir / 'chunks.jsonl'), 'selected_papers': len(papers), 'status': summary['status']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--overlap-tokens', type=int, default=64)
    parser.add_argument('--tokenizer', default='BAAI/bge-m3')
    parser.add_argument('--include-references', action='store_true')
    parser.add_argument('--include-front-matter', action='store_true')
    parser.add_argument('--source-format', choices=['html', 'pdf'], default='html')
    parser.add_argument('--min-section-tokens', type=int, default=100)
    parser.add_argument('--force-reparse', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--paper-id', action='append')
    args = parser.parse_args()
    config = ChunkingConfig(args.max_tokens, args.overlap_tokens, args.tokenizer,
                            args.include_references, args.include_front_matter,
                            args.source_format, args.min_section_tokens)
    result = run_pipeline(args.root, args.manifest, args.output_root, config,
                          args.limit, args.paper_id, args.force_reparse)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result['failed_papers']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
