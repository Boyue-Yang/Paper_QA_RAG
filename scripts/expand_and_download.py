"""Expand the previous 60-work selection to 80 and collect versioned local PDFs.

--prepare creates the new selection without network access.
--download resumes collection: reuse verified local files; use the arXiv metadata
API to resolve unknown versions, then download PDFs by their versioned URLs.
Original catalogs and PDFs are preserved.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import io
import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from collect_papers import arxiv_id, normalize

logging.getLogger('pypdf').setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/corpus_80'
CATALOG = ROOT / 'data/paper_catalog_80_20261026.json'
ATOM = {'a': 'http://www.w3.org/2005/Atom'}
ADDITIONS = {
    '2207.09238': ('架构与分词', '扩至80篇后补入形式化 Transformer 描述，支持算法细节题。'),
    '2406.08466': ('Scaling Laws', '增加线性回归理论视角，与语言模型的经验 scaling laws 对照。'),
    '1904.09237': ('优化器', '增加 Adam 收敛与失败条件的理论材料。'),
    '2402.16788': ('优化器', '补充 Hessian 视角，支持优化器机制与条件解释题。'),
    '2409.20325': ('优化器', '补充优化器统一视角，与具体算法论文形成综合题。'),
    '2602.02710': ('后训练与RL', '扩展 LLM 强化学习研究路线，保留原表条目。'),
    '2508.02193': ('语言扩散模型', '增加大规模语言扩散模型实例，连接方法与系统报告。'),
    '2111.00396': ('架构与分词', '补充 S4，与 Mamba 和状态空间对偶论文串联。'),
    '2305.14342': ('优化器', '补充语言模型预训练中的二阶优化器材料。'),
    '2411.15124': ('后训练与RL', '增加开放语言模型后训练流程的综合材料。'),
    '2501.12599': ('后训练与RL', '增加强化学习与推理扩展的模型实例。'),
    '2601.18734': ('蒸馏', '扩展到自蒸馏，与基础蒸馏及 on-policy distillation 对照。'),
    '2505.23884': ('推理与测试时计算', '补充测试时训练，与测试时计算预算主题区分和比较。'),
    '2407.04620': ('推理与测试时计算', '补充测试时学习与隐藏状态设计的联系。'),
    '2410.21228': ('PEFT', '增加 LoRA 与全量微调的直接比较材料。'),
    '2308.08747': ('PEFT', '补充持续微调的遗忘问题，支持局限与条件类问答。'),
    '2306.09194': ('安全与水印', '增加水印可检测性研究，与现有水印论文对照。'),
    '2401.05566': ('安全与水印', '增加安全训练与后门行为的研究材料。'),
    '2406.04093': ('安全与水印', '补充稀疏自编码器与解释性评测材料。'),
    '2507.11473': ('安全与水印', '补充推理过程监控的机会与局限材料。'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def update_stats(catalog):
    selected = [r for r in catalog['documents'] if r['recommendation'] == 'core']
    stats = catalog['statistics']
    stats.update(proposed_core=len(selected), reserve=len(catalog['documents']) - len(selected),
        core_topic_counts=dict(collections.Counter(r['review_topic'] for r in selected)),
        core_source_presence=dict(collections.Counter(r['source_presence'] for r in selected)),
        core_local_pdf_count=sum(r['download'].get('status') == 'downloaded_pdf' for r in selected),
        core_not_downloaded_count=sum(r['download'].get('status') != 'downloaded_pdf' for r in selected),
        collection_80_downloaded_count=sum(r.get('collection_download', {}).get('status') == 'complete' for r in selected),
        collection_80_version_resolved_count=sum(bool(r.get('selected_version')) for r in selected),
        pinned_version_overlap_needs_verification=sum(bool(r['user_version_labels']) and r['source_presence'] == 'both' and not r.get('version_verified') for r in selected))
    catalog['updated_at_utc'] = now()
    write_json(CATALOG, catalog)
    write_json(DIRECTORY / 'manifest.json', {
        'collection_id': 'llm_papers_80_20261026_v1',
        'status': 'complete' if stats['collection_80_downloaded_count'] == 80 else 'in_progress',
        'document_count': len(selected), 'downloaded_count': stats['collection_80_downloaded_count'],
        'updated_at_utc': catalog['updated_at_utc'], 'documents': selected,
    })


def prepare():
    previous = json.loads((ROOT / 'data/paper_catalog_merged_20261026.json').read_text(encoding='utf-8'))
    catalog = copy.deepcopy(previous)
    original = {r['arxiv_id'] for r in catalog['documents'] if r['recommendation'] == 'core'}
    assert len(original) == 60 and len(ADDITIONS) == 20 and not (original & set(ADDITIONS))
    for record in catalog['documents']:
        record['selection_stage'] = 'original_60' if record['recommendation'] == 'core' else 'reserve'
        if record.get('arxiv_id') in ADDITIONS:
            record['recommendation'] = 'core'
            record['review_topic'], record['selection_reason'] = ADDITIONS[record['arxiv_id']]
            record['selection_stage'] = 'added_20'
        if record['recommendation'] == 'core':
            record['approval_status'] = 'selected_for_collection'
    chosen = [r for r in catalog['documents'] if r['recommendation'] == 'core']
    assert len(chosen) == 80 and all(r.get('arxiv_id') for r in chosen)
    assert sum(bool(r['user_entries']) for r in chosen) == 59
    catalog['catalog_version'] = 'merged_candidates_80_2026-10-26_v1'
    catalog['status'] = 'selected_80_downloading'
    catalog['method']['selection'] = 'Preserve the original 60 works and add 20 existing reserve works, as requested. All 80 are on arXiv.'
    catalog['selection_history'] = {
        'previous_catalog': 'data/paper_catalog_merged_20261026.json',
        'previous_count': 60, 'added_count': 20,
        'added_arxiv_ids': list(ADDITIONS),
    }
    update_stats(catalog)
    print(json.dumps(catalog['statistics'], ensure_ascii=True, indent=2), flush=True)
    return catalog


def pdf_version(path, aid):
    reader = PdfReader(path)
    text = reader.pages[0].extract_text() or ''
    compact = re.sub(r'\s+', '', text)
    versions = set(re.findall(re.escape(aid) + r'(v[1-9]\d*)', compact))
    return (next(iter(versions)) if len(versions) == 1 else None), reader, text


class RateLimited(RuntimeError):
    pass


class Client:
    def __init__(self):
        self.last_finished = 0.0

    def get(self, url, attempts=3, cap=100 * 1024 * 1024):
        for attempt in range(attempts):
            wait = max(0.0, 3.1 - (time.monotonic() - self.last_finished))
            if wait:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={
                'User-Agent': 'PaperQAResearchCollection/0.2 (academic paper collection; sequential requests)',
                'Accept': 'application/atom+xml,application/pdf,text/html;q=0.8'})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = response.read(cap + 1)
                    if len(data) > cap:
                        raise ValueError('Response exceeds configured file-size limit.')
                    return data, response.url, response.headers.get('Content-Type', '')
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    raise RateLimited('HTTP 429: stopped requests; resume later.') from exc
                if exc.code not in (500, 502, 503, 504) or attempt + 1 == attempts:
                    raise
                print(json.dumps({'event': 'retry', 'url': url, 'http_status': exc.code, 'attempt': attempt + 1}), flush=True)
                time.sleep(5 * (attempt + 1))
            except (TimeoutError, urllib.error.URLError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(5 * (attempt + 1))
            finally:
                self.last_finished = time.monotonic()


def metadata_lookup(client, records):
    if not records:
        return
    ids = [r['arxiv_id'] for r in records]
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})
    cache = DIRECTORY / 'metadata_api.xml'
    try:
        if cache.exists():
            data = cache.read_bytes()
        else:
            print(json.dumps({'event': 'metadata_api', 'count': len(ids)}), flush=True)
            data, resolved, content_type = client.get(url, attempts=2, cap=8 * 1024 * 1024)
            feed = ET.fromstring(data)
            cache.write_bytes(data)
        feed = ET.fromstring(data)
        found = {}
        for entry in feed.findall('a:entry', ATOM):
            full = arxiv_id(entry.findtext('a:id', default='', namespaces=ATOM))
            if not full:
                continue
            match = re.fullmatch(r'(.+?)(v\d+)', full)
            if not match:
                continue
            aid, version = match.groups()
            found[aid] = {'version': version,
                'title': ' '.join(entry.findtext('a:title', default='', namespaces=ATOM).split()),
                'authors': [a.findtext('a:name', default='', namespaces=ATOM) for a in entry.findall('a:author', ATOM)],
                'published': entry.findtext('a:published', default='', namespaces=ATOM),
                'updated': entry.findtext('a:updated', default='', namespaces=ATOM),
                'api_url': url, 'cached_response': cache.relative_to(ROOT).as_posix()}
        for record in records:
            if record['arxiv_id'] in found:
                record['arxiv_metadata'] = found[record['arxiv_id']]
                record['selected_version'] = found[record['arxiv_id']]['version']
                record['version_policy'] = 'latest_resolved_by_api'
    except RateLimited:
        raise
    except Exception as exc:
        # PDF downloads do not depend on the metadata API. Unknown versions can
        # also be read from the arXiv stamp in the downloaded PDF.
        print(json.dumps({'event': 'metadata_api_unavailable', 'error': str(exc)[:250], 'fallback': 'PDF URL plus version stamp'}), flush=True)


def check_pdf(data, aid, expected_version=None):
    if not data.lstrip().startswith(b'%PDF-'):
        raise ValueError('Response is not a PDF; not saved as a paper.')
    actual_version, reader, text = pdf_version(io.BytesIO(data), aid)
    if not len(reader.pages):
        raise ValueError('PDF has no pages.')
    if actual_version and expected_version and actual_version != expected_version:
        raise ValueError(f'Requested {expected_version}, PDF stamp reports {actual_version}.')
    version = expected_version or actual_version
    if not version:
        raise ValueError('Cannot determine exact version from API, URL, or PDF stamp.')
    return version, actual_version, reader, text


def download(catalog):
    client = Client()
    pdf_dir = DIRECTORY / 'pdfs'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chosen = [r for r in catalog['documents'] if r['recommendation'] == 'core']
    reusable = {}
    for record in chosen:
        versions = record.get('user_version_labels', [])
        if len(versions) > 1:
            raise ValueError('Conflicting user versions: ' + record['paper_id'])
        if versions:
            record['selected_version'] = versions[0]
            record['version_policy'] = 'user_specified'
        old = record.get('download', {})
        if old.get('status') == 'downloaded_pdf' and old.get('local_path'):
            source = ROOT / old['local_path']
            if source.exists() and sha256(source) == old['sha256']:
                stamp, reader, text = pdf_version(source, record['arxiv_id'])
                if stamp and (not versions or stamp == versions[0]):
                    record['selected_version'] = stamp
                    record.setdefault('version_policy', 'preserve_local_pdf_stamp')
                    reusable[record['paper_id']] = source
    unresolved = [r for r in chosen if not r.get('selected_version')]
    print(json.dumps({'event': 'plan', 'selected': len(chosen), 'reusable': len(reusable), 'unknown_versions': len(unresolved)}), flush=True)
    metadata_lookup(client, unresolved)
    update_stats(catalog)
    results = collections.Counter()
    for index, record in enumerate(chosen, 1):
        aid = record['arxiv_id']
        version = record.get('selected_version')
        url = 'https://arxiv.org/pdf/' + aid + (version or '')
        try:
            if record['paper_id'] in reusable:
                source = reusable[record['paper_id']]
                data = source.read_bytes()
                action = 'reused'
                final_url = record['download'].get('resolved_url', url)
                content_type = 'application/pdf'
            else:
                data, final_url, content_type = client.get(url)
                action = 'downloaded'
            version, stamp, reader, text = check_pdf(data, aid, version)
            path = pdf_dir / (record['paper_id'] + version + '.pdf')
            digest = hashlib.sha256(data).hexdigest()
            if not path.exists() or sha256(path) != digest:
                temp = path.with_suffix('.pdf.part')
                temp.write_bytes(data)
                temp.replace(path)
            tokens = [t for t in normalize(record['title']).split() if len(t) > 3]
            normalized = normalize(text)
            coverage = sum(t in normalized for t in tokens) / len(tokens) if tokens else 0
            record['selected_version'] = version
            record['version_verified'] = True
            record.setdefault('version_policy', 'resolved_from_downloaded_pdf_stamp')
            record['previous_download'] = record.get('previous_download') or copy.deepcopy(record.get('download'))
            record['download'] = {
                'status': 'downloaded_pdf', 'local_path': path.relative_to(ROOT).as_posix(),
                'requested_url': url, 'resolved_url': final_url, 'versioned_pdf_url': 'https://arxiv.org/pdf/' + aid + version,
                'selected_version': version, 'version_evidence': 'PDF arXiv stamp' if stamp else 'explicit versioned PDF URL',
                'pdf_stamp_version': stamp, 'bytes': len(data), 'sha256': digest,
                'page_count': len(reader.pages), 'content_type': content_type, 'verified_at_utc': now(),
                'title_token_coverage_first_page': round(coverage, 3),
                'identity_review': 'title_tokens_found' if coverage >= .6 else 'check_title',
            }
            record['collection_download'] = {'status': 'complete', 'action': action, 'verified_at_utc': now()}
            record['version_note'] = ('已按表格指定 ' + version + ' 保存并校验。' if record['user_version_labels']
                                      else '已固定 ' + version + '，记录 PDF 哈希。')
            results[action] += 1
        except RateLimited:
            update_stats(catalog)
            raise
        except Exception as exc:
            record['collection_download'] = {'status': 'failed', 'error': str(exc)[:500], 'attempted_at_utc': now()}
            results['failed'] += 1
        update_stats(catalog)
        print(json.dumps({'completed': index, 'total': 80, 'id': aid, 'version': record.get('selected_version'),
            'result': record['collection_download']}, ensure_ascii=True), flush=True)
    catalog['status'] = 'selected_80_downloaded' if not results['failed'] else 'selected_80_partially_downloaded'
    catalog['download_run_summary'] = dict(results)
    for overlap in catalog['overlaps']:
        record = next(r for r in catalog['documents'] if r['paper_id'] == overlap['paper_id'])
        overlap['version_note'] = record['version_note']
    update_stats(catalog)
    audit(catalog)


def audit(catalog):
    chosen = [r for r in catalog['documents'] if r['recommendation'] == 'core']
    errors = []
    for record in chosen:
        info = record['download']
        if record.get('collection_download', {}).get('status') != 'complete':
            errors.append({'paper_id': record['paper_id'], 'error': record.get('collection_download', {}).get('error', 'Incomplete')})
            continue
        path = ROOT / info['local_path']
        if sha256(path) != info['sha256']:
            errors.append({'paper_id': record['paper_id'], 'error': 'SHA-256 mismatch'})
        if record['user_version_labels'] and record['selected_version'] != record['user_version_labels'][0]:
            errors.append({'paper_id': record['paper_id'], 'error': 'User version mismatch'})
    report = {'collection_id': 'llm_papers_80_20261026_v1', 'document_count': len(chosen),
        'pdf_count': len(list((DIRECTORY / 'pdfs').glob('*.pdf'))),
        'complete_count': sum(r.get('collection_download', {}).get('status') == 'complete' for r in chosen),
        'unique_work_ids': len({r['arxiv_id'] for r in chosen}),
        'version_resolved_count': sum(bool(r.get('selected_version')) for r in chosen),
        'user_pinned_versions': sum(bool(r['user_version_labels']) for r in chosen),
        'original_60_preserved': sum(r['selection_stage'] == 'original_60' for r in chosen) == 60,
        'added_count': sum(r['selection_stage'] == 'added_20' for r in chosen),
        'total_pdf_bytes': sum(r['download'].get('bytes', 0) for r in chosen if r.get('collection_download', {}).get('status') == 'complete'),
        'identity_review_needed': [r['paper_id'] for r in chosen if r['download'].get('identity_review') == 'check_title'],
        'errors': errors, 'verified_at_utc': now()}
    write_json(DIRECTORY / 'audit.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    catalog = prepare() if args.prepare else json.loads(CATALOG.read_text(encoding='utf-8'))
    if args.download:
        download(catalog)
    elif args.audit:
        audit(catalog)


if __name__ == '__main__':
    main()
