"""Reconcile the uploaded workbook with the saved course snapshot.

Produces a new merged catalog and a proposed 60-work shortlist. Does not delete,
download, or modify any existing PDFs, nor overwrite either source workbook.
"""
from __future__ import annotations

import collections
import copy
import difflib
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import openpyxl

from collect_papers import arxiv_id, identity, normalize

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path(r'C:\Users\boyue\Downloads\modern NLP&LLM papers to be used.xlsx')
OUT = ROOT / 'outputs' / 'paper_reconciliation_20261026'

# Selection is an explicit proposal based on the LLM scope and complementary
# question opportunities, not a measured quality score or final corpus approval.
CORE = {
    '架构与分词': [
        ('1706.03762', 'Transformer 主线基础，连接注意力、推理与长上下文题。'),
        ('1607.06450', '与 RMSNorm 形成归一化设计比较。'),
        ('1910.07467', '与 LayerNorm 形成归一化设计比较。'),
        ('2002.05202', '补充 Transformer 前馈层设计。'),
        ('1508.07909', '保留分词基础，避免语料只覆盖模型结构。'),
        ('1701.06538', 'MoE 基础，与 Switch、Mixtral、DeepSeekMoE 串联。'),
        ('2101.03961', 'MoE 路由与扩展的比较材料。'),
        ('2401.04088', '保留代表性 MoE 模型，与其他 MoE 论文形成比较。'),
        ('2401.06066', '补充 MoE 专家设计的比较材料。'),
        ('2404.19737', '补充多 token 预测这一训练设计维度。'),
        ('2312.00752', '补充选择性状态空间模型，与 Transformer 主线形成对照。'),
        ('2405.21060', '保留 SSM 与 Transformer 联系的材料。'),
        ('2006.16236', '补充线性注意力与循环形式之间的联系。'),
        ('2402.08164', '保留架构局限材料，用于考察条件与限制的回答。'),
    ],
    '长上下文': [
        ('2104.09864', '位置编码基础，与 YaRN、LongRoPE 组成上下文扩展主题。'),
        ('2309.00071', '与 RoFormer、LongRoPE 形成方法比较。'),
        ('2402.13753', '与 RoFormer、YaRN 形成长上下文方法比较。'),
    ],
    '优化器': [
        ('1412.6980', '保留 Adam 作为优化器比较的基础。'),
        ('1711.05101', '保留权重衰减设计，与 Adam 形成比较。'),
        ('1802.09568', '保留矩阵预条件优化器材料。'),
        ('2409.11321', '与 Shampoo、Adam 形成优化器比较。'),
        ('2302.06675', '保留优化算法发现这一不同研究角度。'),
        ('2502.16982', '补充面向 LLM 训练的 Muon 材料。'),
    ],
    'Scaling Laws': [
        ('2001.08361', '与 Chinchilla 形成参数、数据和计算预算的比较题。'),
        ('2203.15556', '与 Kaplan scaling laws 形成可对照的研究主线。'),
    ],
    '训练数据': [
        ('2406.11794', '保留数据集构建与比较材料。'),
        ('2402.16827', '保留一篇数据选择综述，支持概念覆盖与综合题。'),
        ('2501.00656', '保留一份开放模型报告，连接数据与训练流程。'),
        ('2506.04178', '补充推理数据配方，与训练和推理主题连接。'),
    ],
    'PEFT': [
        ('2106.09685', '与 Prefix、Prompt Tuning、QLoRA 组成参数高效微调比较。'),
        ('2101.00190', '与 LoRA、Prompt Tuning 形成适配方式比较。'),
        ('2104.08691', '与 Prefix-Tuning 形成连续提示方法比较。'),
        ('2305.14314', '连接 PEFT 和量化，支持跨主题综合题。'),
    ],
    '量化': [
        ('2208.07339', '与 GPTQ、AWQ 组成量化方法比较。'),
        ('2210.17323', '保留代表性训练后量化材料。'),
        ('2306.00978', '与 GPTQ、LLM.int8 形成量化方法比较。'),
    ],
    '后训练与RL': [
        ('2203.02155', '保留指令微调与人类反馈训练的核心材料。'),
        ('2305.18290', '与 RLHF、PPO 形成偏好优化比较。'),
        ('1707.06347', '为后训练相关问题提供 PPO 基础。'),
        ('2402.03300', '保留数学推理与 GRPO 主线材料。'),
        ('2501.12948', '连接强化学习、推理与蒸馏主题。'),
        ('2503.14476', '保留一份大规模 LLM RL 系统材料。'),
        ('2402.14740', '保留 REINFORCE 风格方法，与 PPO 主线比较。'),
        ('2503.20783', '保留批判性分析，支持限制与实验条件类问题。'),
    ],
    '推理与测试时计算': [
        ('2201.11903', '与 Zero-Shot CoT、Self-Consistency 形成推理方法比较。'),
        ('2205.11916', '与 CoT few-shot 设置形成提示方式比较。'),
        ('2203.11171', '与 CoT 形成生成与答案聚合方式比较。'),
        ('2203.14465', '补充通过生成推理进行训练的研究路线。'),
        ('2412.01951', '保留自改进相关材料，连接训练与推理主题。'),
        ('2408.03314', '补充测试时计算预算这一比较维度。'),
    ],
    '推理系统': [
        ('2211.17192', '保留 speculative decoding 代表材料。'),
        ('2205.14135', '保留注意力计算与访存优化材料。'),
        ('2309.06180', '补充服务阶段的内存管理材料。'),
    ],
    '工具使用': [
        ('2210.03629', '与 Toolformer 形成推理、行动和工具使用的比较。'),
        ('2302.04761', '与 ReAct 形成工具使用机制比较。'),
    ],
    '蒸馏': [
        ('1503.02531', '保留知识蒸馏基础，连接较新的语言模型蒸馏工作。'),
        ('2306.13649', '补充语言模型 on-policy distillation 材料。'),
    ],
    '语言扩散模型': [
        ('2205.14217', '保留语言生成相关扩散方法，控制通用图像扩散的范围。'),
        ('2406.07524', '保留 masked diffusion language model 材料。'),
    ],
    '安全与水印': [
        ('2301.10226', '保留一个与语言模型直接相关的安全主题入口。'),
    ],
}

DEFERRED_USER = {
    '2207.09238': '偏形式化算法描述；首版保留 Transformer 原文，后续增加算法细节题时再纳入。',
    '2406.08466': '首版已保留两篇语言模型 scaling laws，线性回归理论分析放入扩展池。',
    '1904.09237': '偏 Adam 收敛理论；首版先覆盖优化器机制与训练应用。',
    '2402.16788': 'Hessian 解释角度作为扩展；首版优化器比较已有六篇材料。',
    '2409.20325': '优化器理论与统一视角作为扩展，避免首版优化器主题占比过大。',
    '2602.02710': '首版后训练主题已有八篇；该 RL 路线作为后续扩展。',
    '2508.02193': '语言扩散首版先保留两篇基础材料，具体大规模模型报告作为扩展。',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = ROOT / 'data/sources/user_papers_20261026.xlsx'
    if not snapshot.exists():
        shutil.copy2(INPUT, snapshot)
    course_catalog = json.loads((ROOT / 'data/paper_catalog.json').read_text(encoding='utf-8'))
    wb = openpyxl.load_workbook(INPUT, data_only=False)
    occurrences, ignored, anomalies = [], [], []
    for sheet in wb:
        for cells in sheet.iter_rows():
            values = [cell.value for cell in cells]
            if not any(v is not None for v in values):
                continue
            title = str(values[0] or '').strip()
            link_cell = cells[1] if len(cells) > 1 else None
            link = (link_cell.hyperlink.target if link_cell is not None and link_cell.hyperlink else None) or (str(values[1]).strip() if len(values) > 1 and values[1] else '')
            aid = arxiv_id(link)
            if not aid:
                ignored.append({'sheet': sheet.title, 'row': cells[0].row, 'values': values})
                continue
            base = re.sub(r'v\d+$', '', aid)
            version = str(values[4] or '').strip() if sheet.title == 'Sheet1' else ''
            raw_id = values[3] if sheet.title == 'Sheet1' else None
            id_cell_text = str(raw_id) if raw_id is not None else ''
            if raw_id is not None and id_cell_text != base:
                # Numeric storage can drop meaningful trailing zeroes. URLs are authoritative.
                anomalies.append({'kind': 'numeric_identifier_format', 'sheet': sheet.title,
                                  'cell': cells[3].coordinate, 'stored_value': raw_id,
                                  'canonical_id_from_url': base})
            occurrences.append({'source_id': 'user_workbook', 'sheet': sheet.title,
                'row': cells[0].row, 'title_cell': cells[0].coordinate,
                'link_cell': cells[1].coordinate, 'title': title, 'url': link,
                'arxiv_id': base, 'version_as_listed': version or None,
                'topic': str(values[2] or '') if len(values) > 2 else '',
                'lecture_as_listed': values[3] if sheet.title == 'Sheet2' else None})

    user_groups = collections.defaultdict(list)
    for entry in occurrences:
        user_groups['arxiv:' + entry['arxiv_id']].append(entry)
    records = {r['dedupe_key']: copy.deepcopy(r) for r in course_catalog['documents']}
    for key, entries in user_groups.items():
        first = entries[0]
        if key not in records:
            records[key] = {'paper_id': 'arxiv_' + first['arxiv_id'], 'dedupe_key': key,
                'title': first['title'], 'arxiv_id': first['arxiv_id'], 'citation': '',
                'year_as_listed': None,
                'document_type': 'research_paper', 'source_url': first['url'],
                'download_url': 'https://arxiv.org/pdf/' + first['arxiv_id'],
                'sources': [], 'download': {'status': 'not_downloaded'},
                'selection_status': 'candidate'}
        record = records[key]
        record['title_variants'] = list(dict.fromkeys([record['title']] + [e['title'] for e in entries]))
        record['user_entries'] = entries
        record['sources'].extend(copy.deepcopy(entries))
        record['user_version_labels'] = list(dict.fromkeys(e['version_as_listed'] for e in entries if e['version_as_listed']))
        record['preferred_download_url_from_user'] = ('https://arxiv.org/pdf/' + first['arxiv_id'] + record['user_version_labels'][0]) if record['user_version_labels'] else None
        record['version_note'] = ('用户表格指定版本；本地课程 PDF 是未指定修订号的哈希快照，尚未确认与该版本相同。' if record['user_version_labels'] and record['download'].get('local_path')
                                  else '保留用户表格指定版本，尚未下载。' if record['user_version_labels'] else '未指定修订号。')

    core_by_id = {aid: (topic, reason) for topic, items in CORE.items() for aid, reason in items}
    assert len(core_by_id) == 60, len(core_by_id)
    missing = set(core_by_id) - {r.get('arxiv_id') for r in records.values()}
    assert not missing, missing
    duplicate_rows = []
    for r in records.values():
        has_course = any(s['source_id'] == 'cs395t_f26' for s in r['sources'])
        has_user = bool(r.get('user_entries'))
        r['source_presence'] = 'both' if has_course and has_user else 'user_only' if has_user else 'course_only'
        if r['source_presence'] == 'both':
            duplicate_rows.append({'paper_id': r['paper_id'], 'arxiv_id': r['arxiv_id'],
                'course_title': r['title'], 'user_title': r['user_entries'][0]['title'],
                'user_locations': '; '.join(e['sheet'] + '!' + e['title_cell'] for e in r['user_entries']),
                'match_method': 'exact_arxiv_id_ignoring_revision',
                'course_url': r['source_url'], 'user_url': r['user_entries'][0]['url'],
                'user_version_labels': r.get('user_version_labels', []),
                'version_note': r.get('version_note', '')})
        aid = r.get('arxiv_id')
        if aid in core_by_id:
            topic, reason = core_by_id[aid]
            r['recommendation'] = 'core'
            r['review_topic'] = topic
            r['selection_reason'] = reason
        else:
            source = next((s for s in r['sources'] if s['source_id'] == 'cs395t_f26'), {})
            scope = source.get('scope_bucket', '')
            r['recommendation'] = 'reserve'
            r['review_topic'] = (r.get('user_entries') or [{}])[0].get('topic') or scope
            if aid in DEFERRED_USER:
                r['selection_reason'] = DEFERRED_USER[aid]
            elif r['document_type'] in ('book', 'monograph'):
                r['selection_reason'] = '书籍或长篇专著，首版优先采用范围清楚的论文，暂放扩展池。'
            elif scope in ('deep_learning_background', 'math_background'):
                r['selection_reason'] = '通用深度学习历史或数学背景，与首版 LLM 研究问答距离较远。'
            elif scope == 'diffusion_and_flow_models':
                r['selection_reason'] = '首版只保留语言生成直接相关的两篇扩散论文，暂不展开通用扩散与流模型。'
            elif scope == 'safety_and_alignment':
                r['selection_reason'] = '首版安全主题保留语言模型水印，其他安全、审计和解释性子方向后续扩展。'
            elif r['document_type'] == 'web_article':
                r['selection_reason'] = '首版优先论文与技术报告；网页文章保留在候选池。'
            elif '2026-11-18' in [s.get('lecture_date') for s in r['sources']]:
                r['selection_reason'] = '首版保留 OLMo 2 和 DeepSeek-R1 等代表材料，暂不同时加入大量完整模型报告。'
            else:
                r['selection_reason'] = '该子方向已有代表材料，首版控制数量，其余作为后续扩展。'
        r['approval_status'] = 'proposed_not_frozen'
        r.setdefault('user_entries', [])
        r.setdefault('user_version_labels', [])
        r.setdefault('version_note', '沿用课程下载快照，以 SHA-256 标识。')

    # Exact title agreement with conflicting identifiers is flagged, never auto-merged.
    title_groups = collections.defaultdict(list)
    for record in records.values():
        title_groups[normalize(record['title'])].append(record['paper_id'])
    title_collisions = [group for group in title_groups.values() if len(group) > 1]
    core = [r for r in records.values() if r['recommendation'] == 'core']
    presence = collections.Counter(r['source_presence'] for r in records.values())
    stats = {
        'user_rows_by_sheet': dict(collections.Counter(e['sheet'] for e in occurrences)),
        'user_raw_entries': len(occurrences), 'user_unique_works': len(user_groups),
        'user_internal_duplicate_groups': sum(len(v) > 1 for v in user_groups.values()),
        'course_unique_entries': len(course_catalog['documents']),
        'shared_works': presence['both'], 'user_only': presence['user_only'],
        'course_only': presence['course_only'], 'merged_unique_entries': len(records),
        'proposed_core': len(core), 'reserve': len(records) - len(core),
        'core_topic_counts': dict(collections.Counter(r['review_topic'] for r in core)),
        'core_source_presence': dict(collections.Counter(r['source_presence'] for r in core)),
        'core_local_pdf_count': sum(r['download']['status'] == 'downloaded_pdf' for r in core),
        'core_not_downloaded_count': sum(r['download']['status'] == 'not_downloaded' for r in core),
        'pinned_version_overlap_needs_verification': sum(bool(r.get('user_version_labels')) and r['source_presence'] == 'both' for r in records.values()),
        'numeric_identifier_format_issues': len(anomalies),
    }
    assert stats['merged_unique_entries'] == stats['user_unique_works'] + stats['course_unique_entries'] - stats['shared_works']
    output = {
        'schema_version': '1.0', 'catalog_version': 'merged_candidates_2026-10-26_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'merged_with_proposed_selection_not_frozen',
        'source_snapshots': [
            {'source_id': 'user_workbook', 'local_path': snapshot.relative_to(ROOT).as_posix(),
             'sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()},
            {'source_id': 'cs395t_f26', 'catalog': 'data/paper_catalog.json', 'reading_list': 'data/sources/cs395t-f26-readings.json',
             'snapshot_date': '2026-10-23', 'live_site_refreshed': False}],
        'method': {
            'dedupe': 'Exact arXiv work ID from URL, ignoring revisions. Preserve versions separately.',
            'identifier_policy': 'Treat identifiers as strings. Recover numeric truncation from supplied URLs.',
            'selection': 'Proposed 60-work LLM-focused first corpus. Keep the full union as reserve; no file deletions.',
            'review_scope': 'Topic, metadata and source-list review; not a full-text comparative review of every work.',
        },
        'statistics': stats, 'numeric_identifier_issues': anomalies,
        'normalized_title_collisions': title_collisions,
        'user_internal_duplicates': [v for v in user_groups.values() if len(v) > 1],
        'ignored_nonpaper_rows': ignored, 'overlaps': duplicate_rows,
        'documents': list(records.values()),
    }
    (ROOT / 'data/paper_catalog_merged_20261026.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shortlist = {'catalog_version': output['catalog_version'], 'approval_status': 'proposed_not_frozen',
        'document_count': len(core), 'documents': core}
    (ROOT / 'data/paper_shortlist_20261026.json').write_text(json.dumps(shortlist, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'reconciliation_summary.json').write_text(json.dumps({k:v for k,v in output.items() if k != 'documents'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=True, indent=2))
    print('OVERLAPS')
    for match in duplicate_rows:
        print(match['arxiv_id'] + ' | ' + match['course_title'])
    print('CORE')
    for r in core:
        print(r['arxiv_id'] + ' | ' + r['review_topic'] + ' | ' + r['title'])


if __name__ == '__main__':
    main()
