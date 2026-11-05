import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
function argument(name, fallback) { const i=args.indexOf(name); return i>=0 ? args[i+1] : fallback; }
const out = path.resolve(root, argument('--output-dir','outputs/paper_reconciliation_20261026'));
const outputName = argument('--output-name','paper_reconciliation.xlsx');
const catalog = JSON.parse(await fs.readFile(path.resolve(root,argument('--catalog','data/paper_catalog_merged_20261026.json')), 'utf8'));
const stats = catalog.statistics;
const expanded = stats.proposed_core === 80;
const topicOrder = ['架构与分词','长上下文','优化器','Scaling Laws','训练数据','PEFT','量化','后训练与RL','推理与测试时计算','推理系统','工具使用','蒸馏','语言扩散模型','安全与水印'];
const origin = { both: '双方', user_only: '仅表格', course_only: '仅课程' };
const selection = { core: '首批', reserve: '扩展' };
const docs = [...catalog.documents].sort((a,b) => {
  const tier = Number(a.recommendation !== 'core') - Number(b.recommendation !== 'core');
  if (tier) return tier;
  const topic = topicOrder.indexOf(a.review_topic) - topicOrder.indexOf(b.review_topic);
  return topic || (a.arxiv_id || a.paper_id).localeCompare(b.arxiv_id || b.paper_id);
});
const wb = Workbook.create();
const sheet = wb.worksheets.add('候选筛选');
const overlap = wb.worksheets.add('重复论文');
const end = 12 + docs.length;

function base(s, range) {
  s.showGridLines = false;
  s.getRange(range).format.font = { name: 'Arial', size: 10, color: '#202A35' };
  s.getRange(range).format.verticalAlignment = 'center';
}
function header(s, range) {
  s.getRange(range).format = {
    fill: '#254E70', font: { name:'Arial',size:10,bold:true,color:'#FFFFFF' },
    horizontalAlignment:'center',verticalAlignment:'center',rowHeight:32,
  };
}
function widths(s, values, last) {
  values.forEach((width,i) => {
    const letter = String.fromCharCode(65+i);
    s.getRange(`${letter}1:${letter}${last}`).format.columnWidth = width;
  });
}
base(sheet, `A1:N${end}`);
sheet.tabColor = '#254E70';
sheet.getRange('A2').values = [[expanded ? 'Paper QA 80篇语料清单' : 'Paper QA 语料筛选建议']];
sheet.getRange('A2').format.font = {name:'Arial',size:14,bold:true};
sheet.getRange('A4:B10').values = [
  ['表格论文',null],['课程快照条目',null],['双方重复',null],['合并后条目',null],
  ['首批建议',null],['扩展池',null],['首批已有本地PDF',null],
];
sheet.getRange('B4:B10').formulas = [
  [`=COUNTIFS(E13:E${end},"双方")+COUNTIFS(E13:E${end},"仅表格")`],
  [`=COUNTIFS(E13:E${end},"双方")+COUNTIFS(E13:E${end},"仅课程")`],
  [`=COUNTIFS(E13:E${end},"双方")`],
  [`=COUNTA(A13:A${end})`],
  [`=COUNTIFS(D13:D${end},"首批")`],
  [`=COUNTIFS(D13:D${end},"扩展")`],
  [`=COUNTIFS(D13:D${end},"首批",J13:J${end},"已有")`],
];
sheet.getRange('D4:F10').values = [
  ['比较口径','按论文标识去重；不同修订号视为同一篇论文。',null],
  ['课程版本','2026-10-23 已保存快照，本轮未刷新课程网站。',null],
  ['选择依据',expanded ? '保留原60篇，补入20篇；N列标明追加批次。' : '优先 LLM 相关性、主题覆盖及跨论文比较机会。',null],
  ['填写说明','可编辑 D 列“建议”：首批 / 扩展；汇总会更新。',null],
  ['文件状态',expanded ? '选定80篇，保留原始文件；待正文解析与chunking。' : '保留全部原始文件，筛选仅为建议，语料尚未冻结。',null],
  ['版本核对',stats.pinned_version_overlap_needs_verification ? `${stats.pinned_version_overlap_needs_verification} 篇重合论文的指定版本与本地PDF尚未核对。` : '选定论文均已记录固定版本及PDF哈希。',null],
  ['标识修正','2 处数字型 arXiv ID 尾零丢失，已按原始URL恢复。',null],
];
sheet.getRange('E4:E10').format.wrapText = false;
sheet.getRange('D9:E10').format.font = {name:'Arial',size:10,color:'#8A4B08'};
const headers = ['paper_id','论文标题','主题','建议','来源','理由','原表位置','arXiv 标识','固定版本','本地PDF','论文链接','版本说明','本地路径','选入批次'];
sheet.getRange('A12:N12').values = [headers];
const rows = docs.map(r => [
  r.paper_id, r.title, r.review_topic, selection[r.recommendation], origin[r.source_presence], r.selection_reason,
  r.user_entries.map(e => e.sheet+'!'+e.title_cell).join('; '),r.arxiv_id ? 'arXiv:'+r.arxiv_id : '',r.selected_version || r.user_version_labels.join(', '),
  r.download.status === 'downloaded_pdf' ? '已有' : r.download.status === 'not_downloaded' ? '待下载' : r.download.status === 'downloaded_html' ? '仅HTML' : '书籍待评估',
  r.download.versioned_pdf_url || r.user_entries[0]?.url || r.source_url,r.version_note,r.download.local_path || '',
  r.selection_stage==='added_20' ? '新增20篇' : r.selection_stage==='original_60' ? '原60篇' : r.recommendation==='core' ? '原60篇' : '扩展池',
]);
sheet.getRange(`A13:N${end}`).values = rows;
sheet.tables.add(`A12:N${end}`,true,'MergedCandidates');
header(sheet,'A12:N12');
widths(sheet,[26,79,27,14,14,72,22,18,21,18,65,76,52,20],end);
sheet.getRange(`A13:N${end}`).format.rowHeight = 60;
for (const col of ['B','C','F','G','K','L','M']) sheet.getRange(`${col}13:${col}${end}`).format.wrapText = true;
sheet.getRange(`H13:I${end}`).setNumberFormat('@');
sheet.getRange(`D13:D${end}`).dataValidation = { rule:{type:'list',values:['首批','扩展']} };
sheet.getRange(`D13:D${end}`).format.fill = '#FFF2CC';
sheet.getRange(`J13:J${end}`).conditionalFormats.add('containsText',{
  text:'待下载',format:{fill:'#FFF2CC',font:{color:'#8A4B08'}},
});
sheet.freezePanes.freezeRows(12);
sheet.freezePanes.freezeColumns(2);

const overlapEnd = 6 + catalog.overlaps.length;
base(overlap,`A1:J${overlapEnd}`);
overlap.getRange('A2').values = [['两个来源重复的论文']];
overlap.getRange('A2').format.font = {name:'Arial',size:14,bold:true};
overlap.getRange('A3:B4').values = [
  ['重合篇数',null],['去重口径','同一 arXiv work ID 合并，修订号另存。'],
];
overlap.getRange('B3').formulas = [[`=COUNTA(A7:A${overlapEnd})`]];
overlap.getRange('A6:J6').values = [['arXiv 标识','课程标题','表格标题','原表位置','匹配依据','建议','表格指定版本','版本说明','课程链接','表格链接']];
const duplicates = [...catalog.overlaps].sort((a,b)=>a.arxiv_id.localeCompare(b.arxiv_id));
const duplicateRows = duplicates.map(r => {
  const index = docs.findIndex(d=>d.paper_id===r.paper_id) + 13;
  return ['arXiv:'+r.arxiv_id,r.course_title,r.user_title,r.user_locations,'arXiv ID 相同',null,r.user_version_labels.join(', '),r.version_note,r.course_url,r.user_url];
});
overlap.getRange(`A7:J${overlapEnd}`).values = duplicateRows;
duplicates.forEach((r,i)=>{
  overlap.getRange(`F${i+7}`).formulas = [[`=INDEX('候选筛选'!$D$13:$D$${end},MATCH(A${i+7},'候选筛选'!$H$13:$H$${end},0))`]];
});
overlap.tables.add(`A6:J${overlapEnd}`,true,'DuplicateWorks');
header(overlap,'A6:J6');
widths(overlap,[20,78,78,22,22,14,21,76,65,65],overlapEnd);
overlap.getRange(`A7:J${overlapEnd}`).format.rowHeight = 60;
for (const col of ['B','C','D','H','I','J']) overlap.getRange(`${col}7:${col}${overlapEnd}`).format.wrapText = true;
overlap.getRange(`A7:A${overlapEnd}`).setNumberFormat('@');
overlap.freezePanes.freezeRows(6);
overlap.freezePanes.freezeColumns(1);

wb.recalculate();
const expected = [stats.user_unique_works,stats.course_unique_entries,stats.shared_works,stats.merged_unique_entries,stats.proposed_core,stats.reserve,stats.core_local_pdf_count];
const readSummary = await wb.inspect({kind:'region',sheetId:sheet.name,range:'A4:B10',maxChars:3000,tableMaxRows:7,tableMaxCols:2});
console.log(readSummary.ndjson);
// Verify that editable selections update counts and linked duplicate rows.
const coreIndex = docs.findIndex(d=>d.source_presence==='both' && d.recommendation==='core');
const coreCell = `D${coreIndex+13}`;
const duplicateIndex = duplicates.findIndex(d=>d.paper_id===docs[coreIndex].paper_id)+7;
sheet.getRange(coreCell).values = [['扩展']];
wb.recalculate();
const countAfterChange = sheet.getRange('B8:B9').values;
if (Number(countAfterChange[0][0])!==stats.proposed_core-1 || Number(countAfterChange[1][0])!==stats.reserve+1) throw new Error('Selection counts did not update');
if (overlap.getRange(`F${duplicateIndex}`).values[0][0] !== '扩展') throw new Error('Keyed overlap selection did not update');
sheet.getRange(coreCell).values = [['首批']];
wb.recalculate();
const finalCounts = sheet.getRange('B4:B10').values.map(x=>Number(x[0]));
if (JSON.stringify(finalCounts)!==JSON.stringify(expected)) throw new Error('Summary mismatch: '+JSON.stringify(finalCounts));
const scan = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},maxChars:2000});
console.log(scan.ndjson);
await fs.mkdir(out,{recursive:true});
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(out,outputName));
for (const [target,s,range] of [
  ['selection_preview',sheet,'A1:F16'],
  ['duplicates_preview',overlap,'A1:F11'],
  ['selection_reasons_preview',sheet,'C12:G18'],
]) {
  const image = await wb.render({sheetName:s.name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out,target+'.png'),new Uint8Array(await image.arrayBuffer()));
}
console.log(JSON.stringify({output:path.join(out,outputName),finalCounts,selectionControlVerified:true}));
