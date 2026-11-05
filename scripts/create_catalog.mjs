import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'outputs', 'paper_collection_20261023');
const catalog = JSON.parse(await fs.readFile(path.join(root, 'data', 'paper_catalog.json'), 'utf8'));
await fs.mkdir(out, { recursive: true });
const wb = Workbook.create();
const sheet = wb.worksheets.add('Candidates');
sheet.showGridLines = false;
sheet.tabColor = '#254E70';
const headers = ['paper_id', 'title', 'year_as_listed', 'scope_bucket', 'document_type', 'reading_group', 'selection', 'download_status', 'pages', 'identity_review', 'local_path', 'paper_url', 'lecture_topic', 'source_page'];
const rows = catalog.documents.map(r => [
  r.paper_id, r.title, r.year_as_listed, r.sources[0].scope_bucket,
  r.document_type, [...new Set(r.sources.map(s => s.reading_group))].join(', '),
  r.selection_status, r.download.status, r.download.page_count ?? null,
  r.download.identity_review ?? '', r.download.local_path ?? '', r.source_url,
  [...new Set(r.sources.map(s => s.lecture_topic))].join('; '),
  [...new Set(r.sources.map(s => s.source_url))].join('; '),
]);
const start = 9, last = start + rows.length;
sheet.getRange(`A1:N${last}`).format.font = { name: 'Arial', size: 10, color: '#202A35' };
sheet.getRange(`A1:N${last}`).format.verticalAlignment = 'center';
sheet.getRange('A2').values = [['Paper QA corpus candidates']];
sheet.getRange('A2').format.font = { name: 'Arial', size: 14, bold: true };
sheet.getRange('A3:B7').values = [
  ['Course candidates', null],
  ['Downloaded PDFs', null],
  ['Downloaded web articles', null],
  ['User spreadsheet', 'Pending: Google account verification required. Not imported.'],
  ['Editable field', 'selection: candidate / include / exclude / hold. Scope buckets are provisional.'],
];
sheet.getRange('B3:B5').formulas = [
  [`=COUNTA(A${start + 1}:A${last})`],
  [`=COUNTIFS(H${start + 1}:H${last},"downloaded_pdf")`],
  [`=COUNTIFS(H${start + 1}:H${last},"downloaded_html")`],
];
sheet.getRange('B6:B7').format.font = { name: 'Arial', size: 10, color: '#8A4B08' };
sheet.getRange(`A${start}:N${start}`).values = [headers];
sheet.getRange(`A${start + 1}:N${last}`).values = rows;
sheet.getRange(`A${start}:N${start}`).format = {
  fill: '#254E70', font: { name: 'Arial', size: 10, color: '#FFFFFF', bold: true },
  horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 30,
};
sheet.tables.add(`A${start}:N${last}`, true, 'CorpusCandidates');
const widths = [27, 79, 15, 32, 22, 18, 16, 25, 9, 24, 50, 72, 70, 65];
for (let c = 0; c < widths.length; c++) {
  const letter = String.fromCharCode(65 + c);
  sheet.getRange(`${letter}1:${letter}${last}`).format.columnWidth = widths[c];
}
sheet.getRange(`A${start + 1}:N${last}`).format.rowHeight = 48;
sheet.getRange(`B${start + 1}:B${last}`).format.wrapText = true;
sheet.getRange(`K${start + 1}:L${last}`).format.wrapText = true;
sheet.getRange(`M${start + 1}:N${last}`).format.wrapText = true;
sheet.getRange(`C${start + 1}:C${last}`).setNumberFormat('0');
sheet.getRange(`I${start + 1}:I${last}`).setNumberFormat('#,##0');
sheet.getRange(`C${start + 1}:C${last}`).format.horizontalAlignment = 'right';
sheet.getRange(`I${start + 1}:I${last}`).format.horizontalAlignment = 'right';
sheet.getRange(`G${start + 1}:G${last}`).dataValidation = {
  rule: { type: 'list', values: ['candidate', 'include', 'exclude', 'hold'] },
};
sheet.getRange(`G${start + 1}:G${last}`).format.fill = '#FFF2CC';
sheet.getRange(`H${start + 1}:H${last}`).conditionalFormats.add('containsText', {
  text: 'failed', format: { fill: '#FCE4D6', font: { color: '#9C0006' } },
});
sheet.getRange(`J${start + 1}:J${last}`).conditionalFormats.add('containsText', {
  text: 'check_title', format: { fill: '#FFF2CC', font: { color: '#8A4B08' } },
});
sheet.getRange(`J${start + 1}:J${last}`).conditionalFormats.add('containsText', {
  text: 'needs_ocr', format: { fill: '#FFF2CC', font: { color: '#8A4B08' } },
});
sheet.freezePanes.freezeRows(start);
sheet.freezePanes.freezeColumns(2);
wb.recalculate();
const inspected = await wb.inspect({ kind: 'region', sheetId: sheet.name, range: 'A2:C7', maxChars: 2500, tableMaxRows: 6, tableMaxCols: 3 });
console.log(inspected.ndjson);
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!', options: { useRegex: true, maxResults: 20 }, maxChars: 2000 });
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(out, 'paper_catalog_course_pending_merge.xlsx'));
for (const [name, range] of [['catalog_preview', 'A1:F15'], ['download_preview', 'G9:L16']]) {
  const image = await wb.render({ sheetName: sheet.name, range, scale: 1, format: 'png' });
  await fs.writeFile(path.join(out, name + '.png'), new Uint8Array(await image.arrayBuffer()));
}
console.log(JSON.stringify({ output: path.join(out, 'paper_catalog_course_pending_merge.xlsx'), rows: rows.length }));
