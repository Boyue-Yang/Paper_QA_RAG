"""Regression checks for source coverage, Unicode, section isolation and locators."""
import tempfile
import unittest
from pathlib import Path

from scripts.chunk_papers import (ChunkingConfig, chunk_document, detect_heading,
    get_tokenizer, merge_short_sections, order_page_units, parse_html_sections, recursive_chunk_spans,
    token_count, validate_document_chunks)


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = get_tokenizer('BAAI/bge-m3')

    def test_unicode_and_long_unbroken_text_have_no_loss(self):
        text = ('中文研究🙂 mathematical symbols α² ≤ β. ' * 20 + '\n\n' + 'abcdefghijk' * 100)
        spans = recursive_chunk_spans(text, self.encoding, 64, 12)
        end = 0
        for start, finish in spans:
            self.assertFalse(text[end:start].strip())
            self.assertLessEqual(token_count(self.encoding, text[start:finish]), 64)
            self.assertGreater(finish, end)
            self.assertNotIn('\ufffd', text[start:finish])
            end = finish
        self.assertFalse(text[end:].strip())
        self.assertGreater(len(spans), 2)

    def test_two_columns_are_not_interleaved(self):
        units = []
        for side, x in [('left', 50), ('right', 320)]:
            for i in range(8):
                bbox = [x, 100+i*20, x+200, 112+i*20]
                line = {'text': side+' paragraph sentence with enough body text.', 'bbox': bbox,
                        'font_size': 10, 'bold': False}
                units.append({'lines': [line], 'bbox': bbox, 'page_width': 600, 'page_height': 800, 'page': 1})
        result, layout = order_page_units(units, 10)
        self.assertEqual(layout, 'two_columns_heuristic')
        self.assertTrue(all(u['lines'][0]['text'].startswith('left') for u in result[:8]))
        self.assertTrue(all(u['lines'][0]['text'].startswith('right') for u in result[8:]))

    def test_normal_numbered_body_line_is_not_a_heading(self):
        body = {'text': '1 We first evaluate on the validation dataset.', 'font_size': 10, 'bold': False}
        self.assertIsNone(detect_heading(body, 10, {}, 2))
        heading = {'text': '3.2 Training Objective', 'font_size': 11, 'bold': True}
        self.assertEqual(detect_heading(heading, 10, {}, 2)['level'], 2)
        for text in ['1 return x = y', '8 end', '21 days, 96 k20s 12 days, 64 k40s', 'Results:']:
            self.assertIsNone(detect_heading({'text':text,'font_size':10,'bold':True},10,{},2))
        label = {'text':'Scaled Dot-Product Attention','font_size':10,'bold':False}
        outline = {(4,'scaleddotproductattention'):{'level':3}}
        self.assertIsNone(detect_heading(label,10,outline,4))

    def test_references_exclusion_appendix_and_source_mapping(self):
        sections = []
        for i, (title, kind) in enumerate([('1 Method','body'),('References','references'),('A Appendix','body')]):
            text = ('Evidence from '+title+'. ') * 30
            mid = len(text)//2
            sections.append({'section_id': str(i), 'title': title, 'path': [title], 'kind': kind,
                'text': text, 'paragraphs': [
                    {'paragraph_id': str(i)+'a','page':1,'bbox':[0,0,1,1],'text':text[:mid],'char_start':0,'char_end':mid},
                    {'paragraph_id': str(i)+'b','page':2,'bbox':[0,0,1,1],'text':text[mid:],'char_start':mid,'char_end':len(text)}]})
        parsed = {'parser_version':'test','pdf_sha256':'abc','quality_flags':[],'sections':sections}
        paper = {'paper_id':'example','arxiv_id':'0000.00000','title':'Example','selected_version':'v1',
                 'download':{'versioned_pdf_url':'https://example.invalid/paper','local_path':'example.pdf'}}
        config = ChunkingConfig(max_tokens=64,overlap_tokens=12)
        chunks = chunk_document(parsed,paper,config,self.encoding)
        self.assertNotIn('References',{c['section_title'] for c in chunks})
        self.assertIn('A Appendix',{c['section_title'] for c in chunks})
        self.assertTrue(any(len(c['page_numbers'])==2 for c in chunks))
        validate_document_chunks(parsed,chunks,config,self.encoding)
        again = chunk_document(parsed,paper,config,self.encoding)
        self.assertEqual([c['chunk_id'] for c in chunks],[c['chunk_id'] for c in again])

    HTML = (
        '<html><body><article class="ltx_document">'
        '<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
        '<p class="ltx_p" id="abs.p1">We study <math alttext="x^{2}"><semantics><mi>x</mi></semantics></math> carefully.</p></div>'
        '<section class="ltx_section" id="S1"><h2 class="ltx_title ltx_title_section"><span class="ltx_tag">1 </span>Intro</h2>'
        '<div class="ltx_para" id="S1.p1"><p class="ltx_p" id="S1.p1.1">Short intro.</p></div>'
        '<section class="ltx_subsection" id="S1.SS1"><h3 class="ltx_title ltx_title_subsection"><span class="ltx_tag">1.1 </span>Setup</h3>'
        '<div class="ltx_para" id="S1.SS1.p1"><p class="ltx_p" id="S1.SS1.p1.1">Setup text. <span class="ltx_ERROR">\\bad</span></p></div>'
        '<section class="ltx_paragraph" id="S1.SS1.Px1"><h5 class="ltx_title ltx_title_paragraph">Run in</h5>'
        '<div class="ltx_para" id="S1.SS1.Px1.p1"><p class="ltx_p">Run-in body.</p></div></section></section>'
        '<figure class="ltx_table" id="T1"><figcaption>Table 1: Scores.</figcaption>'
        '<span class="ltx_tabular"><span class="ltx_tr"><span class="ltx_td">Model</span><span class="ltx_td">BLEU</span></span>'
        '<span class="ltx_tr"><span class="ltx_td">A</span><span class="ltx_td">27.3</span></span></span></figure>'
        '</section>'
        '<section class="ltx_part" id="Pt1"><section class="ltx_appendix" id="A1">'
        '<h2 class="ltx_title ltx_title_appendix">Appendix A Extra</h2>'
        '<div class="ltx_para"><p class="ltx_p">Appendix text.</p></div></section></section>'
        '<section class="ltx_bibliography" id="bib"><h2 class="ltx_title ltx_title_bibliography">References</h2>'
        '<ul class="ltx_biblist"><li class="ltx_bibitem" id="bib.bib1">K. Cho and Y. Bengio. A paper. 2014.</li></ul></section>'
        '</article></body></html>')

    def parse_html(self):
        paper = {'paper_id': 'ex', 'arxiv_id': '0000.00000', 'selected_version': 'v1', 'title': 'Ex',
                 'html': {'url': 'https://example.invalid/html', 'local_path': 'ex.html'}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ex.html'
            path.write_text(self.HTML, encoding='utf-8')
            return parse_html_sections(path, paper, 'hash'), paper

    def test_html_structure_math_tables_references_and_parts(self):
        parsed, _ = self.parse_html()
        by_title = {s['title']: s for s in parsed['sections']}
        self.assertEqual(by_title['References']['kind'], 'references')
        self.assertIn('Appendix A Extra', by_title)  # section inside \part is kept
        self.assertIn('$x^{2}$', by_title['Abstract']['text'])
        self.assertNotIn('bad', by_title['1.1 Setup']['text'])  # conversion-error element dropped
        self.assertEqual(parsed['latex_error_elements'], 1)
        self.assertIn('Run in', by_title['1.1 Setup']['text'])  # run-in heading stays in its section
        self.assertIn('Model | BLEU\nA | 27.3', by_title['1 Intro']['text'])  # span-based table
        self.assertEqual(by_title['1.1 Setup']['path'], ['1 Intro', '1.1 Setup'])

    def test_short_sections_merge_without_losing_headings(self):
        parsed, paper = self.parse_html()
        config = ChunkingConfig(max_tokens=64, overlap_tokens=8, min_section_tokens=100)
        merged = merge_short_sections(parsed, self.encoding, config.min_section_tokens)
        titles = [s['title'] for s in merged['sections'] if s['kind'] == 'body']
        self.assertLess(len(titles), len([s for s in parsed['sections'] if s['kind'] == 'body']))
        text = ' '.join(s['text'] for s in merged['sections'] if s['kind'] == 'body')
        for heading in ['1.1 Setup', 'Short intro.']:
            self.assertIn(heading, text)
        self.assertNotIn('Appendix text.', ' '.join(s['text'] for s in merged['sections'] if s['title'] == 'Abstract'))
        chunks = chunk_document(merged, paper, config, self.encoding)
        validate_document_chunks(merged, chunks, config, self.encoding)
        self.assertTrue(all(c['source_format'] == 'html' and c['page_numbers'] == [] for c in chunks))
        self.assertTrue(all(s['element_id'] for c in chunks for s in c['source_spans']))
        self.assertNotIn('References', {c['section_title'] for c in chunks})

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            ChunkingConfig(max_tokens=64,overlap_tokens=64)


if __name__ == '__main__':
    unittest.main()
