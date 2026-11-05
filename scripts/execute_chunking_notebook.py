"""Execute the notebook with a project-local kernel and save its actual outputs."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH'] = str(ROOT / '.cache/jupyter/share/jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.cache/jupyter/runtime')
os.environ['IPYTHONDIR'] = str(ROOT / '.cache/ipython')
for directory in ('JUPYTER_RUNTIME_DIR', 'IPYTHONDIR'):
    Path(os.environ[directory]).mkdir(parents=True, exist_ok=True)
target = ROOT / 'notebooks/section_recursive_chunking.ipynb'
notebook = nbformat.read(target, as_version=4)
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'notebook_cell_{index}', 'exec')
client = NotebookClient(notebook, timeout=300, kernel_name='paper-rag',
                        resources={'metadata': {'path': str(ROOT)}})
client.execute()
assert not [output for cell in notebook.cells if cell.cell_type == 'code'
            for output in cell.get('outputs', []) if output.output_type == 'error']
nbformat.write(notebook, target)
print('Executed notebook successfully:', target)
print('Executed code cells:', sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells))
