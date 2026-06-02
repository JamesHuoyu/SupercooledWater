# Notebooks

This directory is reserved for Jupyter notebooks. Keep generated plots, PDFs, HTML files, CSV files, and NumPy arrays outside this directory so the notebook list stays easy to scan.

## Recommended setup cell

Use this cell near the top of each notebook before importing project tools:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import paths

figures_dir = paths.figures_dir / "notebooks"
results_dir = paths.results_dir / "notebooks"
figures_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
```

## Saving outputs

```python
fig.savefig(figures_dir / "my_figure.png", dpi=300, bbox_inches="tight")
summary.to_csv(results_dir / "my_summary.csv", index=False)
```

Use topic-specific subdirectories when a notebook produces many outputs, for example:

```python
figures_dir = paths.figures_dir / "zeta"
results_dir = paths.results_dir / "zeta"
```

## Using tools

Reusable analysis code should be imported from `tools`, not copied into notebooks:

```python
from tools.custom_hbond_analysis import HydrogenBondAnalysis
from tools.zeta_order_parameter import ZetaOrderParameter
from tools.shear_isf import ISFCalculator
```

Long scripts or end-to-end workflows belong in `tools/pipelines/` rather than in this directory.

