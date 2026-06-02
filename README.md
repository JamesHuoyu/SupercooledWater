# SupercooledWater

This repository contains analysis notebooks and Python tools for supercooled-water simulations, including hydrogen-bond analysis, zeta order parameters, cage-jump analysis, S4 analysis, stress/viscosity analysis, and orientation dynamics.

## Repository layout

```text
SupercooledWater/
├── config.py                 # Shared project paths for notebooks and scripts
├── notebooks/                # Jupyter notebooks only
├── tools/                    # Reusable analysis modules
│   ├── hb_analysis/          # Hydrogen-bond post-processing workflows
│   ├── pipelines/            # Standalone runnable pipeline scripts
│   ├── water_orient/         # Water-orientation analysis package
│   └── zeta_cluster_toolkit/ # Zeta-cluster helper package
├── figures/                  # Generated plots, PDFs, and HTML visualizations
├── results/                  # Generated CSV/NPY/NPZ analysis outputs
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Notebook conventions

Notebooks should stay as interactive analysis documents. Generated plots and numerical outputs should be written outside `notebooks/`:

- Save images, PDFs, and HTML visualizations under `figures/`.
- Save CSV, NPY, NPZ, and other numerical outputs under `results/`.
- Put reusable Python code in `tools/`; put long runnable workflows in `tools/pipelines/`.

At the top of a notebook, use the shared path helper:

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

Then save outputs with explicit directories:

```python
fig.savefig(figures_dir / "example_plot.png", dpi=300, bbox_inches="tight")
df.to_csv(results_dir / "example_summary.csv", index=False)
```

## Tool usage

Import reusable modules from `tools`:

```python
from tools.custom_hbond_analysis import HydrogenBondAnalysis
from tools.zeta_order_parameter import ZetaOrderParameter
from tools.s4_analysis import S4Analysis
```

Standalone workflow scripts live in `tools/pipelines/`. Run them from the repository root so relative output paths resolve consistently:

```bash
python tools/pipelines/cage_jump_pipeline.py
python tools/pipelines/propensity_field_pipeline.py
python tools/pipelines/zeta_cluster_pipeline.py
```

## Output organization

Previously scattered notebook and root-level outputs have been grouped as follows:

- `figures/notebooks/`: plots and PDFs created by notebooks.
- `figures/cagejump/`: cage-jump figure outputs.
- `figures/s4_output_default/` and `figures/s4_output_1.5A/`: S4 figure/result bundles from root-level runs.
- `results/cagejump/`: cage-jump numerical arrays and summary CSV.
- `results/notebooks/`: notebook-generated CSV summaries.

