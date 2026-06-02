# Pipeline Scripts

This directory contains standalone workflows that orchestrate several tools and produce figures/results. Run these scripts from the repository root unless a script states otherwise.

## Scripts

- `cage_jump_pipeline.py`: reproduces cage-jump analysis outputs and writes cage-jump figures/results.
- `propensity_field_pipeline.py`: runs the propensity-field cluster workflow.
- `zeta_cluster_pipeline.py`: runs zeta cluster analysis and diagnostic plots.
- `order_paramter.py`: currently empty placeholder kept here so notebook files remain notebook-only.

## Running

```bash
python tools/pipelines/cage_jump_pipeline.py
```

Prefer writing generated images to `figures/<topic>/` and numerical outputs to `results/<topic>/`.

