# zeta_cluster_toolkit v6

Reusable toolkit for supercooled-water structural-domain analysis under shear.

This version separates three concerns:

1. **Physical domain extraction and tracking** use MDAnalysis-backed triclinic PBC.
2. **2D slice visualization** can use a rectangular `brick` display cell, where the tilted triclinic cell is cut and repasted by PBC for readability.
3. **3D visualization** defaults to the real triclinic/sheared Cartesian cell, so the actual domain geometry remains visible.

## Modules

```text
zeta_cluster_toolkit/
├── geometry.py           # MDAnalysis PBC, triclinic boxes, brick/deaffine display transforms
├── io_helpers.py         # read frame payloads from ZOP and run extraction over frames
├── domain_extraction.py  # direct zeta_cg domains and joint P3D-zeta domains
├── tracking.py           # frame-to-frame domain tracking and observable extraction
├── visualization.py      # slices, 3D plots, and track-boundary evolution plots
└── propensity.py         # iso-configurational P_i^3D helper
```

## Coordinate-system policy

### Physical calculations

The following operations use MDAnalysis triclinic PBC logic:

- pair distance and cutoff neighbor search;
- connected-component construction;
- compact cluster centroid;
- radius of gyration;
- centroid distance in tracking.

These operations should be treated as the physical definition of the domain.

### Visualization coordinates

`coordinate_system='brick'` is for 2D visualization. It maps triclinic Cartesian coordinates to an orthogonal display cell:

```text
r_cartesian -> fractional coordinate in the triclinic cell -> [sx Lx, sy Ly, sz Lz]
```

This is only a display transform. It is not used for connectivity or MIC.

`coordinate_system='cartesian'` keeps the real triclinic/sheared Cartesian coordinates.

`coordinate_system='deaffined'` is used only for track-boundary visualization. It first uses the brick coordinate system and then applies a simple shear correction

```text
x_shear <- x_shear - Δγ * y_gradient
```

where `Δγ` is supplied either by `strain_getter(obs)` or by `shear_rate * (frame - frame0) * dt_per_frame`.

## Basic workflow

```python
from zeta_cluster_toolkit import (
    extract_zeta_domains_over_frames,
    track_clusters_over_time,
    summarize_tracks,
)

frames = range(4000, 4800, 10)

def box_getter(frame):
    u.trajectory[int(frame)]
    return u.dimensions.copy()  # [lx, ly, lz, alpha, beta, gamma]

results_z = extract_zeta_domains_over_frames(
    zop=zop,
    frames=frames,
    box=box_getter,
    zeta_field="zeta_cg",
    r_connect=3.6,
    high_seed_pct=90,
    high_grow_pct=75,
    low_seed_pct=10,
    low_grow_pct=25,
    min_cluster_size=10,
    min_core_size=3,
)

tracks_z = track_clusters_over_time(
    results_z,
    kinds=("slow", "fast"),
    min_overlap=3,
    min_jaccard=0.15,
    min_containment=0.35,
    max_centroid_distance=10.0,
    max_gap_steps=1,
)

track_table = summarize_tracks(tracks_z, dt_per_sample=10 * 0.025)
track_table.sort_values(["kind", "duration_steps", "max_n_members"], ascending=[True, False, False]).head(20)
```

## Direct zeta_cg domain extraction

Direct `zeta_cg` extraction does not perform another convolution or local averaging step.

For high-zeta structural domains:

```text
core: zeta_cg >= percentile(zeta_cg, high_seed_pct)
halo/grow: zeta_cg >= percentile(zeta_cg, high_grow_pct)
```

For low-zeta domains:

```text
core: zeta_cg <= percentile(zeta_cg, low_seed_pct)
halo/grow: zeta_cg <= percentile(zeta_cg, low_grow_pct)
```

A grow component is retained only when it contains enough core particles.

## P3D-zeta joint domain extraction

The old joint method remains available:

```python
from zeta_cluster_toolkit import extract_domains_over_frames

results_joint = extract_domains_over_frames(
    zop=zop,
    frames=frames,
    P3D=P3D,
    box=box_getter,
    zeta_field="zeta_cg",
    local_length=3.5,
    local_cutoff=6.0,
    r_connect=3.6,
    alpha=1.0,
    mode="score",
)
```

This method does local averaging and constructs scores such as

```text
slow_score = Z(local zeta_cg) - alpha * Z(local P3D)
fast_score = Z(local P3D) - alpha * Z(local zeta_cg)
```

Use it for structure-dynamics coupling. For structural-domain lifetime, use direct `zeta_cg` domains.

## 2D brick slice visualization

For a tilted shear box, this is the recommended slice view:

```python
from zeta_cluster_toolkit import plot_zeta_cg_slice, plot_zeta_domain_slice

payload = zop.spatial_zeta_cg_map(4000)
u.trajectory[4000]
box = u.dimensions.copy()

fig, ax = plot_zeta_cg_slice(
    positions=payload["positions"],
    zeta_cg=payload["zeta_cg"],
    box=box,
    axis=2,
    slice_center=25.0,
    slice_width=6.0,
    coordinate_system="brick",
)

fig, ax = plot_zeta_domain_slice(
    results_z[0],
    axis=2,
    slice_center=25.0,
    slice_width=6.0,
    coordinate_system="brick",
)
```

`brick` visualization cuts the tilted side and pastes it back by PBC into a rectangular cell, making domain slices easier to read.

## 3D true-triclinic visualization

For actual domain geometry, use the real sheared Cartesian cell:

```python
from zeta_cluster_toolkit import plot_domains_3d_matplotlib

fig3d, ax3d = plot_domains_3d_matplotlib(
    results_z[0],
    show_background=True,
    max_background=1500,
    show_hulls=False,
    background_color_by="zeta_cg",
    coordinate_system="cartesian",  # default; real triclinic/sheared geometry
    draw_box=True,
)
```

Use `coordinate_system='brick'` only as a diagnostic 3D de-skewed view.

## Boundary evolution of one tracked cluster

Pick a long-lived track:

```python
long_tracks = sorted(
    tracks_z,
    key=lambda tr: (tr["kind"], len(tr["observations"])),
)
track = long_tracks[-1]
```

### 1. Overlay members in brick coordinates

```python
from zeta_cluster_toolkit import plot_track_boundary_evolution

fig, ax = plot_track_boundary_evolution(
    results=results_z,
    track=track,
    axis=2,
    index_set="members",
    coordinate_system="brick",
    align_centroid=True,
    mode="overlay",
    max_observations=8,
    draw_hull=True,
)
```

This shows intrinsic boundary-shape changes after removing the visual skew of the box.

### 2. Check whether the core is stable

```python
fig, ax = plot_track_boundary_evolution(
    results=results_z,
    track=track,
    axis=2,
    index_set="core",
    coordinate_system="brick",
    align_centroid=True,
    mode="overlay",
    max_observations=8,
    draw_hull=True,
)
```

If the member hull fluctuates strongly but the core hull remains continuous, the track may still be structurally meaningful.

### 3. Panel view

```python
fig, axes = plot_track_boundary_evolution(
    results=results_z,
    track=track,
    axis=2,
    index_set="members",
    coordinate_system="brick",
    mode="panel",
    max_observations=8,
)
```

### 4. De-affined boundary evolution

Use this only after deciding what shear-rate units your trajectory uses.

If `gamma_dot` is in inverse ps and frame numbers map to `0.025 ps/frame`:

```python
fig, ax = plot_track_boundary_evolution(
    results=results_z,
    track=track,
    axis=2,
    index_set="members",
    coordinate_system="deaffined",
    align_centroid=True,
    shear_rate=gamma_dot_ps_inv,
    dt_per_frame=0.025,
    shear_axis=0,      # x
    gradient_axis=1,   # y
    max_observations=8,
)
```

Alternatively, provide a strain getter:

```python
def strain_getter(obs):
    return gamma_of_frame[obs["frame"]]

fig, ax = plot_track_boundary_evolution(
    results_z,
    track,
    coordinate_system="deaffined",
    strain_getter=strain_getter,
)
```

This applies

```text
x <- x - (gamma(frame) - gamma(first_frame)) * y
```

inside the cluster-centric brick coordinate system.

## Export boundary points for custom analysis

```python
from zeta_cluster_toolkit import collect_track_boundary_points

rows = collect_track_boundary_points(
    results=results_z,
    track=track,
    index_set="members",
    coordinate_system="brick",
    align_centroid=True,
)

for row in rows:
    frame = row["frame"]
    pts = row["points"]   # shape (n_members, 3)
```

You can use these point clouds for alpha shapes, non-convex boundaries, boundary roughness, or stress-boundary correlation.

## Recommended analysis logic

1. Use direct `zeta_cg` domains for structural lifetime.
2. Use MDAnalysis triclinic PBC for all physical distances and connectivity.
3. Use `brick` only for 2D visualization and boundary diagnostics.
4. Use `cartesian` for 3D plots when you want the actual sheared simulation cell.
5. Use `deaffined` only for boundary evolution after validating the shear-rate/strain units.
