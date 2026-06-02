"""
Reusable tools for zeta_cg spatial slices, direct zeta_cg domains,
joint propensity-structure domains, and cluster continuity analysis.

Typical workflow:
    1. Build one frame payload with io_helpers.frame_payload_from_zop(...)
    2. Extract direct zeta_cg domains with extract_zeta_structure_domains(...)
       or joint P3D-zeta domains with extract_joint_propensity_structure_domains(...)
    3. Visualize with plot_zeta_cg_slice(...) or plot_domain_slice(...)
    4. Build framewise results and track clusters with track_clusters_over_time(...)
"""

from .geometry import (
    box_to_mda_dimensions,
    wrap_positions_orthorhombic,
    wrap_positions_pbc,
    cell_matrix_from_box,
    lammps_triclinic_matrix,
    lammps_triclinic_dimensions,
    cell_lengths,
    minimum_image,
    mic_distance,
    unwrap_positions_relative,
    pbc_centroid,
    pbc_radius_of_gyration,
    query_pairs_pbc,
    query_ball_point_pbc,
    positions_to_unsheared_orthogonal,
)

from .io_helpers import (
    frame_payload_from_zop,
    extract_domains_for_frame,
    extract_domains_over_frames,
    extract_zeta_domains_for_frame,
    extract_zeta_domains_over_frames,
)

from .domain_extraction import (
    transform_propensity,
    robust_zscore,
    local_kernel_average,
    component_labels_from_seed_growth,
    extract_zeta_structure_domains,
    extract_joint_propensity_structure_domains,
    print_domain_summary,
)

from .visualization import (
    plot_zeta_cg_slice,
    plot_domain_slice,
    plot_zeta_domain_slice,
    plot_score_distributions,
    plot_domains_3d_plotly,
    plot_single_cluster_3d_plotly,
    plot_domains_3d_matplotlib,
    plot_track_boundary_evolution,
)

from .tracking import (
    flatten_frame_clusters,
    track_clusters_over_time,
    summarize_tracks,
    track_observable,
)
