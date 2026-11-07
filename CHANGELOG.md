# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.1.0] - 2026-11-07

### Added

- Tangent-space Krylov (TaSK) spectral functions of finite MPS: `TaSK`, `task_spectrum`, `task_lanczos`,
  `dynamical_structure_factor`, `dynamical_correlator`, `evaluate_correlator`.
- Projection of N-site, charged (e.g. `spin_vector_operator`) and MPO operators onto the quasiparticle
  tangent space at a site or at momentum `2πn/L`: `project_tangent`.
- Gaussian, Lorentzian and continued-fraction broadening, spectral moments, `correlator_spectrum`.
- TDVP references within the tangent space (`tdvp_correlator`, `tdvp_structure_factor`), Krylov time
  evolution (`task_time_evolve`) and the 2-site tangent-space variance (`tangent_variance_2site`).
- Plots.jl extension `plot_spectral_function`.
