<!-- ![libROM Logo](https://www.librom.net/img/logo-300.png) -->

# Introduction

scaleupROM is a projection-based reduced order model (ROM) with discontinuous Galerkin domain decomposition (DG-DD).
It aims to construct a robust, efficient, large-scale ROM that is trained only from small scale component samples,
for various physics partial differential equations.
scaleupROM is mainly built upon [MFEM](https://mfem.org/) and [libROM](https://www.librom.net).

## Features

- Discontinuous Galerkin domain decomposition
- Projection-based reduced order model
- EQP for nonlinear partial differential equations
- Supporting physics equations:
  - Poisson equation
  - Stokes flow

## Features to be added

- Steady Navier-Stokes flow

# Installation

**TODO**: set up installation instruction

# Using Docker container

Docker container [`scaleuprom_env`](https://ghcr.io/llnl/scaleuprom/scaleuprom_env) provides a containerized environment with all the prerequisites for scaleupROM:
- intel chips: [https://ghcr.io/llnl/scaleuprom/scaleuprom_env:latest](https://ghcr.io/llnl/scaleuprom/scaleuprom_env:latest)
- apple chips: [https://ghcr.io/llnl/scaleuprom/scaleuprom_env:arm64](https://ghcr.io/llnl/scaleuprom/scaleuprom_env:arm64)

# License

scaleupROM is distributed under the MIT license. For more details, see the [LICENSE](https://github.com/LLNL/scaleupROM/blob/master/LICENSE) File.

SPDX-License-Identifier: MIT

LLNL-CODE-857975

# Authors
- "Kevin" Seung Whan Chung (LLNL)
- Youngsoo Choi (LLNL)
- Pratanu Roy (LLNL)
- Thomas Moore (QUT)
- Thomas Roy (LLNL)
- Tiras Y. Lin (LLNL)
- Sarah E. Baker (LLNL)