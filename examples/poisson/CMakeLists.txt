# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

file(COPY poisson.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY meshes/test.2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)