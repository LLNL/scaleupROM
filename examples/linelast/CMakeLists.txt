# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
file(COPY linelast.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY meshes/beam-tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/beam-tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
