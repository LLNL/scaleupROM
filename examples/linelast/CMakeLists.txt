# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
file(COPY test_linelast.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY meshes/beam-tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/beam-tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)

# Temporary for testing
file(COPY reference_fom/displaced_mfem_tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/reference_fom)
file(COPY reference_fom/displaced_mfem_tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/reference_fom)
file(COPY reference_fom/sol_mfem_tri.gf DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/reference_fom)
file(COPY reference_fom/sol_mfem_tet.gf DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/reference_fom)