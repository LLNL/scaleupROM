# Copyright 2024 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

file(COPY stokes.sampling.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes)
file(COPY array.8.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes)

file(COPY generate_configs.py DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes)
file(COPY setup_stokes.sh DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes)

file(COPY meshes/square.o3.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes/meshes)
file(COPY meshes/square-circle.msh.mfem DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes/meshes)
file(COPY meshes/square-triangle.msh.mfem DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes/meshes)
file(COPY meshes/square-star.msh.mfem DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes/meshes)
file(COPY meshes/square-square.msh.mfem DESTINATION ${CMAKE_BINARY_DIR}/examples/stokes/meshes)
