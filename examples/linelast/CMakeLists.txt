# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT
file(COPY linelast.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY linelast.simpleL.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY config/linelast.simpleL.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/config)
file(COPY linelast.lattice.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY config/linelast.lattice.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/config)
file(COPY linelast.force_cantilever.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY linelast.comp_train.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast)
file(COPY config/linelast.comp_train.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/config)

file(COPY meshes/beam-tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/beam-tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/joint2D.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/rod2D_H.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)
file(COPY meshes/rod2D_V.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/linelast/meshes)