# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

file(COPY poisson.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY meshes/test.2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)

file(COPY poisson.component.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY config/2x2_config.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/config)
file(COPY meshes/square.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)
file(COPY meshes/square-circle.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)
file(COPY meshes/square-triangle.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)
file(COPY meshes/square-star.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)
file(COPY meshes/square-square.mesh DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/meshes)

file(COPY poisson.sample.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY poisson.sample.2d.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY poisson.sample.random.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY config/sample.config-01.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/config)
file(COPY config/sample.config-02.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/config)
file(COPY config/sample.config-03.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/config)
file(COPY config/sample.config-04.h5 DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson/config)

file(COPY poisson.train.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)
file(COPY poisson.train.dom.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)

file(COPY poisson.build.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)

file(COPY poisson.full.yml DESTINATION ${CMAKE_BINARY_DIR}/examples/poisson)