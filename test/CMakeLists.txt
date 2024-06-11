# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

add_library(mmsSuiteObj OBJECT
mms_suite.hpp
mms_suite.cpp
$<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_parser test_parser.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test.parser.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)

add_executable(poisson_dd_mms poisson_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(stokes_dd_mms stokes_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(steady_ns_dd_mms steady_ns_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(unsteady_ns_dd_mms unsteady_ns_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(linelast_dd_mms linelast_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(advdiff_dd_mms advdiff_dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
add_executable(dg_integ_mms dg_integ_mms.cpp $<TARGET_OBJECTS:scaleupROMObj> $<TARGET_OBJECTS:mmsSuiteObj>)
file(COPY inputs/dd_mms.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY inputs/dd_mms.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.unit.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/square.tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/dd_mms.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY inputs/dd_mms.comp.3d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.3d.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/dd_mms.3d.tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_param_prob test_param_prob.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_param_prob.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)

add_executable(test_workflow test_workflow.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/stokes.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/steady_ns.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/advdiff.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.2x1.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

file(COPY inputs/test.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/stokes.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/steady_ns.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.1x1.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.global.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

file(COPY inputs/linelast.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY inputs/linelast.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/beam-tri.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_hdf5 test_hdf5_utils.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_topol test_topol.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_topol.2d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.4x4.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test_topol.2d.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

file(COPY inputs/test_topol.3d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.1x1x1.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.2x2x2.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.1x1x1.tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.2x2x2.tet.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test_topol.3d.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_block_smoother test_block_smoother.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(nonlinear_integ_grad nonlinear_integ_grad.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(interfaceinteg_grad interfaceinteg_grad.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_linalg_utils test_linalg_utils.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_rom_nonlinearform test_rom_nonlinearform.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_rom_interfaceform test_rom_interfaceform.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_ns_parallel test_ns_parallel.cpp $<TARGET_OBJECTS:scaleupROMObj>)

function(add_test_dir TEST_DIR)
    # helper function that adds all executables in a given directory as tests for ctest
    get_property(target_names DIRECTORY ${TEST_DIR} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS target_names)
        get_target_property(target_type ${target} TYPE)
        if (${target_type} STREQUAL "EXECUTABLE")
            get_target_property(binary_dir ${target} BINARY_DIR)
            add_test(NAME ${target} COMMAND ${target} WORKING_DIRECTORY "${binary_dir}")
        endif()
    endforeach(target)
endfunction(add_test_dir)

add_test_dir(".")

if(GMSH)
    add_subdirectory(gmsh)
    add_test_dir("gmsh")
endif(GMSH)