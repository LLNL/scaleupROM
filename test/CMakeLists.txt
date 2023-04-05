add_executable(dd_mms dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/dd_mms.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY inputs/dd_mms.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.unit.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/dd_mms.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY inputs/dd_mms.comp.3d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.3d.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_param_prob test_param_prob.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_param_prob.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)

add_executable(test_workflow test_workflow.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_hdf5 test_hdf5_utils.cpp $<TARGET_OBJECTS:scaleupROMObj>)

add_executable(test_topol test_topol.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_topol.2d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.4x4.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test_topol.2d.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

file(COPY inputs/test_topol.3d.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.1x1x1.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test.2x2x2.hex.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/test_topol.3d.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)