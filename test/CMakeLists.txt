add_executable(dd_mms dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/dd_mms.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY inputs/dd_mms.component.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.unit.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)
file(COPY meshes/dd_mms.h5 DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_param_prob test_param_prob.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_param_prob.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)

add_executable(test_workflow test_workflow.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test.base.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/test.2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_hdf5 test_hdf5_utils.cpp $<TARGET_OBJECTS:scaleupROMObj>)