add_executable(dd_mms dd_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/dd_mms.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)
file(COPY meshes/dd_mms.mesh DESTINATION ${CMAKE_BINARY_DIR}/test/meshes)

add_executable(test_param_prob test_param_prob.cpp $<TARGET_OBJECTS:scaleupROMObj>)
file(COPY inputs/test_param_prob.yml DESTINATION ${CMAKE_BINARY_DIR}/test/inputs)