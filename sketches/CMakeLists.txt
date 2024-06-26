add_executable(poisson poisson.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(poisson_serial poisson_serial.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(poisson_1d poisson_1d.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(multidomain multidomain.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(dode domain_decomposition.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(poisson_dg poisson_dg.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(yaml_example yaml_example.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(librom_qr librom_qr.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(librom_svd librom_svd.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(linalg_interface linalg_interface.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(friend_class friend_class.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(static_member static_member.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(block_algebra block_algebra.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(mfem_sparse_hdf mfem_sparse_hdf.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(generate_interface generate_interface.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(mixed_dg_trace mixed_dg_trace_sketch.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes stokes.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_mms stokes_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_mms2 stokes_mms2.cpp $<TARGET_OBJECTS:scaleupROMObj>)
# add_executable(stokes_mms3 stokes_mms3.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_mms4 stokes_mms4.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_dg stokes_dg.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_dg_mms stokes_dg_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(stokes_dd stokes_dd.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(dg_bdr_norm_lf dg_bdr_norm_lf.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(precond precond.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(ns_mms ns_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(ns_dg_mms ns_dg_mms.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(ns_rom ns_rom.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(usns usns.cpp $<TARGET_OBJECTS:scaleupROMObj>)

file(COPY inputs/gen_interface.yml DESTINATION ${CMAKE_BINARY_DIR}/sketches/inputs)
file(COPY meshes/2x2.mesh DESTINATION ${CMAKE_BINARY_DIR}/sketches/meshes)
file(COPY meshes/1x1.mesh DESTINATION ${CMAKE_BINARY_DIR}/sketches/meshes)
