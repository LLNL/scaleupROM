add_executable(rectangle_mesh rectangle_mesh.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(1d_mesh 1d_mesh.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(cubic_mesh cubic_mesh.cpp $<TARGET_OBJECTS:scaleupROMObj>)
add_executable(sol2paraview sol2paraview.cpp $<TARGET_OBJECTS:scaleupROMObj>)

if(GMSH)
    add_executable(gmsh2mfem gmsh2mfem.cpp $<TARGET_OBJECTS:scaleupROMObj>)
endif(GMSH)
