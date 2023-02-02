add_executable(main main.cpp $<TARGET_OBJECTS:scaleupflowObj>)
add_executable(poisson poisson.cpp $<TARGET_OBJECTS:scaleupflowObj>)
