set(module_list main)
set(source_files)
foreach(module IN LISTS module_list)
  list(APPEND source_files ${module}.cpp)
endforeach(module) # IN LISTS module_list
