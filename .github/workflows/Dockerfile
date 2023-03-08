FROM dreamer2368/librom_env:v0.1.0

# install yaml-cpp
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/jbeder/yaml-cpp.git
WORKDIR ./yaml-cpp
WORKDIR ./lib
RUN cmake .. -DYAML_BUILD_SHARED_LIBS=on
RUN make
WORKDIR $LIB_DIR/yaml-cpp/include/yaml-cpp
RUN sudo ln -s ./ yaml-cpp

# flags for libROM cmake
ENV YAML_DIR=$LIB_DIR/yaml-cpp

# install libROM for scaleupROM
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/LLNL/libROM.git
WORKDIR ./libROM/build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
RUN sudo make

ENV LIBROM_DIR=$LIB_DIR/libROM

# create and switch to a user
WORKDIR /home/$USERNAME
