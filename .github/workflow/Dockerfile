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

# create and switch to a user
WORKDIR /home/$USERNAME
