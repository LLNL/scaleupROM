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
WORKDIR ./libROM
# some minor revisions needed for scaleupROM, not merged yet into master branch.
RUN sudo git pull
RUN sudo git checkout error_msg
WORKDIR ./build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
RUN sudo make

ENV LIBROM_DIR=$LIB_DIR/libROM

# install python
RUN sudo apt-get update
RUN sudo apt-get install -yq python3
RUN sudo apt-get install -yq python3-pip

RUN echo "numpy" >> requirements.txt
RUN echo "scipy" >> requirements.txt
RUN echo "argparse" >> requirements.txt
RUN echo "tables" >> requirements.txt
RUN echo "PyYAML" >> requirements.txt
RUN echo "h5py" >> requirements.txt
RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install -r ./requirements.txt

# install h5dump
RUN sudo apt-get install -yq hdf5-tools

# create and switch to a user
WORKDIR /home/$USERNAME
