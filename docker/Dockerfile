# FROM ghcr.io/llnl/librom/librom_env:arm64
FROM ghcr.io/llnl/librom/librom_env:latest

# install parmetis
WORKDIR $LIB_DIR
WORKDIR parmetis-4.0.3
RUN sudo make install

# install mumps
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/scivision/mumps.git
WORKDIR ./mumps
RUN sudo git checkout v5.6.2.1
RUN sudo sed -i 's/option(CMAKE_TLS_VERIFY "Verify TLS certificates" ON)/option(CMAKE_TLS_VERIFY "Verify TLS certificates" OFF)/g' options.cmake
# RUN sudo sed -i 's/TLS_VERIFY true/TLS_VERIFY false/g' cmake/mumps_src.cmake
RUN sudo sed -i 's/if(parallel IN_LIST METIS_FIND_COMPONENTS)/if("parallel" IN_LIST METIS_FIND_COMPONENTS)/g' cmake/FindMETIS.cmake
RUN sudo sed -i '/NAMES parmetis/a PATHS "$ENV{PARMETIS_DIR}/lib"' cmake/FindMETIS.cmake
RUN sudo sed -i '/NAMES metis/a PATHS "$ENV{METIS_DIR}/lib"' cmake/FindMETIS.cmake
RUN sudo sed -i 's/set(metis_inc parmetis.h)/set(metis_inc metis.h)/g' cmake/FindMETIS.cmake
RUN sudo sed -i '/NAMES ${metis_inc}/a PATHS "$ENV{PARMETIS_DIR}/metis/include"' cmake/FindMETIS.cmake
RUN sudo -E cmake -B build -Dparmetis=YES -Dparallel=YES && sudo -E cmake --build build
WORKDIR $LIB_DIR/MUMPS_5.2.0
RUN sudo ln -s $LIB_DIR/mumps/build/_deps/mumps-src/include ./include && sudo ln -s $LIB_DIR/mumps/build ./lib

# make a link to scalapack for mfem cmake-build with mumps
WORKDIR /usr/lib
RUN arch=$(uname -m) && sudo ln -s /usr/lib/${arch}-linux-gnu/libscalapack-openmpi.so.2.1.0 ./

# re-install mfem with mumps, with cmake
WORKDIR $LIB_DIR/mfem/build
RUN sudo -E cmake .. -DBUILD_SHARED_LIBS=YES -DMFEM_USE_MPI=YES -DMFEM_USE_GSLIB=${MG} -DMFEM_USE_METIS=YES -DMFEM_USE_METIS_5=YES -DMFEM_USE_MUMPS=YES -DMETIS_DIR="$PARMETIS_DIR/metis/include" -DGSLIB_DIR="$LIB_DIR/gslib/build" -DMUMPS_DIR="$LIB_DIR/MUMPS_5.2.0"
RUN sudo -E make -j 16
RUN sudo ln -s . include && sudo ln -s . lib

ENV MFEM_DIR=$LIB_DIR/mfem/build
ENV MFEM_INCLUDES=${MFEM_DIR}

# install yaml-cpp
WORKDIR $LIB_DIR
RUN sudo git clone https://github.com/jbeder/yaml-cpp.git
WORKDIR ./yaml-cpp/lib
RUN cmake .. -DYAML_BUILD_SHARED_LIBS=on && make
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
WORKDIR ./build
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DUSE_MFEM=${USE_MFEM} -DMFEM_USE_GSLIB=${MFEM_USE_GSLIB}
RUN sudo make -j 16

ENV LIBROM_DIR=$LIB_DIR/libROM

# install python
RUN sudo apt-get update
RUN sudo apt-get install -yq python3 python3-pip

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

# install gmsh
RUN sudo apt-get install -yq gmsh

# create and switch to a user
WORKDIR /home/$USERNAME

ENV MUMPS_DIR=$LIB_DIR/MUMPS_5.2.0