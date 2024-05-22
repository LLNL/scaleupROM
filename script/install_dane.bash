#!/usr/bin/bash
check_result () {
  # $1: Result output of the previous command ($?)
  # $2: Name of the previous command
  if [ $1 -eq 0 ]; then
      echo "$2 succeeded"
  else
      echo "$2 failed"
      exit -1
  fi
}

#module load gcc/11.2.1
#module load git/2.36.1
#module load mvapich2
#module load cmake/3.26.3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
INSTALL_HELPER=$SCRIPT_DIR/../install-helper
LIB_DIR=$SCRIPT_DIR/../dependencies
mkdir -p $LIB_DIR

# get libROM first
cd $LIB_DIR
git clone https://github.com/LLNL/libROM.git

# scalapack
if [ -f "$LIB_DIR/scalapack-install/lib/libscalapack.a" ]; then
    echo "Using $LIB_DIR/scalapack-install/lib/libscalapack.a"
else
    tar -zxvf ./libROM/dependencies/scalapack-2.2.0.tar.gz
    cd scalapack-2.2.0
    cp $INSTALL_HELPER/scalapack/CMakeLists.txt ./
    mkdir build && cd build
    cmake .. -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_INSTALL_PREFIX=$LIB_DIR/scalapack-install
    check_result $? ScaLAPACK-config
    
    make -j 16
    make install
    check_result $? ScaLAPACK-installation
fi

# hypre
cd $LIB_DIR
if [ ! -d "hypre" ]; then
  wget https://github.com/hypre-space/hypre/archive/v2.28.0.tar.gz
  tar -zxvf v2.28.0.tar.gz
  mv hypre-2.28.0 hypre
  cd hypre/src
  ./configure --disable-fortran
  make -j
  check_result $? hypre-installation
fi
