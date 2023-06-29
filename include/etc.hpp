// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef ETC_HPP
#define ETC_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double UniformRandom();

template <typename T>
inline void DeletePointers(Array<T*> &ptr_array)
{ for (int k = 0; k < ptr_array.Size(); k++) delete ptr_array[k]; }

template <typename T>
inline void DeletePointers(Array2D<T*> &ptr_array)
{
    for (int i = 0; i < ptr_array.NumRows(); i++)
        for (int j = 0; j < ptr_array.NumCols(); j++)
            delete ptr_array(i,j);
}

bool FileExists(const std::string& name);

#endif
