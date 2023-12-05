// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef ETC_HPP
#define ETC_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;

double UniformRandom();
int UniformRandom(const int &min, const int &max);

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
