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

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

#endif
