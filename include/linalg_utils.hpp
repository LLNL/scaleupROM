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

#ifndef LINALG_UTILS_HPP
#define LINALG_UTILS_HPP

#include "mfem.hpp"
#include "librom.h"

using namespace mfem;

namespace CAROM
{

// Serial case interface between CAROM and MFEM
void ComputeCtAB(const Operator& A,
                 const CAROM::Matrix& B,  // Distributed matrix
                 const CAROM::Matrix& C,  // Distributed matrix
                 CAROM::Matrix& CtAB);    // Non-distributed (local) matrix

// Serial case set block of CAROM matrix.
void SetBlock(const CAROM::Matrix& blockMat,
               const int iStart, const int iEnd,
               const int jStart, const int jEnd,
               CAROM::Matrix& globalMat);

// Serial case set block of CAROM vector.
void SetBlock(const CAROM::Vector& blockVec,
               const int iStart, const int iEnd,
               CAROM::Vector& globalVec);

void CopyMatrix(const CAROM::Matrix &carom_mat,
                DenseMatrix &mfem_mat);

void PrintMatrix(const CAROM::Matrix &mat,
                 const std::string &filename);

// TODO: parallel version + hdf5 file format for larger basis dimension.
void PrintVector(const CAROM::Vector &vec,
                 const std::string &filename);

}

namespace mfem
{
// Compute Rt * A * P
void RtAP(DenseMatrix& R,
        const Operator& A,
        DenseMatrix& P,
        DenseMatrix& RAP);

void PrintMatrix(const SparseMatrix &mat,
                const std::string &filename);

void PrintMatrix(const DenseMatrix &mat,
                 const std::string &filename);

// TODO: parallel version + hdf5 file format for larger basis dimension.
void PrintVector(const Vector &vec,
                 const std::string &filename);

SparseMatrix* ReadSparseMatrixFromHDF(const std::string filename);
void WriteSparseMatrixToHDF(const SparseMatrix* mat, const std::string filename);

}

#endif
