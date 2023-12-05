// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

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
SparseMatrix* RtAP(DenseMatrix& R,
                   const Operator& A,
                   DenseMatrix& P);

template<typename T>
void PrintMatrix(const T &mat,
                const std::string &filename);

void PrintMatrix(const DenseMatrix &mat,
                 const std::string &filename);

// TODO: parallel version + hdf5 file format for larger basis dimension.
void PrintVector(const Vector &vec,
                 const std::string &filename);

SparseMatrix* ReadSparseMatrixFromHDF(const std::string filename);
void WriteSparseMatrixToHDF(const SparseMatrix* mat, const std::string filename);

// Matrix-Vector multiplication on the specified rows.
// Rows are not necessarily sorted and they can be duplicated.
void MultSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y);
void AddMultTransposeSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y);
void MultTransposeSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y);
// Currently needed only for DenseMatrix A.
// For a SparseMatrix A, we can utilize PartMult within this routine.
void AddSubMatrixRtAP(const DenseMatrix& R, const Array<int> &Rrows,
                      const DenseMatrix& A,
                      const DenseMatrix& P, const Array<int> &Prows,
                      DenseMatrix& RAP);

// DenseTensor is column major and i is the fastest index. 
// y_k = T_{ijk} * x_i * x_j
void TensorContract(const DenseTensor &tensor, const Vector &xi, const Vector &xj, Vector &yk);
// y_k += w * T_{ijk} * x_i * x_j
void TensorAddScaledContract(const DenseTensor &tensor, const double w, const Vector &xi, const Vector &xj, Vector &yk);
// Contracts along the axis (0 or 1) and add the multipled transpose.
// axis 0: M_{kj} += T_{ijk} * x_i
// axis 1: M_{ki} += T_{ijk} * x_j
void TensorAddMultTranspose(const DenseTensor &tensor, const Vector &x, const int axis, DenseMatrix &M);
// Contracts along the axis (0 or 1) and add the multipled transpose.
// axis 0: M_{kj} += w * T_{ijk} * x_i
// axis 1: M_{ki} += w * T_{ijk} * x_j
void TensorAddScaledMultTranspose(const DenseTensor &tensor, const double w, const Vector &x, const int axis, DenseMatrix &M);

}

#endif
