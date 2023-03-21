// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "linalg_utils.hpp"

using namespace mfem;
using namespace std;

namespace CAROM
{

void ComputeCtAB(const SparseMatrix& A,
                 const CAROM::Matrix& B,  // Non-Distributed matrix
                 const CAROM::Matrix& C,  // Non-Distributed matrix
                 CAROM::Matrix& CtAB)     // Non-distributed (local) matrix
{
   MFEM_VERIFY(B.distributed() && C.distributed() && !CtAB.distributed(),
               "In ComputeCtAB, B, C and CtAB must not be distributed!\n");

   const int num_rows = B.numRows();
   const int num_cols = B.numColumns();
   const int num_rows_A = A.NumRows();

   MFEM_VERIFY(C.numRows() == num_rows_A, "");

   mfem::Vector Bvec(num_rows);
   mfem::Vector ABvec(num_rows_A);

   CAROM::Matrix AB(num_rows_A, num_cols, true);

   for (int i = 0; i < num_cols; ++i) {
      for (int j = 0; j < num_rows; ++j) {
         Bvec[j] = B(j, i);
      }
      A.Mult(Bvec, ABvec);
      for (int j = 0; j < num_rows_A; ++j) {
         AB(j, i) = ABvec[j];
      }
   }

   C.transposeMult(AB, CtAB);
}

void SetBlock(const CAROM::Matrix& blockMat,
               const int iStart, const int iEnd,
               const int jStart, const int jEnd,
               CAROM::Matrix& globalMat)
{
   MFEM_VERIFY(!blockMat.distributed() && !globalMat.distributed(),
               "In SetBlock, blockMat and globalMat must not be distributed!\n");

   const int block_row = blockMat.numRows();
   const int block_col = blockMat.numColumns();
   const int global_row = globalMat.numRows();
   const int global_col = globalMat.numColumns();
   assert(block_row <= global_row);
   assert(block_col <= global_col);
   assert((iStart >= 0) && (iStart < global_row));
   assert((jStart >= 0) && (jStart < global_col));
   assert((iEnd >= 0) && (iEnd <= global_row));
   assert((jEnd >= 0) && (jEnd <= global_col));

   for (int i = iStart; i < iEnd; i++)
   {
      for (int j = jStart; j < jEnd; j++)
      {
         globalMat(i, j) = blockMat(i - iStart, j - jStart);
      }
   }
}

void SetBlock(const CAROM::Vector& blockVec,
               const int iStart, const int iEnd,
               CAROM::Vector& globalVec)
{
   const int block_row = blockVec.dim();
   const int global_row = globalVec.dim();
   assert(block_row <= global_row);
   assert((iStart >= 0) && (iStart < global_row));
   assert((iEnd >= 0) && (iEnd <= global_row));

   for (int i = iStart; i < iEnd; i++)
   {
      globalVec(i) = blockVec(i - iStart);
   }
}

void CopyMatrix(const CAROM::Matrix &carom_mat,
                DenseMatrix &mfem_mat)
{
   const int num_row = carom_mat.numRows();
   const int num_col = carom_mat.numColumns();

   mfem_mat.SetSize(num_row, num_col);
   // This copy is needed since their indexing orders are different.
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
         mfem_mat(i,j) = carom_mat.item(i,j);
}

// Does not support parallelization. for debugging.
void PrintMatrix(const CAROM::Matrix &mat,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < mat.numRows(); i++)
   {
      for (int j = 0; j < mat.numColumns(); j++)
         fprintf(fp, "%.15E\t", mat.item(i,j));
      fprintf(fp, "\n");
   }

   fclose(fp);
   return;
}

}

namespace mfem
{

void RtAP(DenseMatrix& R,
         const SparseMatrix& A,
         DenseMatrix& P,
         DenseMatrix& RtAP)
{
   assert(R.NumRows() == A.NumRows());
   assert(A.NumCols() == P.NumRows());

   const int num_row = R.NumCols();
   const int num_col = P.NumCols();
   RtAP.SetSize(num_row, num_col);
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
      {
         Vector vec_i, vec_j;
         R.GetColumnReference(i, vec_i);
         P.GetColumnReference(j, vec_j);
         // NOTE: mfem::SparseMatrix.InnerProduct(x, y) computes y^t A x
         RtAP(i, j) = A.InnerProduct(vec_j, vec_i);
      }
}

void PrintMatrix(const SparseMatrix &mat,
                const std::string &filename)
{
   std::ofstream file(filename.c_str());
   mat.PrintMatlab(file);
}

// Does not support parallelization. for debugging.
void PrintMatrix(const DenseMatrix &mat,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < mat.NumRows(); i++)
   {
      for (int j = 0; j < mat.NumCols(); j++)
         fprintf(fp, "%.15E\t", mat(i,j));
      fprintf(fp, "\n");
   }

   fclose(fp);
   return;
}

}