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
#include "utils/HDFDatabase.h"

using namespace mfem;
using namespace std;

namespace CAROM
{

void ComputeCtAB(const Operator& A,
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

void PrintVector(const CAROM::Vector &vec,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < vec.dim(); i++)
      fprintf(fp, "%.15E\n", vec.item(i));

   fclose(fp);
   return;
}

}

namespace mfem
{

void RtAP(DenseMatrix& R,
         const Operator& A,
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
         Vector tmp(vec_i.Size());
         tmp = 0.0;
         A.Mult(vec_j, tmp);
         RtAP(i, j) = vec_i * tmp;
      }
}

SparseMatrix* RtAP(DenseMatrix& R,
   const Operator& A, DenseMatrix& P)
{
   assert(R.NumRows() == A.NumRows());
   assert(A.NumCols() == P.NumRows());

   const int num_row = R.NumCols();
   const int num_col = P.NumCols();
   SparseMatrix *RAP = new SparseMatrix(num_row, num_col);
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
      {
         Vector vec_i, vec_j;
         R.GetColumnReference(i, vec_i);
         P.GetColumnReference(j, vec_j);
         // NOTE: mfem::SparseMatrix.InnerProduct(x, y) computes y^t A x
         Vector tmp(vec_i.Size());
         tmp = 0.0;
         A.Mult(vec_j, tmp);
         RAP->Set(i, j, vec_i * tmp);
      }
   RAP->Finalize();

   return RAP;
}

template<typename T>
void PrintMatrix(const T &mat,
                const std::string &filename)
{
   std::ofstream file(filename.c_str());
   mat.PrintMatlab(file);
}

template void PrintMatrix<SparseMatrix>(const SparseMatrix &mat, const std::string &filename);
template void PrintMatrix<BlockMatrix>(const BlockMatrix &mat, const std::string &filename);

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

void PrintVector(const Vector &vec,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < vec.Size(); i++)
      fprintf(fp, "%.15E\n", vec(i));

   fclose(fp);
   return;
}

// TODO: support parallel I/O.
SparseMatrix* ReadSparseMatrixFromHDF(const std::string filename)
{
   bool success = false;
   CAROM::HDFDatabase hdfIO;

   success = hdfIO.open(filename, "r");
   assert(success);

   int size[2];
   hdfIO.getIntegerArray("size", &size[0], 2);

   const int n_entry = hdfIO.getDoubleArraySize("data");
   Array<int> i_read(n_entry), j_read(n_entry);
   Vector data_read(n_entry);
   hdfIO.getIntegerArray("I", i_read.Write(), size[0]+1);
   hdfIO.getIntegerArray("J", j_read.Write(), i_read[size[0]]);
   hdfIO.getDoubleArray("data", data_read.Write(), i_read[size[0]]);

   // Need to transfer the ownership to avoid segfault or double-free.
   int *ip, *jp;
   i_read.StealData(&ip);
   j_read.StealData(&jp);

   SparseMatrix *mat = new SparseMatrix(ip, jp, data_read.StealData(), size[0], size[1]);

   success = hdfIO.close();
   assert(success);

   return mat;
}

void WriteSparseMatrixToHDF(const SparseMatrix* mat, const std::string filename)
{
   assert(mat->Finalized());
   
   bool success = false;
   CAROM::HDFDatabase hdfIO;

   success = hdfIO.create(filename);
   assert(success);

   const int nnz = mat->NumNonZeroElems();
   const int height = mat->Height();
   const int *i_idx = mat->ReadI();
   const int *j_idx = mat->ReadJ();
   const double *data = mat->ReadData();

   hdfIO.putIntegerArray("I", i_idx, height+1);
   hdfIO.putIntegerArray("J", j_idx, i_idx[height]);
   hdfIO.putDoubleArray("data", data, i_idx[height]);

   int size[2];
   size[0] = mat->NumRows();
   size[1] = mat->NumCols();
   hdfIO.putIntegerArray("size", &size[0], 2);

   success = hdfIO.close();
   assert(success);
}

void MultSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y)
{
   const int nrow = rows.Size(), ncol = mat.NumCols(), height = mat.NumRows();
   assert(x.Size() == ncol);
   y.SetSize(nrow);

   const int rmin = rows.Min(), rmax = rows.Max();
   assert((rmin >= 0) && (rmin < mat.NumRows()));
   assert((rmax >= 0) && (rmax < mat.NumRows()));

   const double *d_mat = mat.Read();
   const double *d_x = x.Read();
   const int *d_rows = rows.Read();
   double *d_y = y.GetData();

   double xc = d_x[0];
   for (int r = 0; r < nrow; r++)
      d_y[r] = d_mat[d_rows[r]] * xc;
   d_mat += height;

   for (int c = 1; c < ncol; c++)
   {
      xc = d_x[c];
      for (int r = 0; r < nrow; r++)
         d_y[r] += d_mat[d_rows[r]] * xc;
      d_mat += height;
   }
}

void AddMultTransposeSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y)
{
   const int nrow = rows.Size(), ncol = mat.NumCols(), height = mat.NumRows();
   assert(x.Size() == nrow);
   assert(y.Size() == ncol);

   const int rmin = rows.Min(), rmax = rows.Max();
   assert((rmin >= 0) && (rmin < mat.NumRows()));
   assert((rmax >= 0) && (rmax < mat.NumRows()));

   const double *d_mat = mat.Read();
   const double *d_x = x.Read();
   const int *d_rows = rows.Read();
   double *d_y = y.GetData();

   double yc;
   for (int c = 0; c < ncol; c++)
   {
      yc = 0.0;
      for (int r = 0; r < nrow; r++)
         yc += d_mat[d_rows[r]] * d_x[r];
      d_y[c] += yc;
      d_mat += height;
   }
}

}