// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef LINALG_UTILS_HPP
#define LINALG_UTILS_HPP

#include "mfem.hpp"
#include "librom.h"
#include "etc.hpp"

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

struct MatrixBlocks
{
   int nrows;
   int ncols;
   Array2D<SparseMatrix *> blocks;

   MatrixBlocks() : nrows(0), ncols(0), blocks(0, 0) {}
   MatrixBlocks(const int nrows_, const int ncols_)
      : nrows(nrows_), ncols(ncols_), blocks(nrows_, ncols_)
   { blocks = NULL; }
   MatrixBlocks(const Array2D<SparseMatrix *> &mats)
   { *this = mats; }

   ~MatrixBlocks() { DeletePointers(blocks); }

   void SetSize(const int w, const int h)
   {
      for (int i = 0; i < nrows; i++)
         for (int j = 0; j < ncols; j++)
            if (blocks(i, j)) delete blocks(i, j);

      nrows = w; ncols = h;
      blocks.SetSize(w, h);
      blocks = NULL;
   }

   SparseMatrix*& operator()(const int i, const int j)
   {
      assert((i >= 0) && (i < nrows));
      assert((j >= 0) && (j < ncols));
      return blocks(i, j);
   }

   SparseMatrix const * operator()(const int i, const int j) const
   {
      assert((i >= 0) && (i < nrows));
      assert((j >= 0) && (j < ncols));
      return blocks(i, j);
   }

   // MatrixBlocks has the ownership. no copy assignment.
   MatrixBlocks& operator=(const Array2D<SparseMatrix *> &mats)
   {
printf("operator= const Array2D<SparseMatrix *>.\n");
      nrows = mats.NumRows();
      ncols = mats.NumCols();
      blocks = mats;

      return *this;
   }

   // Copy assignment.
   MatrixBlocks& operator=(const MatrixBlocks &matblock)
   {
printf("operator= const MatrixBlocks.\n");
      if (this == &matblock) return *this;
      DeletePointers(blocks);

      nrows = matblock.nrows;
      ncols = matblock.ncols;
      blocks.SetSize(nrows, ncols);
      blocks = NULL;

      for (int i = 0; i < nrows; i++)
         for (int j = 0; j < ncols; j++)
            if (matblock(i, j)) blocks(i, j) = new SparseMatrix(*matblock(i, j));

      return *this;
   }
};

void AddToBlockMatrix(const Array<int> &ridx, const Array<int> &cidx, const MatrixBlocks &mats, BlockMatrix &bmat);

void modifiedGramSchmidt(DenseMatrix& mat);

/* Orthonormalize mat over mat1 and itself. */
void Orthonormalize(DenseMatrix& mat1, DenseMatrix& mat);

// Compute Rt * A * P
void RtAP(DenseMatrix& R,
        const Operator& A,
        DenseMatrix& P,
        DenseMatrix& RAP);
DenseMatrix* DenseRtAP(DenseMatrix& R,
                       const Operator& A,
                       DenseMatrix& P);
SparseMatrix* SparseRtAP(DenseMatrix& R,
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

class CGOptimizer : public OptimizationSolver
{
protected:
   const double golden_ratio = 0.5 * (1.0 + sqrt(5.0));
   const double Cr = 1.0 - 1. / golden_ratio;
   const double ib = 0.1;
   const double eps = 1.0e-14;

   const int N_mnbrak = 1e4;
   const int N_para = 1e4;

public:
   CGOptimizer() : OptimizationSolver() {}
   virtual ~CGOptimizer() {}

   virtual void SetOperator(const Operator &op) override
   {
      oper = &op;
      height = op.Height();
      width = op.Width();

      resvec.SetSize(height);
   }

   virtual void Mult(const Vector &rhs, Vector &sol) const;

private:
   mutable Vector resvec;
   mutable Vector sol1, g, xi, xi1, h;

   double Objective(const Vector &rhs, const Vector &sol) const;
   void Gradient(const Vector &rhs, const Vector &sol, Vector &grad) const;

   double Brent(const Vector &rhs, const Vector &xi, Vector &sol, double b0 = 1e-1, double lin_tol = 1e-2) const;
};

}

#endif
