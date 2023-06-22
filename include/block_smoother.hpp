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

#ifndef BLOCK_SMOOTHER_HPP
#define BLOCK_SMOOTHER_HPP

#include "mfem.hpp"

using namespace mfem;

namespace mfem
{

class BlockSmoother : public MatrixInverse
{
protected:
   const BlockMatrix *oper;
   Array<DenseMatrix *> diag_inv;

   int num_row_blocks = -1;
   int num_col_blocks = -1;
   Array<int> row_offsets, col_offsets;

public:
   BlockSmoother() { oper = NULL; }

   BlockSmoother(const BlockMatrix &a);

   virtual ~BlockSmoother();

   virtual void SetOperator(const Operator &a);
};

/// Data type for Gauss-Seidel smoother of sparse matrix
class BlockGSSmoother : public BlockSmoother
{
protected:
   int type; // 0, 1, 2 - symmetric, forward, backward
   int iterations;

public:
   /// Create GSSmoother.
   BlockGSSmoother(int t = 0, int it = 1) { type = t; iterations = it; }

   /// Create GSSmoother.
   BlockGSSmoother(const BlockMatrix &a, int t = 0, int it = 1)
      : BlockSmoother(a) { type = t; iterations = it; }

   /// Matrix vector multiplication with GS Smoother.
   virtual void Mult(const Vector &x, Vector &y) const;

   void BlockGaussSeidelForw(const Vector &x, Vector &y) const;
   void BlockGaussSeidelBack(const Vector &x, Vector &y) const;
};

/// Data type for scaled Jacobi-type smoother of sparse matrix
class BlockDSmoother : public BlockSmoother
{
protected:
   int type; // 0, 1, 2 - scaled Jacobi, scaled l1-Jacobi, scaled lumped-Jacobi
//    double scale;
   int iterations;

//    mutable Vector z;

public:
   /// Create Jacobi smoother.
   BlockDSmoother(int t = 0, int it = 1)
   { type = t; iterations = it; }

   /// Create Jacobi smoother.
   BlockDSmoother(const BlockMatrix &a, int t = 0, int it = 1);

   /// Matrix vector multiplication with Jacobi smoother.
   virtual void Mult(const Vector &x, Vector &y) const;
};

}

#endif
