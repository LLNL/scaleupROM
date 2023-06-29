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

#include "block_smoother.hpp"
#include "etc.hpp"
#include "linalg_utils.hpp"

using namespace mfem;

namespace mfem
{

/*
   BlockSmoother
*/

BlockSmoother::BlockSmoother(const BlockMatrix &a)
   : MatrixInverse(a),
     oper(&a),
     num_row_blocks(a.NumRowBlocks()),
     num_col_blocks(a.NumColBlocks()),
     row_offsets(a.RowOffsets()),
     col_offsets(a.ColOffsets())
{
   assert(num_row_blocks == num_col_blocks);

   diag_inv.SetSize(num_row_blocks);
   diag_inv = NULL;
   for (int b = 0; b < num_row_blocks; b++)
   {
      diag_inv[b] = oper->GetBlock(b, b).ToDenseMatrix();
      diag_inv[b]->Invert();
   }
}

BlockSmoother::~BlockSmoother()
{
   DeletePointers(diag_inv);
}

void BlockSmoother::SetOperator(const Operator &a)
{
   oper = dynamic_cast<const BlockMatrix*>(&a);
   if (oper == NULL)
   {
      mfem_error("BlockSmoother::SetOperator : not a BlockMatrix!");
   }
   height = oper->Height();
   width = oper->Width();
}

void BlockSmoother::PrintOperator()
{
   std::string filename = "BlockSmoother.op.txt";
   PrintMatrix(*oper, filename);

   for (int k = 0; k < num_row_blocks; k++)
   {
      filename = "BlockSmoother.diag_inv" + std::to_string(k) + ".txt";
      PrintMatrix(*diag_inv[k], filename);
   }
}

/*
   BlockGSSmoother
*/

/// Matrix vector multiplication with Block GS Smoother.
void BlockGSSmoother::Mult(const Vector &x, Vector &y) const
{
   if (!iterative_mode)
   {
      y = 0.0;
   }
   for (int i = 0; i < iterations; i++)
   {
      if (type != 2)
      {
         BlockGaussSeidelForw(x, y);
      }
      if (type != 1)
      {
         BlockGaussSeidelBack(x, y);
      }
   }
}

void BlockGSSmoother::BlockGaussSeidelForw(const Vector &x, Vector &y) const
{
   Vector xblockview, yblockview, tmp;

   for (int i = 0; i < num_row_blocks; i++)
   {   
      tmp.SetSize(row_offsets[i+1] - row_offsets[i]);
      tmp = 0.0;

      for (int j = 0; j < num_col_blocks; j++)
      {
         if (i == j) continue;

         if (!oper->IsZeroBlock(i, j))
         {
            yblockview.SetDataAndSize(
               y.GetData() + col_offsets[j],
               col_offsets[j+1] - col_offsets[j]);

            // If i < j, yblockview is new y.
            // If i > j, yblockview is old y.
            oper->GetBlock(i, j).AddMult(yblockview, tmp, -1.0);
         }
      }
      for (int k = 0; k < tmp.Size(); k++)
         assert(!std::isnan(tmp(k)));

      xblockview.SetDataAndSize(x.GetData() + row_offsets[i],
                                row_offsets[i+1] - row_offsets[i]);
      tmp += xblockview;
      for (int k = 0; k < tmp.Size(); k++)
         assert(!std::isnan(tmp(k)));

      yblockview.SetDataAndSize(y.GetData() + row_offsets[i],
                                row_offsets[i+1] - row_offsets[i]);
      diag_inv[i]->Mult(tmp, yblockview);
      for (int k = 0; k < yblockview.Size(); k++)
         assert(!std::isnan(yblockview(k)));
   }
}

void BlockGSSmoother::BlockGaussSeidelBack(const Vector &x, Vector &y) const
{
   Vector xblockview, yblockview, tmp;

   for (int i = num_row_blocks-1; i >= 0; i--)
   {   
      tmp.SetSize(row_offsets[i+1] - row_offsets[i]);
      tmp = 0.0;

      for (int j = num_col_blocks-1; j >= 0; j--)
      {
         if (i == j) continue;

         if (!oper->IsZeroBlock(i, j))
         {
            yblockview.SetDataAndSize(
               y.GetData() + col_offsets[j],
               col_offsets[j+1] - col_offsets[j]);

            // If i > j, yblockview is new y.
            // If i < j, yblockview is old y.
            oper->GetBlock(i, j).AddMult(yblockview, tmp, -1.0);
         }
      }
      for (int k = 0; k < tmp.Size(); k++)
         assert(!std::isnan(tmp(k)));

      xblockview.SetDataAndSize(x.GetData() + row_offsets[i],
                                row_offsets[i+1] - row_offsets[i]);
      tmp += xblockview;
      for (int k = 0; k < tmp.Size(); k++)
         assert(!std::isnan(tmp(k)));

      yblockview.SetDataAndSize(y.GetData() + row_offsets[i],
                                row_offsets[i+1] - row_offsets[i]);
      diag_inv[i]->Mult(tmp, yblockview);
      for (int k = 0; k < yblockview.Size(); k++)
         assert(!std::isnan(yblockview(k)));
   }
}

/*
   BlockDSmoother
*/

/// Create the Jacobi smoother.
BlockDSmoother::BlockDSmoother(const BlockMatrix &a, int t, int it)
   : BlockSmoother(a)
{
   type = t;
   // scale = s;
   iterations = it;
}

/// Matrix vector multiplication with Jacobi smoother.
void BlockDSmoother::Mult(const Vector &x, Vector &y) const
{
   if (!iterative_mode && type == 0 && iterations == 1)
   {
      Vector xblockview, yblockview;
      for (int i = 0; i < num_row_blocks; i++)
      {
         xblockview.SetDataAndSize(x.GetData() + row_offsets[i],
                                 row_offsets[i+1] - row_offsets[i]);
         yblockview.SetDataAndSize(y.GetData() + row_offsets[i],
                                 row_offsets[i+1] - row_offsets[i]);
         diag_inv[i]->Mult(xblockview, yblockview);
      }
      return;
   }

   mfem_error("BlockDSmoother::Mult is implemented only for standard preconditioner!\n");

   // z.SetSize(width);

   // Vector *r = &y, *p = &z;

   // if (iterations % 2 == 0)
   // {
   //    Swap<Vector*>(r, p);
   // }

   // if (!iterative_mode)
   // {
   //    *p = 0.0;
   // }
   // else if (iterations % 2)
   // {
   //    *p = y;
   // }
   // for (int i = 0; i < iterations; i++)
   // {
   //    if (type == 0)
   //    {
   //       oper->Jacobi(x, *p, *r, scale, use_abs_diag);
   //    }
   //    else if (type == 1)
   //    {
   //       oper->Jacobi2(x, *p, *r, scale);
   //    }
   //    else if (type == 2)
   //    {
   //       oper->Jacobi3(x, *p, *r, scale);
   //    }
   //    else
   //    {
   //       mfem_error("DSmoother::Mult wrong type");
   //    }
   //    Swap<Vector*>(r, p);
   // }
}

}