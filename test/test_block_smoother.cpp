// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "block_smoother.hpp"
#include "etc.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-13;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(BlockSmootherTest, BlockGSSmoother)
{
   const int nblocks = 4;
   Array<int> block_offsets(nblocks+1);
   block_offsets[0] = 0;
   block_offsets[1] = 2;
   block_offsets[2] = 3;
   block_offsets[3] = 4;
   block_offsets[4] = 5;
   block_offsets.PartialSum();

   Array2D<SparseMatrix *> mats(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
      for (int j = 0; j < nblocks; j++)
      {
         mats(i, j) = new SparseMatrix(block_offsets[i+1] - block_offsets[i],
                                       block_offsets[j+1] - block_offsets[j]);

         for (int ii = 0; ii < block_offsets[i+1] - block_offsets[i]; ii++)
            for (int jj = 0; jj < block_offsets[j+1] - block_offsets[j]; jj++)
            {
               double val = UniformRandom();
               if (i == j) val *= 1.0e1;
               mats(i, j)->Set(ii, jj, val);
            }

         mats(i, j)->Finalize();
      }

   BlockMatrix M(block_offsets), L(block_offsets), U(block_offsets);
   BlockMatrix DL(block_offsets), DU(block_offsets);
   for (int i = 0; i < nblocks; i++)
      for (int j = 0; j < nblocks; j++)
      {
         M.SetBlock(i, j, mats(i, j));
         if (i > j)
         {
            L.SetBlock(i, j, mats(i, j));
            DL.SetBlock(i, j, mats(i, j));
         }
         else if (i < j)
         {
            U.SetBlock(i, j, mats(i, j));
            DU.SetBlock(i, j, mats(i, j));
         }
         else
         {
            DL.SetBlock(i, j, mats(i, j));
            DU.SetBlock(i, j, mats(i, j));
         }
      }

   BlockGSSmoother gs(M);
   DenseMatrix &DLinv(*(DL.CreateMonolithic()->ToDenseMatrix()));
   DenseMatrix &DUinv(*(DU.CreateMonolithic()->ToDenseMatrix()));
   DLinv.Invert();
   DUinv.Invert();

   Vector x(block_offsets.Last()), y0(block_offsets.Last());
   Vector tmp(block_offsets.Last());
   tmp = 0.0;
   for (int k = 0; k < x.Size(); k++)
   {
      x(k) = UniformRandom();
      y0(k) = UniformRandom();
   }
   
   Vector fwy(block_offsets.Last()), fwy_true(block_offsets.Last());
   fwy = y0; fwy_true = y0;

   // true GS forward
   tmp = 0.0;
   U.AddMult(y0, tmp, -1.0);
   tmp += x;
   DLinv.Mult(tmp, fwy_true);

   // GSSmoother forward
   gs.BlockGaussSeidelForw(x, fwy);

   double error = 0.0;
   for (int k = 0; k < fwy.Size(); k++)
      error = max(error, abs(fwy(k) - fwy_true(k)));
   printf("GS forward error: %.5E\n", error);
   EXPECT_TRUE(error < threshold);

   Vector bwy(block_offsets.Last()), bwy_true(block_offsets.Last());
   bwy = y0; bwy_true = y0;

   // true GS forward
   tmp = 0.0;
   L.AddMult(y0, tmp, -1.0);
   tmp += x;
   DUinv.Mult(tmp, bwy_true);

   // GSSmoother forward
   gs.BlockGaussSeidelBack(x, bwy);

   error = 0.0;
   for (int k = 0; k < bwy.Size(); k++)
      error = max(error, abs(bwy(k) - bwy_true(k)));
   printf("GS backward error: %.5E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(BlockSmootherTest, BlockDSmoother)
{
   const int nblocks = 4;
   Array<int> block_offsets(nblocks+1);
   block_offsets[0] = 0;
   block_offsets[1] = 2;
   block_offsets[2] = 3;
   block_offsets[3] = 4;
   block_offsets[4] = 5;
   block_offsets.PartialSum();

   Array2D<SparseMatrix *> mats(nblocks, nblocks);
   for (int i = 0; i < nblocks; i++)
      for (int j = 0; j < nblocks; j++)
      {
         mats(i, j) = new SparseMatrix(block_offsets[i+1] - block_offsets[i],
                                       block_offsets[j+1] - block_offsets[j]);

         for (int ii = 0; ii < block_offsets[i+1] - block_offsets[i]; ii++)
            for (int jj = 0; jj < block_offsets[j+1] - block_offsets[j]; jj++)
            {
               double val = UniformRandom();
               if (i == j) val *= 1.0e1;
               mats(i, j)->Set(ii, jj, val);
            }

         mats(i, j)->Finalize();
      }

   BlockMatrix M(block_offsets), D(block_offsets);
   for (int i = 0; i < nblocks; i++)
      for (int j = 0; j < nblocks; j++)
      {
         M.SetBlock(i, j, mats(i, j));
         if (i == j)
            D.SetBlock(i, j, mats(i, j));
      }

   BlockDSmoother jacobi(M);
   DenseMatrix &Dinv(*(D.CreateMonolithic()->ToDenseMatrix()));
   Dinv.Invert();

   Vector x(block_offsets.Last()), y0(block_offsets.Last());
   for (int k = 0; k < x.Size(); k++)
   {
      x(k) = UniformRandom();
      y0(k) = UniformRandom();
   }
   
   Vector fwy(block_offsets.Last()), fwy_true(block_offsets.Last());

   // true Jacobi forward
   Dinv.Mult(x, fwy_true);

   // GSSmoother forward
   jacobi.Mult(x, fwy);

   double error = 0.0;
   for (int k = 0; k < fwy.Size(); k++)
      error = max(error, abs(fwy(k) - fwy_true(k)));
   printf("Jacobi1 error: %.5E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

// TODO: add more tests from sketches/yaml_example.cpp.

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}