// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "linalg_utils.hpp"
#include "etc.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(linalg_test, MultSubMatrix)
{
   const int nrow = 15, ncol = 7, nsample = 4;
   DenseMatrix mat(nrow, ncol);
   Vector x(ncol), matx(nrow), y(nsample);
   Array<int> rows(nsample);

   for (int i = 0; i < nrow; i++)
      for (int j = 0; j < ncol; j++)
         mat(i, j) = UniformRandom();
   
   for (int j = 0; j < ncol; j++)
      x(j) = UniformRandom();

   for (int s = 0; s < nsample; s++)
      rows[s] = UniformRandom(0, nrow-1);

   y = 0.0;
   matx = 0.0;

   mat.Mult(x, matx);
   MultSubMatrix(mat, rows, x, y);

   for (int s = 0; s < nsample; s++)
      EXPECT_DOUBLE_EQ(y(s), matx(rows[s]));

   return;
}

TEST(linalg_test, AddMultTransposeSubMatrix)
{
   const int nrow = 15, ncol = 7, nsample = 4;
   DenseMatrix mat(nrow, ncol);
   Vector x(nsample), xtotal(nrow), matTx(ncol), y(ncol);
   Array<int> rows(nsample);

   for (int i = 0; i < nrow; i++)
      for (int j = 0; j < ncol; j++)
         mat(i, j) = UniformRandom();
   
   for (int j = 0; j < nsample; j++)
      x(j) = UniformRandom();

   for (int s = 0; s < nsample; s++)
      rows[s] = UniformRandom(0, nrow-1);

   xtotal = 0.0;
   for (int s = 0; s < nsample; s++)
      // there can be duplicated rows.
      xtotal(rows[s]) += x(s);

   y = 0.0;
   matTx = 0.0;

   mat.MultTranspose(xtotal, matTx);
   AddMultTransposeSubMatrix(mat, rows, x, y);

   for (int s = 0; s < ncol; s++)
      EXPECT_DOUBLE_EQ(y(s), matTx(s));

   return;
}

TEST(linalg_test, Orthonormalize)
{
   const double thre = 1.0e-15;
   const int nrow = 15, ncol1 = 7, ncol = 5;
   DenseMatrix mat1(nrow, ncol1);
   DenseMatrix mat(nrow, ncol);
   DenseMatrix test(ncol, ncol), test1(ncol1, ncol);

   for (int i = 0; i < nrow; i++)
   {
      for (int j = 0; j < ncol; j++)
         mat(i, j) = UniformRandom();

      for (int j = 0; j < ncol1; j++)
         mat1(i, j) = UniformRandom();
   }

   modifiedGramSchmidt(mat1);
   Vector scale(ncol1);
   for (int d = 0; d < ncol1; d++)
      scale(d) = UniformRandom();
   mat1.RightScaling(scale);

   Orthonormalize(mat1, mat);

   Vector tmp1, tmp2;
   for (int j = 0; j < ncol; j++)
   {
      mat.GetColumnReference(j, tmp1);
      test.GetColumnReference(j, tmp2);
      mat.MultTranspose(tmp1, tmp2);
   }
   for (int i = 0; i < ncol; i++)
      for (int j = 0; j < ncol; j++)
      {
         double val = (i == j) ? 1.0 : 0.0;
         EXPECT_NEAR(test(i, j), val, thre);
      };
   
   for (int j = 0; j < ncol; j++)
   {
      mat.GetColumnReference(j, tmp1);
      test1.GetColumnReference(j, tmp2);
      mat1.MultTranspose(tmp1, tmp2);
   }
   for (int i = 0; i < ncol1; i++)
      for (int j = 0; j < ncol; j++)
         EXPECT_NEAR(test1(i, j), 0.0, thre);

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