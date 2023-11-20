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

// TODO: add more tests from sketches/yaml_example.cpp.

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}