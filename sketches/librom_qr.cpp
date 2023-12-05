// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iostream>
#include "librom.h"
#include "etc.hpp"

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   printf("Example from RandomizedSVDTest, Test_RandomizedSVD.\n\n");

   double* sample1 = new double[5] {0.5377, 1.8339, -2.2588, 0.8622, 0.3188};
   double* sample2 = new double[5] {-1.3077, -0.4336, 0.3426, 3.5784, 2.7694};
   double* sample3 = new double[5] {-1.3499, 3.0349, 0.7254, -0.0631, 0.7147};

   CAROM::Matrix *snapshot_matrix = new CAROM::Matrix(5, 3, true);
   for (int i = 0; i < snapshot_matrix->numRows(); i++) {
      snapshot_matrix->item(i, 0) = sample1[i];
      snapshot_matrix->item(i, 1) = sample2[i];
      snapshot_matrix->item(i, 2) = sample3[i];
   }

   printf("Snapshot matrix from example.\n");
   for (int i = 0; i < snapshot_matrix->numRows(); i++) {
      for (int j = 0; j < snapshot_matrix->numColumns(); j++) {
         printf("%.15E\t", snapshot_matrix->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   CAROM::Matrix *rand_mat = new CAROM::Matrix(3, 3, false);
   for (int i = 0; i < snapshot_matrix->numColumns(); i++) {
      for (int j = 0; j < 3; j++) {
         rand_mat->item(i, j) = 1;
      }
   }

   printf("rand_mat (%d x %d) in RandomizedSVD::computeSVD (d_debug_algorithm).\n", rand_mat->numRows(), rand_mat->numColumns());
   for (int i = 0; i < rand_mat->numRows(); i++) {
      for (int j = 0; j < rand_mat->numColumns(); j++) {
         printf("%.15E\t", rand_mat->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   CAROM::Matrix* rand_proj = snapshot_matrix->mult(rand_mat);
   printf("rand_proj (%d x %d) in RandomizedSVD::computeSVD.\n", rand_proj->numRows(), rand_proj->numColumns());
   for (int i = 0; i < rand_proj->numRows(); i++) {
      for (int j = 0; j < rand_proj->numColumns(); j++) {
         printf("%.15E\t", rand_proj->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   CAROM::Matrix* Q = rand_proj->qr_factorize();
   printf("Q (%d x %d) in RandomizedSVD::computeSVD.\n", Q->numRows(), Q->numColumns());
   for (int i = 0; i < Q->numRows(); i++) {
      for (int j = 0; j < Q->numColumns(); j++) {
         printf("%.15E\t", Q->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   // Now perturb the snapshot matrix within machine precision.
   for (int i = 0; i < snapshot_matrix->numRows(); i++) {
      for (int j = 0; j < snapshot_matrix->numColumns(); j++) {
         snapshot_matrix->item(i, j) *= 1.0 + 1.0e-15 * (-1.0 + 2.0 * UniformRandom());
      }
   }

   printf("Perturbed snapshot matrix.\n");
   for (int i = 0; i < snapshot_matrix->numRows(); i++) {
      for (int j = 0; j < snapshot_matrix->numColumns(); j++) {
         printf("%.15E\t", snapshot_matrix->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   rand_proj = snapshot_matrix->mult(rand_mat);
   Q = rand_proj->qr_factorize();
   printf("After perturbation, 2nd/3rd columns of Q totally changed.\n");
   for (int i = 0; i < Q->numRows(); i++) {
      for (int j = 0; j < Q->numColumns(); j++) {
         printf("%.15E\t", Q->item(i, j));
      }
      printf("\n");
   }
   printf("\n");

   printf("Both Q are legitimate QR decomposition of rand_proj within 'machine precision'.\n");

   MPI_Finalize();
   return 0;
}