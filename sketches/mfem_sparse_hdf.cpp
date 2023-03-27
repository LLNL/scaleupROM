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

#include <fstream>
#include <iostream>
#include "utils/HDFDatabase.h"
#include "mfem.hpp"
#include "random.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   const int nr = 5;
   const int nc = 7;

   // LIL format sparse matrix data
   const int ne = 6;
   Array<int> i_idx(ne), j_idx(ne);
   Vector data(ne);
   for (int e = 0; e < ne; e++)
   {
      data(e) = UniformRandom();
      i_idx[e] = rand() % nr;
      j_idx[e] = rand() % nc;
   }

   // LIL format sparse matrix initiation.
   SparseMatrix test(nr, nc);
   for (int e = 0; e < ne; e++)
      test.Set(i_idx[e], j_idx[e], data(e));

   // transform to CSR format.
   test.Finalize();
   DenseMatrix tmp;
   test.ToDenseMatrix(tmp);
   printf("Input Matrix (%dx%d)\n", tmp.NumRows(), tmp.NumCols());
   for (int i = 0; i < tmp.NumRows(); i++)
   {
      for (int j = 0; j < tmp.NumCols(); j++)
      {
         printf("%.3E\t", tmp(i,j));
      }
      printf("\n");
   }
   printf("\n");

   const int nnz = test.NumNonZeroElems();
   const int *itest = test.ReadI();
   const int *jtest = test.ReadJ();
   const double *datatest = test.ReadData();

   printf("Extracting from SparseMatrix.\n");
   printf("I\tJ\tData\n");
   for (int e = 0; e < nnz; e++)
   {
      printf("%d\t%d\t%.3E\n", itest[e], jtest[e], datatest[e]);
   }
   printf("\n");

   bool success = false;
   CAROM::HDFDatabase hdfIO;
   std::string filename = "test.h5";
   success = hdfIO.create(filename);
   assert(success);

   hdfIO.putIntegerArray("I", itest, nnz);
   hdfIO.putIntegerArray("J", jtest, nnz);
   hdfIO.putDoubleArray("data", datatest, nnz);

   success = hdfIO.close();
   assert(success);

   success = hdfIO.open(filename, "r");
   assert(success);

   const int data_size = hdfIO.getDoubleArraySize("data");
   assert(data_size == nnz);
   Array<int> i_read(data_size), j_read(data_size);
   Vector data_read(data_size);
   hdfIO.getDoubleArray("data", data_read.Write(), data_size);
   hdfIO.getIntegerArray("I", i_read.Write(), data_size);
   hdfIO.getIntegerArray("J", j_read.Write(), data_size);

   success = hdfIO.close();
   assert(success);

   printf("Reading from test.h5\n");
   printf("I\tJ\tData\n");
   for (int e = 0; e < data_size; e++)
   {
      printf("%d\t%d\t%.3E\n", i_read[e], j_read[e], data_read[e]);
   }
   printf("\n");

   // Need to transfer the ownership to avoid segfault or double-free.
   int *ip, *jp;
   i_read.StealData(&ip);
   j_read.StealData(&jp);
   SparseMatrix test_read(ip, jp, data_read.StealData(), nr, nc);

   test_read.ToDenseMatrix(tmp);
   printf("Output Matrix (%dx%d)\n", tmp.NumRows(), tmp.NumCols());
   for (int i = 0; i < tmp.NumRows(); i++)
   {
      for (int j = 0; j < tmp.NumCols(); j++)
      {
         printf("%.3E\t", tmp(i,j));
      }
      printf("\n");
   }
   printf("\n");

   return 0;
}