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
#include "librom.h"

// uniform random number in [0., 1.]
double UniformRandom();

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   CAROM::Options* rom_options;
   CAROM::BasisGenerator *basis_generator;
   CAROM::BasisReader *basis_reader;
   std::string basis_prefix = "librom_svd_test";

   int max_num_snapshots = 100;
   bool update_right_SV = false;
   bool incremental = false;

   int dim = 5;
   int num_samples = 3;
   double* sample1 = new double[dim] {0.5377, 1.8339, -2.2588, 0.8622, 0.3188};
   double* sample2 = new double[dim] {-1.3077, -0.4336, 0.3426, 3.5784, 2.7694};
   double* sample3 = new double[dim] {-1.3499, 3.0349, 0.7254, -0.0631, 0.7147};

   {  // reading snapshots.
      std::string filename(basis_prefix + "_basis");
      bool addSample;

      rom_options = new CAROM::Options(dim, max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, filename);
      addSample = basis_generator->takeSample(sample1, 0.0, 0.01);
      // basis_generator->writeSnapshot();
      addSample = basis_generator->takeSample(sample2, 0.0, 0.01);
      // basis_generator->writeSnapshot();
      addSample = basis_generator->takeSample(sample3, 0.0, 0.01);
      // basis_generator->writeSnapshot();

      basis_generator->endSamples();
      const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
      printf("Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3f\t", rom_sv->item(d));
      printf("\n");

      const CAROM::Matrix* spatialbasis;
      spatialbasis = basis_generator->getSpatialBasis();
      printf("Basis.\n");
      for (int i = 0; i < spatialbasis->numRows(); i++) {
         for (int j = 0; j < spatialbasis->numColumns(); j++) {
            printf("%.5E\t", spatialbasis->item(i, j));
         }
         printf("\n");
      }
      printf("\n");
   }

   MPI_Finalize();
   return 0;
}

double UniformRandom()
{
   return static_cast<double> (rand()) / static_cast<double> (RAND_MAX);
}