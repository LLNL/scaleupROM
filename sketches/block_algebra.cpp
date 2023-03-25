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
#include "mfem.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   const int numSub = 5;
   const int ne = 1;
   const int order = 1;

   Mesh mesh = Mesh::MakeCartesian2D(ne, ne, Element::QUADRILATERAL, false, 1.0, 1.0, false);
   // Mesh mesh = Mesh::MakeCartesian1D(ne, 1.0);
   const int dim = mesh.Dimension();
   const int udim = dim;

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   Array<FiniteElementSpace *> fes(numSub);
   for (int k = 0; k < numSub; k++)
   {
      fes[k] = new FiniteElementSpace(&mesh, fec, udim);
   }

   Array<int> block_offsets, domain_offsets;
   block_offsets.SetSize(udim * numSub + 1);
   domain_offsets.SetSize(numSub + 1);
   block_offsets[0] = 0;
   domain_offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
   {
      domain_offsets[i + 1] = fes[i]->GetTrueVSize();
      for (int d = 0; d < udim; d++)
      {
         block_offsets[d + i * udim + 1] = fes[i]->GetNDofs();
      }
   }
   block_offsets.PartialSum();
   domain_offsets.PartialSum();

   // Solution/Rhs are set with block offsets.
   BlockVector *U = new BlockVector(block_offsets);
   BlockVector *RHS = new BlockVector(block_offsets);

   // GridFunctions are set with domain offsets.
   Array<GridFunction*> us(numSub);
   for (int m = 0; m < numSub; m++)
   {
      us[m] = new GridFunction(fes[m], U->GetBlock(m * udim), 0);
      (*us[m]) = 0.0;
   }

   // Example rhs vector function.
   Vector rhs_val(udim);
   for (int d = 0; d < udim; d++) rhs_val(d) = 0.5 * (d+1);
   VectorConstantCoefficient rhs_coeff(rhs_val);

   // operators are also set with domain offsets.
   Array<LinearForm*> bs(numSub);
   Array<BilinearForm*> as(numSub);
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m * udim).GetData());
      // Vector and Scalar system will need different integrator classes.
      // While it's okay to have udim for scalar system, it is not reasonable
      // to change the integrator/function coefficients even for scalar.
      bs[m]->AddDomainIntegrator(new VectorDomainLFIntegrator(rhs_coeff));

      as[m] = new BilinearForm(fes[m]);
      // This also cannot use scalar integrator for vector system.
      as[m]->AddDomainIntegrator(new VectorMassIntegrator);

      bs[m]->Assemble();
      as[m]->Assemble();

      // Do we really need SyncAliasMemory?
      // bs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
      as[m]->Finalize();

// {
//    DenseMatrix tmp;
//    as[m]->SpMat().ToDenseMatrix(tmp);
//    printf("as[%d]\n", m);
//    for (int i = 0; i < tmp.NumRows(); i++)
//    {
//       for (int j = 0; j < tmp.NumCols(); j++)
//       {
//          printf("%.3E\t", tmp(i,j));
//       }
//       printf("\n");
//    }
//    printf("\n");
// }
   }

   // Set up block matrixes, based on domain offsets.
   Array2D<SparseMatrix*> mats(numSub, numSub);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i == j) {
            mats(i, i) = &(as[i]->SpMat());
         } else {
            mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());
         }
      }
   }

   // globalMat is based on domain offsets, while U, RHS are based on block_offsets.
   BlockMatrix *globalMat = new BlockMatrix(domain_offsets);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i != j) mats(i, j)->Finalize();

         globalMat->SetBlock(i, j, mats(i, j));
      }
   }

   int maxIter = 100;
   double rtol = 1.0e-10;
   double atol = 1.0e-10;
   int print_level = 1;

   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(*globalMat);
   solver.SetPrintLevel(print_level);

   BlockDiagonalPreconditioner *globalPrec;
   if (true)
   {
      // This works both with block_offsets and domain_offsets. But probably the system is too simple to check the difference.
      globalPrec = new BlockDiagonalPreconditioner(domain_offsets);
      solver.SetPreconditioner(*globalPrec);
   }

   *U = 0.0;
   solver.Mult(*RHS, *U);

   printf("Solution\n");
   for (int d = 0; d < U->Size(); d++)
      printf("%.3E\n", (*U)[d]);

   return 0;
}