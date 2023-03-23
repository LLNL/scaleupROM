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
#include "random.hpp"
#include "librom.h"
#include "mfem.hpp"
#include "linalg/svd/StaticSVD.h"
#include "linalg_utils.hpp"

using namespace mfem;

// uniform random number in [0., 1.]
double UniformRandom();

int main(int argc, char *argv[])
{
   // Let's not care for MPI at this moment.
   // MPI_Init(&argc, &argv);

   const int order = 1;
   const int dim = 2;
   const int nsample = 3;

   // 2 by 2 cartesian mesh.
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, false, 1.0, 1.0, false);

   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec);

   // librom snapshots.
   Array<CAROM::Vector*> snapshots(nsample);
   for (int s = 0; s < nsample; s++)
   {
      snapshots[s] = new CAROM::Vector(fes->GetTrueVSize(), false);
      for (int i = 0; i < fes->GetTrueVSize(); i++)
         snapshots[s]->item(i) = UniformRandom();
   }

   // librom svd
   // SVD class constructors are protected, so using BasisGenerator.
   CAROM::Options options(fes->GetTrueVSize(), 100, 1, true);
   std::string filename("test");
   CAROM::BasisGenerator *generator = new CAROM::BasisGenerator(options, false, filename);
   for (int s = 0; s < nsample; s++)
   {
      bool addSample = generator->takeSample(snapshots[s]->getData(), 0.0, 0.01);
   }

   // librom snapshot matrix.
   const CAROM::Matrix *carom_mat = generator->getSnapshotMatrix();
   printf("carom_mat (%d x %d)\n", carom_mat->numRows(), carom_mat->numColumns());
   for (int i = 0; i < carom_mat->numRows(); i++)
   {
      for (int j = 0; j < carom_mat->numColumns(); j++)
         printf("%.3E\t", carom_mat->item(i,j));
      printf("\n");
   }
   printf("\n");

   // librom pod basis.
   const CAROM::Matrix *basis = generator->getSpatialBasis();
   printf("basis (%d x %d)\n", basis->numRows(), basis->numColumns());
   for (int i = 0; i < basis->numRows(); i++)
   {
      for (int j = 0; j < basis->numColumns(); j++)
         printf("%.3E\t", basis->item(i,j));
      printf("\n");
   }
   printf("\n");

   // librom singular value;
   const CAROM::Vector *sv = generator->getSingularValues();
   printf("singular values (%d)\n", sv->dim());
   for (int i = 0; i < sv->dim(); i++)
      printf("%.3E\t", sv->item(i));
   printf("\n\n");

   // librom right singular vector.
   const CAROM::Matrix *right_basis = generator->getTemporalBasis();
   printf("right_basis (%d x %d)\n", right_basis->numRows(), right_basis->numColumns());
   for (int i = 0; i < right_basis->numRows(); i++)
   {
      for (int j = 0; j < right_basis->numColumns(); j++)
         printf("%.3E\t", right_basis->item(i,j));
      printf("\n");
   }
   printf("\n");

   // mass matrix for fem inner product.
   BilinearForm *a = new BilinearForm(fes);
   a->AddDomainIntegrator(new MassIntegrator);
   a->Assemble();
   SparseMatrix mat = a->SpMat();
   Vector one(fes->GetTrueVSize());
   one = 1.0;
   SparseMatrix identity(one);

   // Normal svd satisfies arithmatic orthonormality.
   CAROM::Matrix *orthonormality = new CAROM::Matrix(nsample, nsample, false, false);
   ComputeCtAB(identity, *basis, *basis, *orthonormality);

   printf("orthonormality (%d x %d)\n", orthonormality->numRows(), orthonormality->numColumns());
   for (int i = 0; i < orthonormality->numRows(); i++)
   {
      for (int j = 0; j < orthonormality->numColumns(); j++)
         printf("%.3E\t", orthonormality->item(i,j));
      printf("\n");
   }
   printf("\n");
   delete orthonormality;

   // Normal svd does not satisfy fem orthonormality.
   orthonormality = new CAROM::Matrix(nsample, nsample, false, false);
   ComputeCtAB(mat, *basis, *basis, *orthonormality);

   printf("fem orthonormality (%d x %d)\n", orthonormality->numRows(), orthonormality->numColumns());
   for (int i = 0; i < orthonormality->numRows(); i++)
   {
      for (int j = 0; j < orthonormality->numColumns(); j++)
         printf("%.3E\t", orthonormality->item(i,j));
      printf("\n");
   }
   printf("\n");
   delete orthonormality;

   printf("generalized svd with mass matrix\n\n");

   // square-root of mass matrix for generalized svd.
   mat.Finalize();
   DenseMatrix lmat_inv, lmat, tmp;
   {  // This takes way too long in the real case.
      mat.ToDenseMatrix(lmat_inv);
      lmat_inv.SquareRootInverse();
      lmat = lmat_inv;
      lmat.Invert();
   }
   
   // printf("L = sqrt(M) (%d x %d)\n", lmat.Height(), lmat.Width());
   // for (int i = 0; i < lmat.Height(); i++)
   // {
   //    for (int j = 0; j < lmat.Width(); j++)
   //       printf("%.3E\t", lmat(i,j));
   //    printf("\n");
   // }
   // printf("\n");

   // tmp.SetSize(lmat.Height(), lmat.Width());
   // mfem::Mult(lmat, lmat, tmp);
   // tmp -= *(mat.ToDenseMatrix());
   // printf("L * L ?= M (%d x %d)\n", tmp.Height(), tmp.Width());
   // for (int i = 0; i < tmp.Height(); i++)
   // {
   //    for (int j = 0; j < tmp.Width(); j++)
   //       printf("%.3E\t", tmp(i,j));
   //    printf("\n");
   // }
   // printf("\n");

   // librom matrix is stored in row-major, while mfem matrix is in column-major.
   // librom matrix does not support transpose for rectangular matrix.
   // since snapshots are initially mfem::Vector, multiply lmat on mfem space then save it as librom snapshots.
   Array<Vector*> transformed_snapshots(nsample);
   for (int s = 0; s < nsample; s++)
   {
      // this mfem vector is only a proxy to perfom linear algebra.
      // Vector tmp(snapshots[s]->getData(), fes->GetTrueVSize());
      transformed_snapshots[s] = new Vector(fes->GetTrueVSize());
      lmat.Mult(snapshots[s]->getData(), transformed_snapshots[s]->GetData());
   }

   delete generator;
   generator = new CAROM::BasisGenerator(options, false, filename);
   for (int s = 0; s < nsample; s++)
   {
      bool addSample = generator->takeSample(transformed_snapshots[s]->GetData(), 0.0, 0.01);
   }

   // librom pod basis.
   basis = generator->getSpatialBasis();
   // printf("basis (%d x %d)\n", basis->numRows(), basis->numColumns());
   // for (int i = 0; i < basis->numRows(); i++)
   // {
   //    for (int j = 0; j < basis->numColumns(); j++)
   //       printf("%.3E\t", basis->item(i,j));
   //    printf("\n");
   // }
   // printf("\n");

   // librom singular value;
   sv = generator->getSingularValues();
   printf("singular values (%d)\n", sv->dim());
   for (int i = 0; i < sv->dim(); i++)
      printf("%.3E\t", sv->item(i));
   printf("\n\n");

   // librom right singular vector.
   right_basis = generator->getTemporalBasis();
   printf("right_basis (%d x %d)\n", right_basis->numRows(), right_basis->numColumns());
   for (int i = 0; i < right_basis->numRows(); i++)
   {
      for (int j = 0; j < right_basis->numColumns(); j++)
         printf("%.3E\t", right_basis->item(i,j));
      printf("\n");
   }
   printf("\n");

   // Need to inverse-transform with lmat_inv.
   // since librom matrix is stored in row-major, it's not possible to take memory address of a column vector.
   // it is more convenient to perform linear algebra on librom space.
   // lmat_inv is symmetric, so its transpose is itself.
   // This is only a proxy to perform linear algebra, so not copying the data.
   CAROM::Matrix carom_lmat_inv(lmat_inv.GetData(), fes->GetTrueVSize(), fes->GetTrueVSize(), false, false);
   // for linear algebra, need an undistributed proxy.
   CAROM::Matrix basis_tmp(basis->getData(), basis->numRows(), basis->numColumns(), false, false);
   CAROM::Matrix *tmp_basis = carom_lmat_inv.mult(basis_tmp);
   // for linear algebra, need a distributed proxy.
   CAROM::Matrix transformed_basis(tmp_basis->getData(), tmp_basis->numRows(), tmp_basis->numColumns(), true, false);

   // Now transformed svd satisfies fem orthonormality.
   orthonormality = new CAROM::Matrix(nsample, nsample, false, false);
   ComputeCtAB(mat, transformed_basis, transformed_basis, *orthonormality);

   printf("fem orthonormality (%d x %d)\n", orthonormality->numRows(), orthonormality->numColumns());
   for (int i = 0; i < orthonormality->numRows(); i++)
   {
      for (int j = 0; j < orthonormality->numColumns(); j++)
         printf("%.3E\t", orthonormality->item(i,j));
      printf("\n");
   }
   printf("\n");
   delete orthonormality;

   {  // cholesky decomposition
      printf("using cholesky decomposition\n\n");
      mat.ToDenseMatrix(tmp);
      CholeskyFactors chol(tmp.GetData());
      chol.Factor(mat.NumRows());

      for (int s = 0; s < nsample; s++)
      {  // directly stores at snapshots.
         chol.UMult(fes->GetTrueVSize(), 1, snapshots[s]->getData());
      }

      delete generator;
      generator = new CAROM::BasisGenerator(options, false, filename);
      for (int s = 0; s < nsample; s++)
      {
         bool addSample = generator->takeSample(snapshots[s]->getData(), 0.0, 0.01);
      }

      basis = generator->getSpatialBasis();
      CAROM::Matrix transformed_basis(basis->numRows(), basis->numColumns(), true, false);
      for (int j = 0; j < basis->numColumns(); j++)
      {
         Vector basis_j(basis->numRows());
         for (int i = 0; i < basis->numRows(); i++) basis_j(i) = basis->item(i, j);
         chol.USolve(fes->GetTrueVSize(), 1, basis_j.GetData());
         for (int i = 0; i < basis->numRows(); i++) transformed_basis.item(i,j) = basis_j(i);
      }

      // Now transformed svd satisfies fem orthonormality.
      orthonormality = new CAROM::Matrix(nsample, nsample, false, false);
      ComputeCtAB(mat, transformed_basis, transformed_basis, *orthonormality);

      printf("fem orthonormality (%d x %d)\n", orthonormality->numRows(), orthonormality->numColumns());
      for (int i = 0; i < orthonormality->numRows(); i++)
      {
         for (int j = 0; j < orthonormality->numColumns(); j++)
            printf("%.3E\t", orthonormality->item(i,j));
         printf("\n");
      }
      printf("\n");
      delete orthonormality;

      // librom singular value;
      sv = generator->getSingularValues();
      printf("singular values (%d)\n", sv->dim());
      for (int i = 0; i < sv->dim(); i++)
         printf("%.3E\t", sv->item(i));
      printf("\n\n");

      // librom right singular vector.
      right_basis = generator->getTemporalBasis();
      printf("right_basis (%d x %d)\n", right_basis->numRows(), right_basis->numColumns());
      for (int i = 0; i < right_basis->numRows(); i++)
      {
         for (int j = 0; j < right_basis->numColumns(); j++)
            printf("%.3E\t", right_basis->item(i,j));
         printf("\n");
      }
      printf("\n");
   }

   for (int s = 0; s < transformed_snapshots.Size(); s++) delete transformed_snapshots[s];
   delete a;
   // should not delete since owned by generator.
   // delete carom_mat;
   delete generator;
   for (int s = 0; s < snapshots.Size(); s++) delete snapshots[s];
   delete fes;
   delete fec;

   // MPI_Finalize();
   return 0;
}
