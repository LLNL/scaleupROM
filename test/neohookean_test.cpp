// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <random>
#include "mms_suite.hpp"
#include "nlelast_integ.hpp"

using namespace std;
using namespace mfem;
namespace mfem
{
 void MFEMAssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                 const double weight, DenseMatrix &A) 
 {
    
    /// set params
    double g = 1.34;
    double mu = 3.14;
    double K = 4.13;

    int dof = DS.Height(), dim = DS.Width();
 
    DenseMatrix Z(dim);
    DenseMatrix G(dof, dim);
    DenseMatrix C(dof, dim);
 
    double dJ = J.Det();
    double sJ = dJ/g;
    double a  = mu*pow(dJ, -2.0/dim);
    double bc = a*(J*J)/dim;
    double b  = bc - K*sJ*(sJ - 1.0);
    double c  = 2.0*bc/dim + K*sJ*(2.0*sJ - 1.0);
 
    CalcAdjugateTranspose(J, Z);
    Z *= (1.0/dJ); // Z = J^{-t}
 
    MultABt(DS, J, C); // C = DS J^t
    MultABt(DS, Z, G); // G = DS J^{-1}
 
    a *= weight;
    b *= weight;
    c *= weight;
 
    // 1.
    for (int i = 0; i < dof; i++)
       for (int k = 0; k <= i; k++)
       {
          double s = 0.0;
          for (int d = 0; d < dim; d++)
          {
             s += DS(i,d)*DS(k,d);
          }
          s *= a;
 
          for (int d = 0; d < dim; d++)
          {
             A(i+d*dof,k+d*dof) += s;
          }
 
          if (k != i)
             for (int d = 0; d < dim; d++)
             {
                A(k+d*dof,i+d*dof) += s;
             }
       }
 
    a *= (-2.0/dim);
 
    // 2.
    for (int i = 0; i < dof; i++)
       for (int j = 0; j < dim; j++)
          for (int k = 0; k < dof; k++)
             for (int l = 0; l < dim; l++)
             {
                A(i+j*dof,k+l*dof) +=
                   a*(C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                   b*G(i,l)*G(k,j) + c*G(i,j)*G(k,l);
             }
 }
/* 
void SimpleAssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                                 const double weight, DenseMatrix &A) const
{
    int num_rows = DS.Height(); // Number of rows in DS
    int num_cols = DS.Width();  // Number of columns in DS

    // Initialize matrices
    DenseMatrix Z(dim);
    DenseMatrix G(dof, dim);
    DenseMatrix C(dof, dim);

    // Calculate determinant of J
    double dJ = J.Det();
    double sJ = dJ / g;
    double a = mu * pow(dJ, -2.0 / num_cols);
    double bc = a * (J * J) / num_cols;
    double b = bc - K * sJ * (sJ - 1.0);
    double c = 2.0 * bc / num_cols + K * sJ * (2.0 * sJ - 1.0);

    // Calculate adjugate transpose of J
    CalcAdjugateTranspose(J, Z);
    Z *= (1.0 / dJ); // Z = J^{-t}

    // Calculate products DS J^t and DS J^{-1}
    MultABt(DS, J, C); // C = DS J^t
    MultABt(DS, Z, G); // G = DS J^{-1}

    // Scale coefficients by weight
    a *= weight;
    b *= weight;
    c *= weight;

    // Calculate the first part of the assembly using matrix operations
    DenseMatrix DS_transpose = Transpose(DS); // Transpose of DS
    DenseMatrix DS_product = DS * DS_transpose; // Product of DS and its transpose
    DenseMatrix A_first_part = a * DS_product; // Scale the product by coefficient 'a'

    // Update A matrix with the first part of the assembly
    A.AddSubMatrix(num_rows, num_rows, 0, 0, A_first_part);
    A.AddSubMatrix(num_rows, num_rows, num_rows, num_rows, A_first_part.Transpose());

    // Calculate the second part of the assembly using matrix operations
    DenseMatrix G_transpose = Transpose(G); // Transpose of G
    DenseMatrix CG_product = C * G; // Product of C and G
    DenseMatrix GC_product = G * C; // Product of G and C
    DenseMatrix GG_product = G * G_transpose; // Product of G and its transpose

    // Combine terms with coefficients 'a', 'b', and 'c'
    DenseMatrix A_second_part = a * (CG_product + GC_product) + b * GG_product + c * GG_product;

    // Update A matrix with the second part of the assembly
    A.AddSubMatrix(num_rows, num_rows, 0, 0, A_second_part);
} */
} //namespace mfem

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

// Check that SimpleAssembleH works
TEST(Assemble_H, Test_NLElast)
{
   int dim = 2;
   int ndofs = 4;

   DenseMatrix J(dim);
   DenseMatrix DS(ndofs, dim);
   const double w = 1.2;
   DenseMatrix A(ndofs*dim, ndofs*dim);
   A = 0.0;

   double lower_bound = -1;
   double upper_bound = 1;
 
    uniform_real_distribution<double> unif(lower_bound,
                                           upper_bound);
    default_random_engine re;

   for (size_t i = 0; i < dim; i++)
      for (size_t j = 0; j < dim; j++)
    {
      J(i,j) = unif(re);
    }

    for (size_t i = 0; i < ndofs; i++)
      for (size_t j = 0; j < dim; j++)
    {
      DS(i,j) = unif(re);
    }
    
   MFEMAssembleH(J, DS, w, A);

    //diff_matrix.Add(-1.0, a1.SpMat());
    
   double norm_diff = A.FNorm();

    //cout << "Nonlinear Stiffness matrix norm: " << J->MaxNorm() << endl;
    //cout << "Linear Stiffness matrix norm: " << a1.SpMat().MaxNorm() << endl;
    cout << "Stiffness matrix difference norm: " << norm_diff << endl;

   return;
} 
 

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}