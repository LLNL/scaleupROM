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
                   a * (C(i,j)*G(k,l) + G(i,j)*C(k,l)) +
                   b*G(i,l)*G(k,j)  + c*G(i,j)*G(k,l); 
             }
 }

void SimpleAssembleH(const DenseMatrix &J, const DenseMatrix &DS,
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
    const double a2 = a * (-2.0/dim);

   for (size_t i = 0; i < dof; i++) 
      {
         for (size_t j = 0; j < dim; j++) // Looping over each entry in residual
         {
            const int ij = j * dof + i;

            for (size_t m = 0; m < dof; m++) 
            for (size_t n = 0; n < dim; n++) // Looping over derivatives with respect to U
            {
               const int mn = n * dof + m;
               double temp = 0.0;
               for (size_t k = 0; k < dim; k++)
               {
                  const int S_jk = k * dim + j;
                  //temp += Dmat(S_jk, mn) * w * gshape(i,k);
                  const double s1 = (j==n) ?  a * DS(m,k) : 0.0;
                  const double s2 = a2 * (J(j,k)*G(m,n) + Z(j,k)*C(m,n))
                    + b*Z(n,k)*G(m,j) + c*Z(j,k)*G(m,n);

                  temp += DS(i,k)*(s1 + s2);
               } 
               A(ij, mn) += temp;
            }
         }
      }
} 
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
   DenseMatrix A1(ndofs*dim, ndofs*dim);
   DenseMatrix A2(ndofs*dim, ndofs*dim);
   A1 = 0.0;
   A2 = 0.0;

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
    
   MFEMAssembleH(J, DS, w, A1);
   SimpleAssembleH(J, DS, w, A2);

   PrintMatrix(A1, "A1.txt");
   PrintMatrix(A2, "A2.txt");

    cout << "MFEM Stiffness matrix norm: " << A1.FNorm() << endl;
    cout << "ScaleupROM Stiffness matrix norm: " << A2.FNorm() << endl;
    A1.Add(-1.0, A2);
   double norm_diff = A1.FNorm();
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