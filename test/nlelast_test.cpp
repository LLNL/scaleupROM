// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <random>
#include "mms_suite.hpp"
#include "nlelast_integ.hpp"

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0);
}

// Closed because of work on nonlinear version
TEST(TempLinStiffnessMatrices, Test_NLElast)
{
   // Temporary test that the nonlinear operators do the correct things
    Mesh mesh("../examples/linelast/meshes/joint2D.mesh", 1, 1);
   int dim = mesh.Dimension();
   int order = 1;
   double alpha = -1.0;
   double kappa = -1.0;
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec, dim);

VectorFunctionCoefficient init_x(dim, InitDisplacement);

   Vector lambda(mesh.attributes.Max());
   lambda = 1.0;      // Set lambda = 1 for all element attributes.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   mu = 1.0;      // Set mu = 1 for all element attributes.
   PWConstCoefficient mu_c(mu);

    Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   BilinearForm a1(&fespace);
   a1.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
   //a1.AddInteriorFaceIntegrator(
      //new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa)); //Needed??
   //a1.AddBdrFaceIntegrator(
     // new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);
   a1.Assemble();


   TestLinModel model(1.0, 1.0);
   NonlinearForm a2(&fespace);
   a2.AddDomainIntegrator(new HyperelasticNLFIntegratorHR(&model));
   //a2.AddInteriorFaceIntegrator(
      //new DGHyperelasticNLFIntegrator(lambda_c, mu_c, alpha, kappa)); //Needed?
   //a2.AddBdrFaceIntegrator(
      //new DGHyperelasticNLFIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);
    
    // Create vectors to hold the values of the forms
    Vector x, y1, y2;

    x.SetSize(fespace.GetTrueVSize());

    double lower_bound = -1;
    double upper_bound = 1;
 
    uniform_real_distribution<double> unif(lower_bound,
                                           upper_bound);
    default_random_engine re;

    for (size_t i = 0; i < x.Size(); i++)
    {
      x[i] = unif(re);
    }
    

    y1.SetSize(fespace.GetTrueVSize());
    y1 = 0.0;

    y2.SetSize(fespace.GetTrueVSize());
    y2 = 0.0;
    cout<< 5 <<endl;

    a1.Mult(x, y1);
    a2.Mult(x, y2);

   double norm_diff = 0.0;
   cout << "Linear residual norm: " << y1.Norml2() << endl;
    cout << "Nonlinear residual norm: " << y2.Norml2() << endl;

    y1 -= y2;
    norm_diff = y1.Norml2();
    cout << "Residual difference norm: " << norm_diff << endl;

    Operator *J_op = &(a2.GetGradient(x));
    SparseMatrix *J = dynamic_cast<SparseMatrix *>(J_op);

    SparseMatrix diff_matrix(*J);
    diff_matrix.Add(-1.0, a1.SpMat());
    
    norm_diff = diff_matrix.MaxNorm();

    cout << "Linear Stiffness matrix norm: " << J->MaxNorm() << endl;
    cout << "Nonlinear Stiffness matrix norm: " << a1.SpMat().MaxNorm() << endl;
    cout << "Stiffness matrix difference norm: " << norm_diff << endl;


    LinearForm b1(&fespace);
   b1.AddBdrFaceIntegrator(
      new DGElasticityDirichletLFIntegrator(
         init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
   b1.Assemble();

   LinearForm b2(&fespace);
   b2.AddBdrFaceIntegrator(
      new DGHyperelasticDirichletNLFIntegrator(
         init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
   b2.Assemble();

    b1 -= b2;
    norm_diff = b1.Norml2();

    // Print the norm of the difference
    cout << "RHS difference norm: " << norm_diff << endl;

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