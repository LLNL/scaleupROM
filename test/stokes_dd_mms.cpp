#include<gtest/gtest.h>
#include "mfem.hpp"
#include "stokes_solver.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

static double nu;

void uFun_ex(const Vector & x, Vector & u);
void mlap_uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = 2.0 * nu * cos(xi)*sin(yi);
   u(1) = - 2.0 * nu * sin(xi)*cos(yi);
}
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);

StokesSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DDSerialTest, Test_convergence)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 3;
   CheckConvergence();

   return;
}

TEST(DDSerial_component_wise_test, Test_convergence)
{
   config = InputParser("inputs/dd_mms.component.yml");
   config.dict_["discretization"]["order"] = 1;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 3;
   CheckConvergence();

   return;
}

// TEST(DDSerial_component_3D_hex_test, Test_convergence)
// {
//    config = InputParser("inputs/dd_mms.comp.3d.yml");
//    CheckConvergence();

//    return;
// }

// TEST(DDSerial_component_3D_tet_test, Test_convergence)
// {
//    config = InputParser("inputs/dd_mms.comp.3d.yml");
//    config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "meshes/dd_mms.3d.tet.mesh";
//    CheckConvergence();

//    return;
// }

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}

void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * nu * sin(xi)*sin(yi);
}

void fFun(const Vector & x, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = 4.0 * nu * cos(xi) * sin(yi);
   f(1) = 0.0;
}

double gFun(const Vector & x)
{
   assert(x.Size() == 2);

   return 0.0;
}

StokesSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   StokesSolver *test = new StokesSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(uFun_ex);
   test->AddRHSFunction(fFun);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

test->SaveVisualization();

   return test;
}

void CheckConvergence()
{
   nu = config.GetOption<double>("stokes/nu", 1.0);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);

   printf("Num. Elem.\tRel. vel. Error\tConv Rate\tNorm\tRel. pres. Error\tConv Rate\tNorm\n");

   Vector uconv_rate(num_refine), pconv_rate(num_refine);
   uconv_rate = 0.0;
   pconv_rate = 0.0;
   double uerror1 = 0.0, perror1 = 0.0;
   for (int r = 0; r < num_refine; r++)
   {
      StokesSolver *test = SolveWithRefinement(r);

      // Compare with exact solution
      int dim = test->GetDim();
      VectorFunctionCoefficient exact_usol(dim, uFun_ex);
      FunctionCoefficient exact_psol(pFun_ex);

      // For all velocity dirichlet bc, pressure does not have the absolute value.
      // specify the constant scalar for the reference value.
      double p_const = 0.0;
      int ps = 0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         GridFunction p_ex(*pk);
         p_ex.ProjectCoefficient(exact_psol);
         ps += p_ex.Size();
         p_const += p_ex.Sum();
         // If p_ex is the view vector of pk, then this will prevent false negative test result.
         p_ex += 1.0;
      }
      p_const /= static_cast<double>(ps);

      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         (*pk) += p_const;
      }

      int uorder = test->GetVelFEOrder();
      int porder = test->GetPresFEOrder();
      int order_quad = max(2, 2*uorder+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double unorm = 0.0, pnorm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         unorm += pow(ComputeLpNorm(2.0, exact_usol, *mk, irs), 2);
         pnorm += pow(ComputeLpNorm(2.0, exact_psol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      unorm = sqrt(unorm);
      pnorm = sqrt(pnorm);

      double uerror = 0.0, perror = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetVelGridFunction(k);
         GridFunction *pk = test->GetPresGridFunction(k);
         uerror += pow(uk->ComputeLpError(2, exact_usol), 2);
         perror += pow(pk->ComputeLpError(2, exact_psol), 2);
      }
      uerror = sqrt(uerror);
      perror = sqrt(perror);

      uerror /= unorm;
      perror /= pnorm;
      
      if (r > 0)
      {
         uconv_rate(r) = uerror1 / uerror;
         pconv_rate(r) = perror1 / perror;
      }
      printf("%d\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n", numEl, uerror, uconv_rate(r), unorm, perror, pconv_rate(r), pnorm);

      // reported convergence rate
      if (r > 0)
      {
         EXPECT_TRUE(uconv_rate(r) > pow(2.0, uorder+1) - 0.1);
         EXPECT_TRUE(pconv_rate(r) > pow(2.0, porder+1) - 0.1);
      }

      uerror1 = uerror;
      perror1 = perror;

      delete test;
   }

   return;
}