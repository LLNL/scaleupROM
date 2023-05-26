#include<gtest/gtest.h>
#include "mfem.hpp"
#include "poisson_solver.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

static const double pi = 4.0 * atan(1.0);
static double amp[3];
static double L[3];
static double offset[3];
static double constant;

double ExactSolution(const Vector &);
double ExactRHS(const Vector &);
PoissonSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DDSerialTest, Test_convergence)
{
   config = InputParser("test.component.yml");
   CheckConvergence();

   return;
}

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}

double ExactSolution(const Vector &x)
{
   double result = constant;
   for (int d = 0; d < x.Size(); d++)
      result += amp[d] * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
   return result;
}

double ExactRHS(const Vector &x)
{
   double result = 0.0;
   for (int d = 0; d < x.Size(); d++)
      result += amp[d] * (2.0 * pi / L[d]) * (2.0 * pi / L[d]) * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
   return result;
}

PoissonSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   PoissonSolver *test = new PoissonSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(ExactSolution);
   test->AddRHSFunction(ExactRHS);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

   return test;
}

void CheckConvergence()
{
   amp[0] = config.GetOption<double>("manufactured_solution/amp1", 0.22);
   amp[1] = config.GetOption<double>("manufactured_solution/amp2", 0.13);
   amp[2] = config.GetOption<double>("manufactured_solution/amp3", 0.37);
   L[0] = config.GetOption<double>("manufactured_solution/L1", 0.31);
   L[1] = config.GetOption<double>("manufactured_solution/L2", 0.72);
   L[2] = config.GetOption<double>("manufactured_solution/L2", 0.47);
   offset[0] = config.GetOption<double>("manufactured_solution/offset1", 0.35);
   offset[1] = config.GetOption<double>("manufactured_solution/offset2", 0.73);
   offset[2] = config.GetOption<double>("manufactured_solution/offset3", 0.59);
   constant = config.GetOption<double>("manufactured_solution/constant", -0.27);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   // Compare with exact solution
   FunctionCoefficient exact_sol(ExactSolution);

   printf("Num. Elem.\tRelative Error\tConvergence Rate\tNorm\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      PoissonSolver *test = SolveWithRefinement(r);

      int order = test->GetDiscretizationOrder();
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double norm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         norm += pow(ComputeLpNorm(2.0, exact_sol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      norm = sqrt(norm);

      double error = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetGridFunction(k);
         error += pow(uk->ComputeLpError(2, exact_sol), 2);
      }
      error = sqrt(error);
      error /= norm;
      
      if (r > base_refine)
      {
         conv_rate(r) = error1 / error;
      }
      printf("%d\t%.15E\t%.15E\t%.15E\n", numEl, error, conv_rate(r), norm);

      // reported convergence rate
      if (r > base_refine)
         EXPECT_TRUE(conv_rate(r) > pow(2.0, order+1) - 0.5);

      error1 = error;
   }

   return;
}