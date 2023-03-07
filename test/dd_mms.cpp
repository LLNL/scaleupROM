#include<gtest/gtest.h>
#include "mfem.hpp"
#include "multiblock_solver.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

static const double pi = 4.0 * atan(1.0);
static double amp1, amp2;
static double L1, L2;
static double offset1, offset2;
static double constant;

double ExactSolution(const Vector &);
double ExactRHS(const Vector &);
MultiBlockSolver *SolveWithRefinement(const int num_refinement);

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
    SUCCEED();
}

// int main(int argc, char *argv[])
TEST(DDSerialTest, Test_convergence)
{
   // const char *input_file = "test.input";
   // OptionsParser args(argc, argv);
   // args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   // args.ParseCheck();
   config = InputParser("inputs/dd_mms.yml");

   amp1 = config.GetOption<double>("manufactured_solution/amp1", 0.22);
   amp2 = config.GetOption<double>("manufactured_solution/amp2", 0.13);
   L1 = config.GetOption<double>("manufactured_solution/L1", 0.31);
   L2 = config.GetOption<double>("manufactured_solution/L2", 0.72);
   offset1 = config.GetOption<double>("manufactured_solution/offset1", 0.35);
   offset2 = config.GetOption<double>("manufactured_solution/offset2", 0.73);
   constant = config.GetOption<double>("manufactured_solution/constant", -0.27);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);

   // Compare with exact solution
   FunctionCoefficient exact_sol(ExactSolution);

   printf("Num. Elem.\tRelative Error\tConvergence Rate\tNorm\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = 0; r < num_refine; r++)
   {
      MultiBlockSolver *test = SolveWithRefinement(r);

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
         norm += ComputeLpNorm(2.0, exact_sol, *mk, irs);
         numEl += mk->GetNE();
      }

      double error = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetGridFunction(k);
         error += uk->ComputeLpError(2, exact_sol);
      }
      error /= norm;
      
      if (r > 0)
      {
         conv_rate(r) = error1 / error;
      }
      printf("%d\t%.15E\t%.15E\t%.15E\n", numEl, error, conv_rate(r), norm);

      // reported convergence rate
      if (r > 0)
         if (conv_rate(r) < pow(2.0, order+1) - 0.5)
         {
            printf("Convergence rate below threshold: %.3f!\n", conv_rate(r));
            exit(-1);
         }

      error1 = error;
   }

   return;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

double ExactSolution(const Vector &x)
{
   return constant + amp1 * sin(2.0 * pi / L1 * (x(0) - offset1))
                   + amp2 * cos(2.0 * pi / L2 * (x(1) - offset2));
}

double ExactRHS(const Vector &x)
{
   return amp1 * (2.0 * pi / L1) * (2.0 * pi / L1) * sin(2.0 * pi / L1 * (x(0) - offset1))
            + amp2 * (2.0 * pi / L2) * (2.0 * pi / L2) * cos(2.0 * pi / L2 * (x(1) - offset2));
}

MultiBlockSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   MultiBlockSolver *test = new MultiBlockSolver();

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