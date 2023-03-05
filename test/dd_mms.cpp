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

int main(int argc, char *argv[])
{
   const char *input_file = "test.input";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.ParseCheck();
   config = InputParser(input_file);

   amp1 = config.GetOption<double>("manufactured_solution/amp1", 0.22);
   amp2 = config.GetOption<double>("manufactured_solution/amp2", 0.13);
   L1 = config.GetOption<double>("manufactured_solution/L1", 0.31);
   L2 = config.GetOption<double>("manufactured_solution/L2", 0.72);
   offset1 = config.GetOption<double>("manufactured_solution/offset1", 0.22);
   offset2 = config.GetOption<double>("manufactured_solution/offset2", 0.13);
   constant = config.GetOption<double>("manufactured_solution/constant", -0.27);

   MultiBlockSolver test;

   test.InitVariables();
   test.InitVisualization();

   test.AddBCFunction(ExactSolution);
   test.AddRHSFunction(ExactRHS);

   test.BuildOperators();

   test.SetupBCOperators();

   test.Assemble();

   test.Solve();
   // test.SaveVisualization();
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
