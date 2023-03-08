#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include "multiblock_solver.hpp"
#include "input_parser.hpp"
#include <fstream>
#include <iostream>
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "mfem/Utilities.hpp"

using namespace std;
using namespace mfem;

double dbc2(const Vector &);
double dbc4(const Vector &);

int main(int argc, char *argv[])
{
   const char *input_file = "test.input";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.ParseCheck();
   config = InputParser(input_file);

   // MultiBlockSolver test(argc, argv);
   MultiBlockSolver test;

   test.InitVariables();
   test.InitVisualization();

   test.AddBCFunction(dbc2, 2);
   test.AddBCFunction(dbc4, 4);
   test.AddRHSFunction(1.0);

   test.BuildOperators();

   test.SetupBCOperators();

   test.Assemble();

   test.Solve();
   test.SaveVisualization();

}

double dbc2(const Vector &x)
{
   return 0.1 - 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

double dbc4(const Vector &x)
{
   return -0.1 + 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

