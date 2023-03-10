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

void RunExample();
void GenerateSamples();

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const char *input_file = "test.input";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.ParseCheck();
   config = InputParser(input_file);
   
   std::string mode = config.GetOption<std::string>("main/mode", "run_example");
   // TODO: change this to run a single full-order simulation.
   if (mode == "run_example")
   {
      // TODO: make it parallel-run compatible.
      if (rank == 0) RunExample();
   }
   else if (mode == "sample_generation")
   {
      GenerateSamples();
   }
   else
   {
      if (rank == 0) printf("Unknown mode %s!\n", mode.c_str());
      exit(-1);
   }

}

double dbc2(const Vector &x)
{
   return 0.1 - 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

double dbc4(const Vector &x)
{
   return -0.1 + 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

void RunExample()
{
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

void GenerateSamples()
{

}