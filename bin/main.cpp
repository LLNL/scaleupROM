#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include "multiblock_solver.hpp"
#include "parameterized_problem.hpp"
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
void GenerateSamples(MPI_Comm comm);
void BuildROM(MPI_Comm comm);

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
   // TODO: change this to a single full-order simulation.
   if (mode == "run_example")
   {
      // TODO: make it parallel-run compatible.
      if (rank == 0) RunExample();
   }
   else if (mode == "sample_generation")
   {
      GenerateSamples(MPI_COMM_WORLD);
   }
   else if (mode == "build_rom")
   {
      BuildROM(MPI_COMM_WORLD);
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

void GenerateSamples(MPI_Comm comm)
{
   std::string problem_name = config.GetRequiredOption<std::string>("parameterized_problem/name");

   ParameterizedProblem *problem = NULL;
   MultiBlockSolver *test = NULL;

   if (problem_name == "poisson0")
   {
      problem = new Poisson0(comm);
   }
   else
   {
      mfem_error("Unknown parameterized problem!\n");
   }

   for (int s = 0; s < problem->GetTotalSampleSize(); s++)
   {
      if (!problem->IsMyJob(s)) continue;

      test = new MultiBlockSolver();
      test->InitVariables();

      problem->SetParams(s);
      test->SetParameterizedProblem(problem);

      test->InitVisualization();
      test->BuildOperators();
      test->SetupBCOperators();
      test->Assemble();
      test->Solve();
      test->SaveVisualization();

      test->SaveSnapshot(s);

      delete test;
   }

   delete problem;
}

void BuildROM(MPI_Comm comm)
{
   std::string problem_name = config.GetRequiredOption<std::string>("parameterized_problem/name");

   ParameterizedProblem *problem = NULL;
   MultiBlockSolver *test = NULL;

   if (problem_name == "poisson0")
   {
      problem = new Poisson0(comm);
   }
   else
   {
      mfem_error("Unknown parameterized problem!\n");
   }

   const int total_samples = problem->GetTotalSampleSize();

   test = new MultiBlockSolver();
   test->InitVariables();
   test->BuildOperators();
   test->SetupBCOperators();
   test->Assemble();
   
   test->FormReducedBasis(total_samples);
   // test->InitVariables();

   delete test;
   delete problem;
}