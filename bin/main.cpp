#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include "multiblock_solver.hpp"
#include "parameterized_problem.hpp"
#include "sample_generator.hpp"
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
void SingleRun();

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
   if (mode == "run_example")
   {
      if (rank == 0) RunExample();
   }
   else if (mode == "sample_generation")
   {
      GenerateSamples(MPI_COMM_WORLD);
   }
   else if (mode == "build_rom")
   {
      // TODO: need some refactoring to fully separate from single run.
      BuildROM(MPI_COMM_WORLD);
   }
   else if (mode == "single_run")
   {
      // TODO: make it parallel-run compatible.
      SingleRun();
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
   ParameterizedProblem *problem = InitParameterizedProblem();
   SampleGenerator *sample_generator = new SampleGenerator(comm, problem);
   sample_generator->GenerateParamSpace();
   MultiBlockSolver *test = NULL;

   for (int s = 0; s < sample_generator->GetTotalSampleSize(); s++)
   {
      if (!sample_generator->IsMyJob(s)) continue;

      test = new MultiBlockSolver();
      test->InitVariables();

      sample_generator->SetSampleParams(s);
      problem->SetParameterizedProblem(test);

      const std::string visual_path = sample_generator->GetSamplePath(s, test->GetVisualizationPrefix());
      test->InitVisualization(visual_path);
      test->BuildOperators();
      test->SetupBCOperators();
      test->Assemble();
      test->Solve();
      test->SaveVisualization();

      test->SaveSnapshot(s);

      delete test;
   }

   delete sample_generator;
   delete problem;
}

void BuildROM(MPI_Comm comm)
{
   ParameterizedProblem *problem = InitParameterizedProblem();
   SampleGenerator *sample_generator = new SampleGenerator(comm, problem);
   MultiBlockSolver *test = NULL;

   sample_generator->SetParamSpaceSizes();
   const int total_samples = sample_generator->GetTotalSampleSize();

   test = new MultiBlockSolver();
   test->InitVariables();
   // test->InitVisualization();

   // NOTE: you need this to set bc/rhs coefficients!
   // This case, we can use default parameter values of the problem.
   problem->SetParameterizedProblem(test);

   // TODO: there are skippable operations depending on rom/fom mode.
   test->BuildOperators();
   test->SetupBCOperators();
   test->Assemble();
   
   test->FormReducedBasis(total_samples);
   // test->LoadReducedBasis();
   // TODO: need to be able to save operator matrix.
   test->ProjectOperatorOnReducedBasis();

   // // TODO: separate unto single run mode.
   // test->ProjectRHSOnReducedBasis();
   // test->SolveROM();

   // test->SaveVisualization();

   delete test;
   delete sample_generator;
   delete problem;
}

void SingleRun()
{
   ParameterizedProblem *problem = InitParameterizedProblem();
   MultiBlockSolver *test = new MultiBlockSolver();
   test->InitVariables();
   test->InitVisualization();

   std::string problem_name = problem->GetProblemName();
   std::string param_list_str("single_run/" + problem_name);
   YAML::Node param_list = config.FindNode(param_list_str);
   MFEM_ASSERT(param_list, "Single Run - cannot find the problem name!\n");

   size_t num_params = param_list.size();
   for (int p = 0; p < num_params; p++)
   {
      std::string param_name = config.GetRequiredOptionFromDict<std::string>("parameter_name", param_list[p]);
      double value = config.GetRequiredOptionFromDict<double>("value", param_list[p]);
      problem->SetParams(param_name, value);
   }

   problem->SetParameterizedProblem(test);

   // TODO: there are skippable operations depending on rom/fom mode.
   test->BuildOperators();
   test->SetupBCOperators();
   test->Assemble();

   if (test->UseRom())
   {
      // test->AllocROMMat();
      test->LoadReducedBasis();

      // TODO: need to implement save/load sparse matrix and remove this line.
      std::string rom_handler_str = config.GetOption<std::string>("model_reduction/rom_handler_type", "base");
      if (rom_handler_str == "mfem")
         test->ProjectOperatorOnReducedBasis();
   }

   StopWatch solveTimer;
   solveTimer.Start();
   if (test->UseRom())
   {
      test->ProjectRHSOnReducedBasis();
      test->SolveROM();
   }
   else
   {
      test->Solve();
   }
   solveTimer.Stop();
   std::string solveType = (test->UseRom()) ? "ROM" : "FOM";
   printf("%s-solve time: %f seconds.\n", solveType.c_str(), solveTimer.RealTime());

   test->SaveVisualization();

   bool compare_sol = config.GetOption<bool>("model_reduction/compare_solution", false);
   if (test->UseRom() && compare_sol)
      test->CompareSolution();

   delete test;
   delete problem;
}