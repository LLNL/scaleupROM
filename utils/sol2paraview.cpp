// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "main_workflow.hpp"
#include "multiblock_solver.hpp"

using namespace std;
using namespace mfem;

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
   // Do not need ROMHandler.
   config.dict_["main"]["use_rom"] = false;

   MultiBlockSolver *test = InitSolver();
   test->InitVariables();
   // Use MultiBlockSolver visualization setting. 
   test->InitVisualization();

   const std::string sol_file = config.GetRequiredOption<std::string>("sol2paraview/solution_file");
   test->LoadSolution(sol_file);

   test->SaveVisualization();
   
   delete test;
   MPI_Finalize();
}