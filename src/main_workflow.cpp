// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "main_workflow.hpp"
#include "multiblock_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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

SampleGenerator* InitSampleGenerator(MPI_Comm comm, ParameterizedProblem* problem)
{
   SampleGenerator* generator = NULL;

   std::string type = config.GetOption<std::string>("sample_generation/type", "base");

   if (type == "base")
   {
      generator = new SampleGenerator(comm, problem);
   }
   else if (type == "random")
   {
      generator = new RandomSampleGenerator(comm, problem);
   }
   else
   {
      mfem_error("Unknown sample generator type!\n");
   }

   return generator;
}

void GenerateSamples(MPI_Comm comm)
{
   ParameterizedProblem *problem = InitParameterizedProblem();
   SampleGenerator *sample_generator = InitSampleGenerator(comm, problem);
   sample_generator->GenerateParamSpace();
   MultiBlockSolver *test = NULL;

   for (int s = 0; s < sample_generator->GetTotalSampleSize(); s++)
   {
      if (!sample_generator->IsMyJob(s)) continue;

      test = new MultiBlockSolver();
      if (!test->UseRom()) mfem_error("ROM must be enabled for sample generation!\n");
      test->InitVariables();

      sample_generator->SetSampleParams(s);
      problem->SetParameterizedProblem(test);
printf("%.5E\n", function_factory::poisson0::k);
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
   SampleGenerator *sample_generator = InitSampleGenerator(comm, problem);
   MultiBlockSolver *test = NULL;

   sample_generator->SetParamSpaceSizes();
   const int total_samples = sample_generator->GetTotalSampleSize();

   test = new MultiBlockSolver();
   if (!test->UseRom()) mfem_error("ROM must be enabled for BuildROM!\n");
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

   test->SaveBasisVisualization();

   delete test;
   delete sample_generator;
   delete problem;
}

double SingleRun()
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
printf("%.5E\n", function_factory::poisson0::k);
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

   double error = -1.0;
   bool compare_sol = config.GetOption<bool>("model_reduction/compare_solution", false);
   if (test->UseRom() && compare_sol)
      error = test->CompareSolution();

   delete test;
   delete problem;

   return error;
}

