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
#include "poisson_solver.hpp"
#include "stokes_solver.hpp"
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
   PoissonSolver test;

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

MultiBlockSolver* InitSolver()
{
   std::string solver_type = config.GetOption<std::string>("main/solver", "poisson");
   MultiBlockSolver *solver = NULL;
   if (solver_type == "poisson")       { solver = new PoissonSolver; }
   else if (solver_type == "stokes")   { solver = new StokesSolver; }
   else
   {
      printf("Unknown MultiBlockSolver %s!\n", solver_type.c_str());
      exit(-1);
   }

   return solver;
}

SampleGenerator* InitSampleGenerator(MPI_Comm comm)
{
   SampleGenerator* generator = NULL;

   std::string type = config.GetOption<std::string>("sample_generation/type", "base");

   if (type == "base")
   {
      generator = new SampleGenerator(comm);
   }
   else if (type == "random")
   {
      generator = new RandomSampleGenerator(comm);
   }
   else
   {
      mfem_error("Unknown sample generator type!\n");
   }

   return generator;
}

void GenerateSamples(MPI_Comm comm)
{
   // save the original config.dict_
   YAML::Node dict0 = YAML::Clone(config.dict_);
   ParameterizedProblem *problem = InitParameterizedProblem();
   SampleGenerator *sample_generator = InitSampleGenerator(comm);
   sample_generator->SetParamSpaceSizes();
   MultiBlockSolver *test = NULL;

   for (int s = 0; s < sample_generator->GetTotalSampleSize(); s++)
   {
      if (!sample_generator->IsMyJob(s)) continue;

      // NOTE: this will change config.dict_
      sample_generator->SetSampleParams(s);

      test = InitSolver();
      if (!test->UseRom()) mfem_error("ROM must be enabled for sample generation!\n");
      test->InitVariables();

      problem->SetSingleRun();
      test->SetParameterizedProblem(problem);

      int file_idx = s + sample_generator->GetFileOffset();
      const std::string visual_path = sample_generator->GetSamplePath(file_idx, test->GetVisualizationPrefix());
      std::string sol_file = sample_generator->GetSamplePath(file_idx, test->GetSolutionFilePrefix());
      sol_file += ".h5";
      test->InitVisualization(visual_path);
      test->BuildOperators();
      test->SetupBCOperators();
      test->Assemble();
      test->Solve();
      test->SaveSolution(sol_file);
      test->SaveVisualization();

      // test->SaveSnapshot(file_idx);
      // View-Vector of the entire solution of test, splitted according to ROM basis setup.
      BlockVector *U_snapshots = NULL;
      // Basis tags for each block of U_snapshots.
      std::vector<std::string> basis_tags;
      test->PrepareSnapshots(U_snapshots, basis_tags);
      sample_generator->SaveSnapshot(U_snapshots, basis_tags);

      sample_generator->ReportStatus(s);

      delete U_snapshots;
      delete test;
   }
   sample_generator->WriteSnapshots();

   delete sample_generator;
   delete problem;
   // restore the original config.dict_
   config.dict_ = dict0;
}

void TrainROM(MPI_Comm comm)
{
   SampleGenerator *sample_generator = InitSampleGenerator(comm);

   YAML::Node basis_list = config.FindNode("basis/tags");
   if (!basis_list) mfem_error("TrainROM - cannot find the basis tag list!\n");

   std::string basis_prefix = config.GetOption<std::string>("basis/prefix", "basis");
   const int num_basis_default = config.GetOption<int>("basis/number_of_basis", -1);

   for (int p = 0; p < basis_list.size(); p++)
   {
      std::string basis_tag = config.GetRequiredOptionFromDict<std::string>("name", basis_list[p]);
      const int num_basis = config.GetOptionFromDict<int>("number_of_basis", num_basis_default, basis_list[p]);
      assert(num_basis > 0);

      std::vector<std::string> file_list =
         config.GetOptionFromDict<std::vector<std::string>>(
            "snapshot_files", std::vector<std::string>(0), basis_list[p]);
      if (file_list.size() == 0)
      {
         std::string filename = sample_generator->GetBaseFilename(sample_generator->GetSamplePrefix(), basis_tag);
         filename += "_snapshot";
         file_list.push_back(filename);
      }

      sample_generator->FormReducedBasis(basis_prefix, basis_tag, file_list, num_basis);
   }  // for (int p = 0; p < basis_list.size(); p++)

   delete sample_generator;

   // MultiBlockSolver *test = NULL;

   // test = InitSolver();
   // if (!test->UseRom()) mfem_error("ROM must be enabled for BuildROM!\n");
   // test->InitVariables();
   
   // test->FormReducedBasis();

   // delete test;
}

void BuildROM(MPI_Comm comm)
{
   ParameterizedProblem *problem = InitParameterizedProblem();
   MultiBlockSolver *test = NULL;

   test = InitSolver();
   if (!test->UseRom()) mfem_error("ROM must be enabled for BuildROM!\n");
   test->InitVariables();
   // test->InitVisualization();

   // The ROM operator will be built based on the parameter specified for single-run.
   problem->SetSingleRun();
   test->SetParameterizedProblem(problem);

   // TODO: there are skippable operations depending on rom/fom mode.
   test->BuildOperators();
   test->SetupBCOperators();
   test->Assemble();
   
   ROMHandler *rom = test->GetROMHandler();
   rom->LoadReducedBasis();
   
   TopologyHandlerMode topol_mode = test->GetTopologyMode();
   ROMBuildingLevel save_operator = rom->SaveOperator();
   switch (topol_mode)
   {
      case TopologyHandlerMode::SUBMESH:
      {
         if (save_operator == ROMBuildingLevel::GLOBAL)
            test->ProjectOperatorOnReducedBasis();
         else if (save_operator == ROMBuildingLevel::COMPONENT)
            mfem_error("Unsupported rom building level!\n");
         break;
      }  // case TopologyHandlerMode::SUBMESH:
      case TopologyHandlerMode::COMPONENT:
      {
         switch (save_operator)
         {
            case ROMBuildingLevel::COMPONENT:
            {
               test->AllocateROMElements();
               test->BuildROMElements();
               std::string filename = rom->GetOperatorPrefix() + ".h5";
               test->SaveROMElements(filename);
               break;
            }
            case ROMBuildingLevel::GLOBAL:
            {
               test->ProjectOperatorOnReducedBasis();
               break;
            }
         }  // switch (save_operator)
         break;
      }  // case TopologyHandlerMode::COMPONENT:
      default:
      {
         mfem_error("Unknown TopologyHandler Mode!\n");
         break;
      }
   }  // switch (topol_mode)

   test->SaveBasisVisualization();

   delete test;
   delete problem;
}

double SingleRun(const std::string output_file)
{
   ParameterizedProblem *problem = InitParameterizedProblem();
   MultiBlockSolver *test = InitSolver();
   test->InitVariables();
   test->InitVisualization();

   StopWatch solveTimer;
   std::string solveType = (test->UseRom()) ? "ROM" : "FOM";

   problem->SetSingleRun();
   test->SetParameterizedProblem(problem);

   // TODO: there are skippable operations depending on rom/fom mode.
   test->BuildRHSOperators();
   test->SetupRHSBCOperators();
   test->AssembleRHS();

   double rom_assemble = -1.0, rom_solve = -1.0;
   double fom_assemble = -1.0, fom_solve = -1.0;

   ROMHandler *rom = NULL;
   if (test->UseRom())
   {
      rom = test->GetROMHandler();
      rom->LoadReducedBasis();
   }

   solveTimer.Start();
   if (test->UseRom())
   {
      printf("ROM with ");
      ROMBuildingLevel save_operator = rom->SaveOperator();
      TopologyHandlerMode topol_mode = test->GetTopologyMode();
      switch (topol_mode)
      {
         case TopologyHandlerMode::SUBMESH:
         {
            printf("SubMesh Topology - ");
            switch (save_operator)
            {
               case ROMBuildingLevel::GLOBAL:
               {
                  printf("loading operator file.. ");
                  rom->LoadOperatorFromFile();
                  break;
               }
               case ROMBuildingLevel::NONE:
               {
                  printf("building operator file all the way from FOM.. ");
                  test->BuildDomainOperators();
                  test->SetupDomainBCOperators();
                  test->AssembleOperator();
                  test->ProjectOperatorOnReducedBasis();
                  break;
               }
               default:
               {
                  mfem_error("Unsupported rom building level!\n");
                  break;
               }
            }
            break;
         }  // case TopologyHandlerMode::SUBMESH:
         case TopologyHandlerMode::COMPONENT:
         {
            printf("Component-wise Topology - ");
            // TODO: bottom-up assembly.
            switch (save_operator)
            {
               case ROMBuildingLevel::COMPONENT:
               {
                  printf("loading component operator file.. ");
                  test->AllocateROMElements();
                  std::string filename = rom->GetOperatorPrefix() + ".h5";
                  test->LoadROMElements(filename);
                  test->AssembleROM();
                  break;
               }
               case ROMBuildingLevel::GLOBAL:
               {
                  printf("loading global operator file.. ");
                  rom->LoadOperatorFromFile();
                  break;
               }
               case ROMBuildingLevel::NONE:
               {
                  printf("building operator file all the way from FOM.. ");
                  test->BuildDomainOperators();
                  test->SetupDomainBCOperators();
                  test->AssembleOperator();
                  test->ProjectOperatorOnReducedBasis();
                  break;
               }
            }
            break;
         }  // case TopologyHandlerMode::COMPONENT:
         default:
         {
            mfem_error("Unknown TopologyHandler Mode!\n");
            break;
         }
      }  // switch (topol_mode)
      printf("Done!\n");

      printf("Projecting RHS to ROM.. ");
      test->ProjectRHSOnReducedBasis();
      printf("Done!\n");
   }  // if (test->UseRom())
   else
   {
      test->BuildDomainOperators();
      test->SetupDomainBCOperators();
      test->AssembleOperator();
   }  // not if (test->UseRom())
   solveTimer.Stop();
   printf("%s-assemble time: %f seconds.\n", solveType.c_str(), solveTimer.RealTime());

   if (test->UseRom()) rom_assemble = solveTimer.RealTime();

   solveTimer.Clear();
   solveTimer.Start();
   if (test->UseRom())
   {
      test->SolveROM();
   }
   else
   {
      // TODO: move matrix assembly to here.
      test->Solve();
   }
   solveTimer.Stop();
   printf("%s-solve time: %f seconds.\n", solveType.c_str(), solveTimer.RealTime());

   if (test->UseRom()) rom_solve = solveTimer.RealTime();

   double error = -1.0;
   bool compare_sol = config.GetOption<bool>("model_reduction/compare_solution/enabled", false);
   bool load_sol = config.GetOption<bool>("model_reduction/compare_solution/load_solution", false);
   if (test->UseRom() && compare_sol)
   {
      BlockVector *romU = test->GetSolutionCopy();

      if (load_sol)
      {
         printf("Comparing with the existing FOM solution.\n");
         std::string fom_file = config.GetRequiredOption<std::string>("model_reduction/compare_solution/fom_solution_file");
         test->LoadSolution(fom_file);
      }
      else
      {
         solveTimer.Clear();
         solveTimer.Start();
         test->BuildDomainOperators();
         test->SetupDomainBCOperators();
         test->AssembleOperator();
         solveTimer.Stop();
         printf("FOM-assembly time: %f seconds.\n", solveTimer.RealTime());
         fom_assemble = solveTimer.RealTime();

         solveTimer.Clear();
         solveTimer.Start();
         test->Solve();
         solveTimer.Stop();
         printf("FOM-solve time: %f seconds.\n", solveTimer.RealTime());
         fom_solve = solveTimer.RealTime();
      }

      error = test->CompareSolution(*romU);

      if (output_file.length() > 0)
      {
         hid_t file_id;
         herr_t errf = 0;
         file_id = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
         assert(file_id >= 0);

         hdf5_utils::WriteAttribute(file_id, "rom_assemble", rom_assemble);
         hdf5_utils::WriteAttribute(file_id, "rom_solve", rom_solve);
         hdf5_utils::WriteAttribute(file_id, "fom_assemble", fom_assemble);
         hdf5_utils::WriteAttribute(file_id, "fom_solve", fom_solve);
         hdf5_utils::WriteAttribute(file_id, "rel_error", error);

         errf = H5Fclose(file_id);
         assert(errf >= 0);
      }

      bool save_reduced_sol = config.GetOption<bool>("model_reduction/compare_solution/save_reduced_solution", false);
      if (save_reduced_sol)
      {
         ROMHandler *rom = test->GetROMHandler();
         rom->SaveReducedSolution("rom_reduced_sol.txt");

         // use ROMHandler::reduced_rhs as a temporary variable.
         rom->ProjectRHSOnReducedBasis(test->GetSolution());
         rom->SaveReducedRHS("fom_reduced_sol.txt");
      }

      // Recover the original ROM solution.
      test->CopySolution(romU);

      delete romU;
   }

   // Save solution and visualization.
   test->SaveSolution();
   test->SaveVisualization();
   
   delete test;
   delete problem;

   return error;
}

