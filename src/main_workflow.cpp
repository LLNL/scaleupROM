// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "main_workflow.hpp"
#include "component_topology_handler.hpp"
#include "multiblock_solver.hpp"
#include "poisson_solver.hpp"
#include "linelast_solver.hpp"
#include "stokes_solver.hpp"
#include "steady_ns_solver.hpp"
#include "etc.hpp"
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
   std::string solver_type = config.GetRequiredOption<std::string>("main/solver");
   MultiBlockSolver *solver = NULL;
   if (solver_type == "poisson")       { solver = new PoissonSolver; }
   else if (solver_type == "stokes")   { solver = new StokesSolver; }
   else if (solver_type == "steady-ns")   { solver = new SteadyNSSolver; }
   //else if (solver_type == "linelast")   { solver = new LinElastSolver; } // TODO: make this work
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

std::vector<std::string> GetGlobalBasisTagList(const TopologyHandlerMode &topol_mode, const TrainMode &train_mode, bool separate_variable_basis)
{
   std::vector<std::string> basis_tags(0);

   std::vector<std::string> component_list(0);
   if (train_mode == TrainMode::INDIVIDUAL)
   {
      TopologyHandler *topol_handler;
      if (topol_mode == TopologyHandlerMode::SUBMESH)          topol_handler = new SubMeshTopologyHandler();
      else if (topol_mode == TopologyHandlerMode::COMPONENT)   topol_handler = new ComponentTopologyHandler();
      else
         mfem_error("GetGlobalBasisTagList - TopologyHandlerMode is not set!\n");

      for (int c = 0; c < topol_handler->GetNumSubdomains(); c++)
         component_list.push_back("dom" + std::to_string(c));

      delete topol_handler;
   }
   else  // if (train_mode == TrainMode::UNIVERSAL)
   {
      if (topol_mode == TopologyHandlerMode::SUBMESH)
      {
         component_list.push_back("comp0");
      }
      else if (topol_mode == TopologyHandlerMode::COMPONENT)
      {
         YAML::Node component_dict = config.FindNode("mesh/component-wise/components");
         assert(component_dict);
         for (int p = 0; p < component_dict.size(); p++)
         component_list.push_back(config.GetRequiredOptionFromDict<std::string>("name", component_dict[p]));
      }
      else
         mfem_error("GetGlobalBasisTagList - TopologyHandlerMode is not set!\n");
   }  // if (train_mode == TrainMode::UNIVERSAL)

   std::vector<std::string> var_list(0);
   if (separate_variable_basis)
   {
      std::string solver_type = config.GetRequiredOption<std::string>("main/solver");
      if (solver_type == "poisson")          var_list = PoissonSolver::GetVariableNames();
      else if (solver_type == "stokes")      var_list = StokesSolver::GetVariableNames();
      else if (solver_type == "steady-ns")   var_list = SteadyNSSolver::GetVariableNames();
      else
      {
         printf("Unknown MultiBlockSolver %s!\n", solver_type.c_str());
         exit(-1);
      }
   }

   for (int c = 0; c < component_list.size(); c++)
   {
      if (separate_variable_basis)
         for (int v = 0; v < var_list.size(); v++)
            basis_tags.push_back(component_list[c] + "_" + var_list[v]);
      else
         basis_tags.push_back(component_list[c]);
   }

   return basis_tags;
}

void GenerateSamples(MPI_Comm comm)
{
   // save the original config.dict_
   YAML::Node dict0 = YAML::Clone(config.dict_);
   ParameterizedProblem *problem = InitParameterizedProblem();
   SampleGenerator *sample_generator = InitSampleGenerator(comm);
   SampleGeneratorType sample_gen_type = sample_generator->GetType();
   sample_generator->SetParamSpaceSizes();
   MultiBlockSolver *test = NULL;

   int s = 0;
   while (s < sample_generator->GetTotalSampleSize())
   {
      if (!sample_generator->IsMyJob(s)) continue;

      // NOTE: this will change config.dict_
      sample_generator->SetSampleParams(s);

      test = InitSolver();
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

      bool converged = test->Solve();
      if (!converged)
      {
         // If deterministic, terminate the sampling here.
         if (sample_gen_type == BASE)
            mfem_error("A sample solution fails to converge!\n");
         else if (sample_gen_type == RANDOM)
         {
            // if random, try another sample.
            mfem_warning("A sample solution failed to converge. Trying another sample.\n");
            delete test;
            continue;
         }
      }
         
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

      s++;
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

   TopologyHandlerMode topol_mode = SetTopologyHandlerMode();
   TrainMode train_mode = SetTrainMode();
   bool separate_variable_basis = config.GetOption<bool>("model_reduction/separate_variable_basis", false);

   // Find the all required basis tags.
   std::vector<std::string> basis_tags = GetGlobalBasisTagList(topol_mode, train_mode, separate_variable_basis);

   // tag-specific optional inputs.
   YAML::Node basis_list = config.FindNode("basis/tags");

   std::string basis_prefix = config.GetOption<std::string>("basis/prefix", "basis");
   const int num_basis_default = config.GetOption<int>("basis/number_of_basis", -1);

   // loop over the required basis tag list.
   for (int p = 0; p < basis_tags.size(); p++)
   {
      std::vector<std::string> file_list(0);
      int num_basis;

      // if optional inputs are specified, parse them first.
      if (basis_list)
      {
         // Find if additional inputs are specified for basis_tags[p].
         YAML::Node basis_tag_input = config.LookUpFromDict("name", basis_tags[p], basis_list);
         
         // If basis_tags[p] has additional inputs, parse them.
         if (basis_tag_input)
         {
            // parse tag-specific number of basis.
            num_basis = config.GetOptionFromDict<int>("number_of_basis", num_basis_default, basis_tag_input);

            // parse the sample snapshot file list.
            file_list = config.GetOptionFromDict<std::vector<std::string>>(
                        "snapshot_files", std::vector<std::string>(0), basis_tag_input);
            YAML::Node snapshot_format = config.FindNodeFromDict("snapshot_format", basis_tag_input);
            if (snapshot_format)
            {
               FilenameParam snapshot_param("", snapshot_format);
               snapshot_param.ParseFilenames(file_list);
            }
         }  // if (basis_tag_input)
         else
            num_basis = num_basis_default;
      }
      else
         // if additional inputs are not specified, use default number of basis.
         num_basis = num_basis_default;

      assert(num_basis > 0);

      // if additional inputs are not specified for snapshot files, set default snapshot file name.
      if (file_list.size() == 0)
      {
         std::string filename = sample_generator->GetBaseFilename(sample_generator->GetSamplePrefix(), basis_tags[p]);
         filename += "_snapshot";
         file_list.push_back(filename);
      }

      sample_generator->FormReducedBasis(basis_prefix, basis_tags[p], file_list, num_basis);
   }  // for (int p = 0; p < basis_tags.size(); p++)

   delete sample_generator;

   AuxiliaryTrainROM(comm);
}

void AuxiliaryTrainROM(MPI_Comm comm)
{
   std::string solver_type = config.GetRequiredOption<std::string>("main/solver");
   bool separate_variable_basis = config.GetOption<bool>("model_reduction/separate_variable_basis", false);

   /* Supremizer enrichment */
   if ((separate_variable_basis) && ((solver_type == "stokes") || (solver_type == "steady-ns")))
   {
      ParameterizedProblem *problem = InitParameterizedProblem();
      StokesSolver *solver = NULL;
      if (solver_type == "stokes")           { solver = new StokesSolver; }
      else if (solver_type == "steady-ns")   { solver = new SteadyNSSolver; }

      if (!solver->UseRom()) mfem_error("ROM must be enabled for supremizer enrichment!\n");

      solver->InitVariables();
      // This time needs to be ROMHandler, in order not to run StokesSolver::LoadSupremizer.
      solver->GetROMHandler()->LoadReducedBasis();

      solver->EnrichSupremizer();

      delete problem;
      delete solver;
   }

   // TODO: EQP weight optimization procedure.
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
   
   test->LoadReducedBasis();
   
   TopologyHandlerMode topol_mode = test->GetTopologyMode();
   ROMBuildingLevel save_operator = test->GetROMHandler()->GetBuildingLevel();

   // NOTE(kevin): global operator required only for global rom operator.
   if (save_operator == ROMBuildingLevel::GLOBAL)
      test->Assemble();

   switch (save_operator)
   {
      case ROMBuildingLevel::COMPONENT:
      {
         if (topol_mode == TopologyHandlerMode::SUBMESH)
            mfem_error("Submesh does not support component rom building level!\n");

         test->AllocateROMElements();
         test->BuildROMElements();
         std::string filename = test->GetROMHandler()->GetOperatorPrefix() + ".h5";
         test->SaveROMElements(filename);
         break;
      }
      case ROMBuildingLevel::GLOBAL:
      {
         test->ProjectOperatorOnReducedBasis();
         test->SaveROMOperator();
         break;
      }
      case ROMBuildingLevel::NONE:
      {
         printf("BuildROM - ROM building level is set to none. No ROM is saved.\n");
         break;
      }
   }  // switch (save_operator)

   test->SaveBasisVisualization();

   delete test;
   delete problem;
}

double SingleRun(MPI_Comm comm, const std::string output_file)
{
   if (config.GetOption<bool>("single_run/choose_from_random_sample", false))
   {
      RandomSampleGenerator *generator = new RandomSampleGenerator(comm);
      generator->SetParamSpaceSizes();
      int idx = UniformRandom(0, generator->GetTotalSampleSize()-1);
      // NOTE: this will change config.dict_
      generator->SetSampleParams(idx);
      delete generator;
   }

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

   const int num_var = test->GetNumVar();
   Vector rom_assemble(1), rom_solve(1), fom_assemble(1), fom_solve(1), error(num_var);
   rom_assemble = -1.0; rom_solve = -1.0;
   fom_assemble = -1.0; fom_solve = -1.0;
   error = -1.0;

   ROMHandlerBase *rom = NULL;
   if (test->UseRom())
   {
      rom = test->GetROMHandler();
      test->LoadReducedBasis();
   }

   solveTimer.Start();
   if (test->UseRom())
   {
      printf("ROM with ");
      ROMBuildingLevel save_operator = rom->GetBuildingLevel();
      TopologyHandlerMode topol_mode = test->GetTopologyMode();

      if (topol_mode == TopologyHandlerMode::SUBMESH)
         printf("using SubMesh topology.\n");
      else if (topol_mode == TopologyHandlerMode::COMPONENT)
         printf("using Component-wise topology.\n");
      else
         mfem_error("Unknown TopologyHandler Mode!\n");

      switch (save_operator)
      {
         case ROMBuildingLevel::COMPONENT:
         {
            if (topol_mode == TopologyHandlerMode::SUBMESH)
               mfem_error("Submesh does not support component rom building level!\n");

            printf("Loading component operator file.. ");
            test->AllocateROMElements();
            std::string filename = rom->GetOperatorPrefix() + ".h5";
            test->LoadROMElements(filename);
            test->AssembleROM();
            break;
         }
         case ROMBuildingLevel::GLOBAL:
         {
            printf("Loading global operator file.. ");
            test->LoadROMOperatorFromFile();
            break;
         }
         case ROMBuildingLevel::NONE:
         {
            printf("Building operator file all the way from FOM.. ");
            test->BuildDomainOperators();
            test->SetupDomainBCOperators();
            test->AssembleOperator();
            test->ProjectOperatorOnReducedBasis();
            break;
         }
      }
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

   if (test->UseRom())
      rom_assemble = solveTimer.RealTime();
   else
      fom_assemble = solveTimer.RealTime();

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

   if (test->UseRom())
      rom_solve = solveTimer.RealTime();
   else
      fom_solve = solveTimer.RealTime();

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

      test->CompareSolution(*romU, error);

      bool save_reduced_sol = config.GetOption<bool>("model_reduction/compare_solution/save_reduced_solution", false);
      if (save_reduced_sol)
      {
         ROMHandlerBase *rom = test->GetROMHandler();
         rom->SaveReducedSolution("rom_reduced_sol.txt");

         // use ROMHandler::reduced_rhs as a temporary variable.
         rom->ProjectRHSOnReducedBasis(test->GetSolution());
         rom->SaveReducedRHS("fom_reduced_sol.txt");
      }

      // Recover the original ROM solution.
      test->CopySolution(romU);

      delete romU;
   }

   // save results to output file.
   if (output_file.length() > 0)
   {
      hid_t file_id;
      herr_t errf = 0;
      file_id = H5Fcreate(output_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert(file_id >= 0);

      hdf5_utils::WriteDataset(file_id, "rom_assemble", rom_assemble);
      hdf5_utils::WriteDataset(file_id, "rom_solve", rom_solve);
      hdf5_utils::WriteDataset(file_id, "fom_assemble", fom_assemble);
      hdf5_utils::WriteDataset(file_id, "fom_solve", fom_solve);
      hdf5_utils::WriteDataset(file_id, "rel_error", error);

      errf = H5Fclose(file_id);
      assert(errf >= 0);
   }

   // Save solution and visualization.
   test->SaveSolution();
   test->SaveVisualization();
   
   delete test;
   delete problem;

   // return the maximum error over all variables.
   return error.Max();
}

