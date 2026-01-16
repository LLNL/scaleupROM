// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "steady_ns_solver.hpp"
#include "component_topology_handler.hpp"
#include "hyperreduction_integ.hpp"
#include "nonlinear_integ.hpp"
// #include "input_parser.hpp"
// #include "hdf5_utils.hpp"
// #include "linalg_utils.hpp"
// #include "dg_bilinear.hpp"
#include "dg_linear.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

void SteadyNSSolver::SchwarzROM(const int M, const int N, ParameterizedProblem *problem)
{
   assert(use_rom);
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(N * N == numSub);
   ComponentTopologyHandler *comp_topol = static_cast<ComponentTopologyHandler *>(topol_handler);

   int Ns = N - M + 1;
   Array<ComponentTopologyHandler *> sub_topols(Ns * Ns); // will be owned by sub MultiBlockSolvers defined subsequently.
   Array<SteadyNSSolver *> sub_solvers(Ns * Ns);
   Array<Array<int> *> subset2orig(Ns * Ns);
   for (int i = 0; i < Ns; i++)
      for (int j = 0; j < Ns; j++)
      {
         int index = i * Ns + j;
         subset2orig[index] = new Array<int>;
         sub_topols[index] = new ComponentTopologyHandler(comp_topol, i, j, N, M, *subset2orig[index]);
         sub_solvers[index] = new SteadyNSSolver(sub_topols[index]);
      }
   
   // Base initialization and boundary setup by the global problem.
   for (int k = 0; k < Ns * Ns; k++)
   {
      sub_solvers[k]->InitVariables();
      // sub_solvers[k]->InitVisualization();
      sub_solvers[k]->SetParameterizedProblem(problem);
   }

   // Setting Dirichlet on internal boundaries.
   Array<bool> ensure_incomp(Ns * Ns);
   ensure_incomp = true;
   for (int k = 0; k < Ns * Ns; k++)
   {
      for (int b = 0; b < sub_solvers[k]->global_bdr_attributes.Size(); b++)
      {
         if (sub_solvers[k]->bdr_type[b] == BoundaryType::NUM_BDR_TYPE)
            sub_solvers[k]->bdr_type[b] = BoundaryType::DIRICHLET;
      }

      // save whether incompressibility should be ensured.
      for (int b = 0; b < sub_solvers[k]->global_bdr_attributes.Size(); b++)
      {
         if (sub_solvers[k]->bdr_type[b] == BoundaryType::NEUMANN)
         {
            ensure_incomp[k] = false;
            break;
         }
      }
   }

   // Setting global solution as boundary condition pointer.
   Array<Array<VectorGridFunctionCoefficient *> *> internal_bdr_funcs(Ns * Ns);
   Array<Array<int> *> internal_bdr_meshes(Ns * Ns);
   Array<Array<int> *> internal_bdr_attr(Ns * Ns);
   for (int k = 0; k < Ns * Ns; k++)
   {
      internal_bdr_funcs[k] = new Array<VectorGridFunctionCoefficient *>(0);
      internal_bdr_meshes[k] = new Array<int>(0);
      internal_bdr_attr[k] = new Array<int>(0);
      
      for (int b = 0; b < sub_solvers[k]->global_bdr_attributes.Size(); b++)
      {
         // Internal boundary is Dirichlet BC without a function coefficient yet.
         bool dirichlet = (sub_solvers[k]->bdr_type[b] == BoundaryType::DIRICHLET);
         bool no_func = (!sub_solvers[k]->ud_coeffs[b]);
         if (!(dirichlet && no_func))
            continue;

         // Find all the meshes in the subset that have the boundary attribute.
         int battr = sub_solvers[k]->global_bdr_attributes[b];
         for (int m = 0; m < M * M; m++)
         {
            Mesh *mesh = sub_topols[k]->GetMesh(m);
            int idx = mesh->bdr_attributes.Find(battr);
            if (idx < 0)
               continue;

            int global_m = (*subset2orig[k])[m];

            internal_bdr_meshes[k]->Append(m);
            internal_bdr_attr[k]->Append(battr);
            internal_bdr_funcs[k]->Append(new VectorGridFunctionCoefficient(vels[global_m]));
         }
      }
   }

   // Assemble ROM operators for sub_solvers.
   for (int k = 0; k < Ns * Ns; k++)
   {
      printf("=== Assembly of %d-th sub_solver ===\n", k+1);
      ROMHandlerBase *rom = sub_solvers[k]->GetROMHandler();
      sub_solvers[k]->LoadReducedBasis();

      // SteadyNS is always nonlinear.
      // NOTE: will need if-statement for equation generalization.
      sub_solvers[k]->AllocateROMNlinElems();

      ROMBuildingLevel save_operator = rom->GetBuildingLevel();
      if (save_operator != ROMBuildingLevel::COMPONENT)
         mfem_error("SteadyNSSolver::SchwarzROM- SchwarzROM only supports component-level ROM building!\n");

      printf("Loading ROM projected elements.. ");
      sub_solvers[k]->LoadROMLinElems(filename);
      printf("Done!\n");

      printf("Assembling ROM linear matrix.. ");
      sub_solvers[k]->AssembleROMMat();
      printf("Done!\n");

      sub_solvers[k]->LoadROMNlinElems(rom->GetOperatorPrefix());
      sub_solvers[k]->AssembleROMNlinOper();
   }
   
   DeletePointers(sub_solvers);
   DeletePointers(subset2orig);
}