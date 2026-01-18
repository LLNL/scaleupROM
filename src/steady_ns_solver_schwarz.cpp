// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "steady_ns_solver.hpp"
#include "component_topology_handler.hpp"
#include "hyperreduction_integ.hpp"
#include "nonlinear_integ.hpp"
#include "dg_linear.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

void SteadyNSSolver::SchwarzROM(const int M, const int N, ParameterizedProblem *problem,
                                const int maxIter, const double threshold)
{
   assert(use_rom);
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(N * N == numSub);
   ComponentTopologyHandler *comp_topol = static_cast<ComponentTopologyHandler *>(topol_handler);

   int Ns = N - M + 1;
   Array<ComponentTopologyHandler *> sub_topols(Ns * Ns); // will be owned by sub MultiBlockSolvers defined subsequently.
   Array<SteadyNSSolver *> sub_solvers(Ns * Ns);
   Array<int> i0s(Ns * Ns), j0s(Ns * Ns);
   Array<Array<int> *> subset2orig(Ns * Ns);
   for (int i = 0; i < Ns; i++)
      for (int j = 0; j < Ns; j++)
      {
         int index = i * Ns + j;
         i0s[index] = i;
         j0s[index] = j;
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
         dirichlet = dirichlet || (sub_solvers[k]->bdr_type[b] == BoundaryType::ZERO);
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

   // Set up sub_solvers FOM RHS BC operators.
   for (int k = 0; k < Ns * Ns; k++)
   {
      sub_solvers[k]->BuildRHSOperators();
      sub_solvers[k]->SetupRHSBCOperators();
      // Setup RHS BC operator for internal boundaries
      sub_solvers[k]->SetupSubsetRHSBCOperators(internal_bdr_meshes[k],
                                                internal_bdr_attr[k],
                                                internal_bdr_funcs[k]);
   }

   // Assemble ROM operators for sub_solvers.
   for (int k = 0; k < Ns * Ns; k++)
   {
      sub_solvers[k]->InitROMHandler();

      printf("\n=== Assembly of %d-th sub_solver ===\n\n", k+1);
      ROMHandlerBase *rom = sub_solvers[k]->GetROMHandler();
      sub_solvers[k]->LoadReducedBasis();

      // SteadyNS is always nonlinear.
      // NOTE: will need if-statement for equation generalization.
      sub_solvers[k]->AllocateROMNlinElems();

      ROMBuildingLevel save_operator = rom->GetBuildingLevel();
      if (save_operator != ROMBuildingLevel::COMPONENT)
         mfem_error("SteadyNSSolver::SchwarzROM- SchwarzROM only supports component-level ROM building!\n");

      printf("Loading ROM projected elements.. ");
      std::string filename = rom->GetOperatorPrefix() + ".h5";
      sub_solvers[k]->LoadROMLinElems(filename);
      printf("Done!\n");

      printf("Assembling ROM linear matrix.. ");
      sub_solvers[k]->AssembleROMMat();
      printf("Done!\n");

      sub_solvers[k]->LoadROMNlinElems(rom->GetOperatorPrefix());
      sub_solvers[k]->AssembleROMNlinOper();
   }

   // Global solution initialization
   // HACK: we assume the ud_coeff is the same for all non-zero Dirichlet condition.
   for (int b = 0; b < global_bdr_attributes.Size(); b++)
   {
      if ((bdr_type[b] == BoundaryType::DIRICHLET) && ud_coeffs[b])
      {
         for (int m = 0; m < numSub; m++)
            vels[m]->ProjectCoefficient(*ud_coeffs[b]);
         break;
      }
   }
   // for (int k = 0; k < U->Size(); k++)
   //    (*U)[k] = 1.0e-5 * UniformRandom();

   // Main Schwarz loop.
   bool use_restart = config.GetOption<bool>("rom_solver/use_restart", false);
   double error = 0.0;
   for (int iter = 0; iter < maxIter; iter++)
   {
      error = 0.0;
      // Sweep through sub-solvers.
      for (int k = 0; k < Ns * Ns; k++)
      {
         // int k = sweep_index[s];
         printf("Switched to %dth subsolver.\n", k+1);

         // Adjust global solution to ensure divergence-free BC.
         if (ensure_incomp[k])
            SetSubsetComplementaryFlux(N, M, i0s[k], j0s[k],
                                       sub_solvers[k]->global_bdr_attributes,
                                       sub_solvers[k]->bdr_type, problem);

         // Project global solution to subsolver solution.
         if (use_restart)
         {
            for (int m = 0; m < sub_solvers[k]->numSub; m++)
            {
               const int orig_idx = (*subset2orig[k])[m];
               (*(sub_solvers[k]->vels[m])) = (*vels[orig_idx]);
            }
         }

         // Assemble FOM-level RHS.
         // All RHS BC operators are already defined,
         // and linked to the adjusted global solution.
         sub_solvers[k]->AssembleRHS();

         sub_solvers[k]->ProjectRHSOnReducedBasis();

         // Solve for the subsolver
         sub_solvers[k]->SolveROM();

         // Compute relative error after iteration.
         double error1 = 0.0;
         int norm = 0.0;
         for (int m = 0; m < sub_solvers[k]->numSub; m++)
         {
            double subdomain_error, subdomain_norm;

            const int orig_idx = (*subset2orig[k])[m];
            ComputeSubdomainErrorAndNorm(vels[orig_idx], sub_solvers[k]->vels[m],
                                         subdomain_error, subdomain_norm);
            norm += subdomain_norm * subdomain_norm;
            error += subdomain_error * subdomain_error;
         }
         norm = sqrt(norm);
         error1 = sqrt(error1);
         error1 /= norm;
         error = max(error, error1);

         // Project subsolver solution to global solution.
         for (int m = 0; m < sub_solvers[k]->numSub; m++)
         {
            const int orig_idx = (*subset2orig[k])[m];
            (*vels[orig_idx]) = (*(sub_solvers[k]->vels[m]));
         }
      }  // for (int k = 0; k < Ns * Ns; k++)

      printf("Iteration %d error: %.4e\n", iter+1, error);
      // Exit the iterations if error is below threshold.
      if (error <= threshold)
      {
         printf("SteadyNSSolver::SchwarzROM- Schwarz iteration converged.\n");
         break;
      }
   }  // for (int iter = 0; iter < maxIter; iter++)

   if (error > threshold)
      mfem_error("SteadyNSSolver::SchwarzROM- Schwarz iteration failed to converge!\n");
   
   DeletePointers(sub_solvers);
   DeletePointers(subset2orig);
}

void SteadyNSSolver::SetupSubsetRHSBCOperators(
   const Array<int> *bmeshes, const Array<int> *battrs,
   const Array<VectorGridFunctionCoefficient *> *bfuncs)
{
   const int N = bmeshes->Size();
   assert((battrs->Size() == N) && (bfuncs->Size() == N));

   for (int k = 0; k < N; k++)
   {
      const int m = (*bmeshes)[k];
      const int battr = (*battrs)[k];
      VectorGridFunctionCoefficient *bfunc = (*bfuncs)[k];
      const int global_idx = global_bdr_attributes.Find(battr);
      const int bidx = meshes[m]->bdr_attributes.Find(battr);

      assert(fs[m] && gs[m]);
      assert(bidx >= 0);
      assert(bdr_type[global_idx] == BoundaryType::DIRICHLET);
      assert(!BCExistsOnBdr(global_idx)); // For internal boundary, global ud coefficient is not defined.

      fs[m]->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(*bfunc, *nu_coeff, sigma, kappa), *bdr_markers[global_idx]);

      if (full_dg)
         gs[m]->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(*bfunc), *bdr_markers[global_idx]);
      else
         gs[m]->AddBoundaryIntegrator(new DGBoundaryNormalLFIntegrator(*bfunc), *bdr_markers[global_idx]);
   }
}