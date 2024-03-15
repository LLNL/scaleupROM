// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "advdiff_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

AdvDiffSolver::AdvDiffSolver()
   : PoissonSolver()
{
   // ConvectionIntegrator does not support L2 space.
   assert(!full_dg);

   Pe = config.GetOption<double>("adv-diff/peclet_number", 0.0);

   flow_coeffs.SetSize(numSub);

   save_flow = config.GetOption<bool>("adv-diff/save_flow", false);
   load_flow = config.GetOption<bool>("adv-diff/load_flow", false);
   if (save_flow || load_flow)
      flow_file = config.GetRequiredOption<std::string>("adv-diff/flow_file");
}

AdvDiffSolver::~AdvDiffSolver()
{
   DeletePointers(flow_coeffs);
   delete stokes_solver;
}

void AdvDiffSolver::BuildDomainOperators()
{
   PoissonSolver::BuildDomainOperators();

   for (int m = 0; m < numSub; m++)
   {
      assert(flow_coeffs[m]);
      as[m]->AddDomainIntegrator(new ConvectionIntegrator(*flow_coeffs[m], Pe));
   }
}

void AdvDiffSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   mfem_error("AdvDiffSolver::BuildCompROMElement is not implemented yet!\n");

   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = fes_comp.Size();
   assert(comp_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      BilinearForm a_comp(fes_comp[c]);

      a_comp.AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         a_comp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));

      // TODO(kevin): ConvectionIntegrator needs to be added

      a_comp.Assemble();
      a_comp.Finalize();

      // Poisson equation has only one solution variable.
      comp_mats[c]->SetSize(1, 1);
      (*comp_mats[c])(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
   }
}

bool AdvDiffSolver::Solve()
{
   // If using direct solver, returns always true.
   bool converged = true;

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   // TODO: need to change when the actual parallelization is implemented.
   if (direct_solve)
   {
      assert(mumps);
      mumps->SetPrintLevel(print_level);
      mumps->Mult(*RHS, *U);
   }
   else
   {
      GMRESSolver *solver = NULL;
      HypreBoomerAMG *M = NULL;
      BlockDiagonalPreconditioner *globalPrec = NULL;
      
      // HypreBoomerAMG makes a meaningful difference in computation time.
      if (use_amg)
      {
         // Initializating HypreParMatrix needs the monolithic sparse matrix.
         assert(globalMat_mono != NULL);

         solver = new GMRESSolver(MPI_COMM_SELF);

         M = new HypreBoomerAMG(*globalMat_hypre);
         M->SetPrintLevel(print_level);

         solver->SetPreconditioner(*M);
         solver->SetOperator(*globalMat_hypre);
      }
      else
      {
         solver = new GMRESSolver();
         
         if (config.GetOption<bool>("solver/block_diagonal_preconditioner", true))
         {
            globalPrec = new BlockDiagonalPreconditioner(var_offsets);
            solver->SetPreconditioner(*globalPrec);
         }
         solver->SetOperator(*globalMat);
      }
      solver->SetAbsTol(atol);
      solver->SetRelTol(rtol);
      solver->SetMaxIter(maxIter);
      solver->SetPrintLevel(print_level);

      *U = 0.0;
      // The time for the setup above is much smaller than this Mult().
      // StopWatch test;
      // test.Start();
      solver->Mult(*RHS, *U);
      // test.Stop();
      // printf("test: %f seconds.\n", test.RealTime());
      converged = solver->GetConverged();

      // delete the created objects.
      if (use_amg)
      {
         delete M;
      }
      else
      {
         if (globalPrec != NULL) delete globalPrec;
      }
      delete solver;
   }

   return converged;
}

void AdvDiffSolver::SetFlowAtSubdomain(std::function<void(const Vector &, Vector &)> F, const int m)
{
   assert(flow_coeffs.Size() == numSub);
   assert((m == -1) || ((m >= 0) && (m < numSub)));

   if (m == -1)
   {
      for (int k = 0; k < numSub; k++)
         flow_coeffs[k] = new VectorFunctionCoefficient(dim, F);
   }
   else
      flow_coeffs[m] = new VectorFunctionCoefficient(dim, F);
}

void AdvDiffSolver::SetParameterizedProblem(ParameterizedProblem *problem)
{
   if (!function_factory::advdiff_problem::analytic_flow)
      GetFlowField(function_factory::advdiff_problem::flow_problem);

   PoissonSolver::SetParameterizedProblem(problem);
}

void AdvDiffSolver::SetMUMPSSolver()
{
   assert(globalMat_hypre);
   mumps = new MUMPSSolver();
   mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps->SetOperator(*globalMat_hypre);
}

void AdvDiffSolver::GetFlowField(ParameterizedProblem *flow_problem)
{
   assert(flow_problem);
   mfem_warning("AdvDiffSolver: Obtaining flow field. This may take a while depending on the domain size.\n");

   stokes_solver = new StokesSolver;
   stokes_solver->InitVariables();
   stokes_solver->InitVisualization();

   if (load_flow && FileExists(flow_file))
      stokes_solver->LoadSolution(flow_file);
   else
   {
      stokes_solver->SetParameterizedProblem(flow_problem);
      // currently only support FOM.
      stokes_solver->BuildOperators();
      stokes_solver->SetupBCOperators();
      stokes_solver->Assemble();
      stokes_solver->Solve();
   }

   if (save_flow)
      stokes_solver->SaveSolution(flow_file);

   DeletePointers(flow_coeffs);
   const int stokes_numvar = stokes_solver->GetNumVar();
   for (int m = 0; m < numSub; m++)
      flow_coeffs[m] = new VectorGridFunctionCoefficient(stokes_solver->GetGridFunction(m * stokes_numvar));

   /*
      VectorGridFunctionCoefficient does not own the grid function,
      and it requires the grid function for its lifetime.
      Thus stokes_solver will be deleted at ~AdvDiffSolver().
   */
}