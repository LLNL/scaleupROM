// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "unsteady_ns_solver.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

/*
   UnsteadyNSSolver
*/

UnsteadyNSSolver::UnsteadyNSSolver()
   : SteadyNSSolver()
{
   nonlinear_mode = true;

   /* unsteady ns solver only uses LaxFridriechs flux solver */
   if (oper_type != OperType::LF)
      mfem_error("UnsteadyNSSolver only support Lax-Fridriechs flux integrator!\n");

   nt = config.GetRequiredOption<int>("time-integration/number_of_timesteps");
   dt = config.GetRequiredOption<double>("time-integration/timestep_size");
   time_order = config.GetOption<int>("time-integration/bdf_order", 1);

   if (time_order != 1)
      mfem_error("UnsteadyNSSolver supports only first-order time integration for now!\n");
}

UnsteadyNSSolver::~UnsteadyNSSolver()
{
   DeletePointers(mass);
   delete massMat;
   delete uu;
   delete U_step;
   delete RHS_step;
   delete U_stepview;
   delete RHS_stepview;
   delete Hop;
}

void UnsteadyNSSolver::InitVariables()
{
   StokesSolver::InitVariables();
}

void UnsteadyNSSolver::BuildDomainOperators()
{
   SteadyNSSolver::BuildDomainOperators();

   mass.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      mass[m] = new BilinearForm(ufes[m]);
      mass[m]->AddDomainIntegrator(new VectorMassIntegrator);
   }
}

void UnsteadyNSSolver::AssembleOperator()
{
   SteadyNSSolver::AssembleOperator();

   for (int m = 0; m < numSub; m++)
   {
      mass[m]->Assemble();
      mass[m]->Finalize();
   }

   massMat = new BlockMatrix(u_offsets);
   for (int i = 0; i < numSub; i++)
      massMat->SetBlock(i, i, &(mass[i]->SpMat()));
}

bool UnsteadyNSSolver::Solve()
{
   bool converged = true;

   InitializeTimeIntegration();

   SortByVariables(*U, *U_step);

   double time = 0.0;
   for (int step = 0; step < nt; step++)
   {
      Step(time, step);
   }

   SortBySubdomains(*U_step, *U);

   return converged;
}

void UnsteadyNSSolver::InitializeTimeIntegration()
{
   assert(dt > 0.0);

   uu = new SparseMatrix(vblock_offsets[1]);
   SparseMatrix *tmp = massMat->CreateMonolithic();
   (*uu) += *tmp;
   (*uu) *= bd0 / dt;
   (*uu) += (*M); // add viscous flux operator
   uu->Finalize();
   delete tmp;

   // This should be run after AssembleOperator
   assert(systemOp);
   systemOp->SetBlock(0, 0, uu);
   systemOp->SetBlock(0, 1, Bt);
   systemOp->SetBlock(1, 0, B);

   StokesSolver::SetupMUMPSSolver(true);

   Hop = new BlockOperator(u_offsets);
   for (int m = 0; m < numSub; m++)
      Hop->SetDiagonalBlock(m, hs[m]);

   offsets_byvar.SetSize(num_var * numSub + 1);
   offsets_byvar = 0;
   for (int k = 0; k < numSub; k++)
   {
      offsets_byvar[k+1] = u_offsets[k+1];
      offsets_byvar[k+1 + numSub] = p_offsets[k+1] + u_offsets.Last();
   }

   U_step = new BlockVector(offsets_byvar);
   RHS_step = new BlockVector(offsets_byvar);
   U_stepview = new BlockVector(U_step->GetData(), vblock_offsets);
   RHS_stepview = new BlockVector(RHS_step->GetData(), vblock_offsets);

   u1.SetSize(U_stepview->BlockSize(0));
   Cu1.SetSize(U_stepview->BlockSize(0));
}

void UnsteadyNSSolver::Step(double &time, int step)
{
   /* copy velocity */
   u1 = U_stepview->GetBlock(0);

   /* evaluate nonlinear advection at previous time step */
   Hop->Mult(u1, Cu1);
   nl_itf->InterfaceAddMult(u1, Cu1);

   /* Base right-hand side for boundary conditions and forcing */
   SortByVariables(*RHS, *RHS_step);

   /* Add nonlinear convection */
   RHS_stepview->GetBlock(0).Add(-ab1, Cu1);

   /* Add time derivative term */
   // TODO: extend for high order bdf schemes
   massMat->AddMult(u1, RHS_stepview->GetBlock(0), -bd1 / dt);

   /* Solve for the next step */
   mumps->Mult(*RHS_step, *U_step);

   /* remove pressure scalar if all dirichlet bc */
   if (!pres_dbc)
   {
      double p_const = U_stepview->GetBlock(1).Sum() / U_stepview->GetBlock(1).Size();

      U_stepview->GetBlock(1) -= p_const;
   }

   time += dt;
}