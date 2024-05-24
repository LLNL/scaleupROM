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
}

UnsteadyNSSolver::~UnsteadyNSSolver()
{

}

void UnsteadyNSSolver::InitVariables()
{
   StokesSolver::InitVariables();
}

bool UnsteadyNSSolver::Solve()
{
   bool converged = true;

   return converged;
}