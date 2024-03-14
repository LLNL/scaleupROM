// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_ADVDIFF_SOLVER_HPP
#define SCALEUPROM_ADVDIFF_SOLVER_HPP

#include "poisson_solver.hpp"
#include "stokes_solver.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class AdvDiffSolver : public PoissonSolver
{

friend class ParameterizedProblem;

protected:
   double Pe = 0.0;  // Peclet number

public:
   AdvDiffSolver();

   virtual ~AdvDiffSolver();

   void BuildDomainOperators() override;

   // Component-wise assembly
   void BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp) override;

protected:
   void SetMUMPSSolver() override;
};

#endif
