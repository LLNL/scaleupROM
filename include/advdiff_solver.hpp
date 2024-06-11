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

   // coefficient for prescribed velocity field.
   // can be analytic function or solution from Stokes/SteadyNS equation.
   Array<VectorCoefficient *> flow_coeffs;

   /*
      flow solver to obtain the prescribed velocity field. both StokesSolver / SteadyNSSolver can be used.
   */
   StokesSolver *stokes_solver = NULL;
   bool load_flow = false;
   bool save_flow = false;
   std::string flow_file = "";

   /* grid functions for visualizaing flow field */
   /* NOTE(kevin): this will be set up at SaveVisualization. */
   Array<FiniteElementSpace *> flow_fes;
   Array<GridFunction *> flow_visual;
   FiniteElementSpace *global_flow_fes = NULL;
   Array<GridFunction *> global_flow_visual;

public:
   AdvDiffSolver();

   virtual ~AdvDiffSolver();

   void BuildDomainOperators() override;

   // Component-wise assembly
   void BuildCompROMLinElems(Array<FiniteElementSpace *> &fes_comp) override;

   bool Solve() override;

   void SetFlowAtSubdomain(std::function<void(const Vector &, double, Vector &)> F, const int m=-1);

   void SetParameterizedProblem(ParameterizedProblem *problem) override;

   void SaveVisualization() override;

protected:
   void SetMUMPSSolver() override;

private:
   void GetFlowField(ParameterizedProblem *flow_problem);
};

#endif
