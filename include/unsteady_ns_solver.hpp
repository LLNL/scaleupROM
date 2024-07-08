// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_UNSTEADY_NS_SOLVER_HPP
#define SCALEUPROM_UNSTEADY_NS_SOLVER_HPP

#include "steady_ns_solver.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class UnsteadyNSSolver : public SteadyNSSolver
{

friend class ParameterizedProblem;
friend class SteadyNSOperator;

protected:

   // number of timesteps
   int nt = -1;
   // timestep size
   double dt = -1.0;
   // BDF time integrator order
   int time_order = -1;
   // report interval for time integration
   int report_interval = 0;
   // restart save interval
   int restart_interval = 0;

   // BDFk/EXTk coefficients.
   /* use first order for now. */
   double bd0 = 1.0;
   double bd1 = -1.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 1.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   /* velocity and its convection at previous time step */
   Vector u1;
   Vector Cu1;

   /* mass matrix operator for time-derivative term */
   Array<BilinearForm *> mass;
   BlockMatrix *massMat = NULL;

   /* For coupled solution approach */
   SparseMatrix *uu = NULL;

   /* proxy variables for time integration */
   Array<int> offsets_byvar;
   BlockVector *U_step = NULL;
   BlockVector *RHS_step = NULL;
   BlockVector *U_stepview = NULL;
   BlockVector *RHS_stepview = NULL;

   BlockOperator *Hop = NULL;

   /* mass matrix operator for ROM */
   Array<int> rom_u_offsets;
   BlockMatrix *rom_mass = NULL;

public:
   UnsteadyNSSolver();

   virtual ~UnsteadyNSSolver();

   using SteadyNSSolver::GetVariableNames;
   
   void BuildDomainOperators() override;
   void AssembleOperator() override;

   void SaveROMOperator(const std::string input_prefix="") override
   { mfem_error("UnsteadyNSSolver::SaveROMOperator is not implemented yet!\n"); }
   void LoadROMOperatorFromFile(const std::string input_prefix="") override
   { mfem_error("UnsteadyNSSolver::LoadROMOperatorFromFile is not implemented yet!\n"); }

   bool Solve(SampleGenerator *sample_generator = NULL) override;

   using MultiBlockSolver::SaveVisualization;
   void SaveVisualization(const int step, const double time) override;

   void SetParameterizedProblem(ParameterizedProblem *problem) override;

   BlockVector* PrepareSnapshots(std::vector<BasisTag> &basis_tags) override;

   void ProjectOperatorOnReducedBasis() override
   { mfem_error("UnsteadyNSSolver::ProjectOperatorOnReducedBasis is not implemented yet!\n"); }

   void BuildCompROMLinElems() override;

   void SolveROM() override
   { mfem_error("UnsteadyNSSolver::SolveROM is not implemented yet!\n"); }

   void InitROMHandler() override;

   void BuildROMTensorElems() override
   { mfem_error("UnsteadyNSSolver::BuildROMTensorElems is not implemented yet!\n"); }

private:
   void InitializeTimeIntegration();
   void Step(double &time, int step);

   void SanityCheck(const int step)
   {
      if (isnan(U_step->Min()) || isnan(U_step->Max()))
      {
         printf("Step : %d\n", step);
         mfem_error("UnsteadyNSSolver: Solution blew up!!\n");
      }
   }
   double ComputeCFL(const double dt);
   void SetTime(const double time);

   void AssembleROMMat(BlockMatrix &romMat) override;

};

#endif
