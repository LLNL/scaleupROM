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


public:
   UnsteadyNSSolver();

   virtual ~UnsteadyNSSolver();

   using SteadyNSSolver::GetVariableNames;

   void InitVariables() override;

   void SaveROMOperator(const std::string input_prefix="") override
   { mfem_error("UnsteadyNSSolver::SaveROMOperator is not implemented yet!\n"); }
   void LoadROMOperatorFromFile(const std::string input_prefix="") override
   { mfem_error("UnsteadyNSSolver::LoadROMOperatorFromFile is not implemented yet!\n"); }

   bool Solve() override;

   void ProjectOperatorOnReducedBasis() override
   { mfem_error("UnsteadyNSSolver::ProjectOperatorOnReducedBasis is not implemented yet!\n"); }

   void SolveROM() override
   { mfem_error("UnsteadyNSSolver::SolveROM is not implemented yet!\n"); }

   void AllocateROMEQPElems() override
   { mfem_error("UnsteadyNSSolver::AllocateROMEQPElems is not implemented yet!\n"); }
   void TrainEQPElems(SampleGenerator *sample_generator) override
   { mfem_error("UnsteadyNSSolver::TrainEQPElems is not implemented yet!\n"); }
   void SaveEQPElems(const std::string &filename) override
   { mfem_error("UnsteadyNSSolver::SaveEQPElems is not implemented yet!\n"); }
   void LoadEQPElems(const std::string &filename) override
   { mfem_error("UnsteadyNSSolver::LoadEQPElems is not implemented yet!\n"); }
   void AssembleROMEQPOper() override
   { mfem_error("UnsteadyNSSolver::AssembleROMEQPOper is not implemented yet!\n"); }

   /* Tensorial ROM is not supported for unsteady NS */
   void AllocateROMTensorElems() override
   { mfem_error("UnsteadyNSSolver::AllocateROMTensorElems is not implemented yet!\n"); }
   void BuildROMTensorElems() override
   { mfem_error("UnsteadyNSSolver::BuildROMTensorElems is not implemented yet!\n"); }
   void SaveROMTensorElems(const std::string &filename) override
   { mfem_error("UnsteadyNSSolver::SaveROMTensorElems is not implemented yet!\n"); }
   void LoadROMTensorElems(const std::string &filename) override
   { mfem_error("UnsteadyNSSolver::LoadROMTensorElems is not implemented yet!\n"); }
   void AssembleROMTensorOper() override
   { mfem_error("UnsteadyNSSolver::AssembleROMTensorOper is not implemented yet!\n"); }

};

#endif
