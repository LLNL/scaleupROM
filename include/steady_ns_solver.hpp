// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_STEADY_NS_SOLVER_HPP
#define SCALEUPROM_STEADY_NS_SOLVER_HPP

#include "stokes_solver.hpp"
#include "rom_nonlinearform.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

// A proxy Operator used for FOM Newton Solver.
// TODO(kevin): used a hack of having SteadyNSSolver *solver.
// Ultimately, we should implement InterfaceForm to pass, not the MultiBlockSolver itself.
class SteadyNSOperator : public Operator
{
protected:
   bool direct_solve;

   mutable Vector x_u, y_u;

   Array<int> u_offsets, vblock_offsets;
   InterfaceForm *nl_itf = NULL;
   Array<NonlinearForm *> hs;
   BlockOperator *Hop = NULL;

   BlockMatrix *linearOp = NULL;
   SparseMatrix *M = NULL, *B = NULL, *Bt = NULL;

   // Jacobian matrix objects
   mutable BlockMatrix *system_jac = NULL;
   mutable Array2D<SparseMatrix *> hs_mats;
   mutable BlockMatrix *hs_jac = NULL;
   mutable SparseMatrix *uu_mono = NULL;
   mutable SparseMatrix *mono_jac = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;

   HYPRE_BigInt sys_glob_size;
   mutable HYPRE_BigInt sys_row_starts[2];
public:
   SteadyNSOperator(BlockMatrix *linearOp_, Array<NonlinearForm *> &hs_, InterfaceForm *nl_itf_,
                    Array<int> &u_offsets_, const bool direct_solve_=true);

   virtual ~SteadyNSOperator();

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};

// A proxy Operator used for ROM Newton Solver.
class SteadyNSROM : public Operator
{
protected:
   bool direct_solve;
   bool separate_variable;

   const int num_var = 2;
   int numSub = -1;

   Array<int> block_offsets;
   Array<Array<int> *> block_idxs;
   SparseMatrix *linearOp = NULL;

   mutable Vector x_comp, y_comp;
   mutable SparseMatrix *jac_mono = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;

   HYPRE_BigInt sys_glob_size;
   mutable HYPRE_BigInt sys_row_starts[2];
public:
   SteadyNSROM(SparseMatrix *linearOp_, const int numSub_, const Array<int> &block_offsets_, const bool direct_solve_=true);

   virtual ~SteadyNSROM();

   virtual void Mult(const Vector &x, Vector &y) const = 0;
   virtual Operator &GetGradient(const Vector &x) const = 0;
};

class SteadyNSTensorROM : public SteadyNSROM
{
protected:
   Array<DenseTensor *> hs; // not owned by SteadyNSTensorROM.

public:
   SteadyNSTensorROM(SparseMatrix *linearOp_, Array<DenseTensor *> &hs_, const Array<int> &block_offsets_, const bool direct_solve_=true)
      : SteadyNSROM(linearOp_, hs_.Size(), block_offsets_, direct_solve_), hs(hs_) {}

   virtual ~SteadyNSTensorROM() {}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};

class SteadyNSEQPROM : public SteadyNSROM
{
protected:
   Array<ROMNonlinearForm *> hs; // not owned by SteadyNSEQPROM.

public:
   SteadyNSEQPROM(SparseMatrix *linearOp_, Array<ROMNonlinearForm *> &hs_, const Array<int> &block_offsets_, const bool direct_solve_=true)
      : SteadyNSROM(linearOp_, hs_.Size(), block_offsets_, direct_solve_), hs(hs_) {}

   virtual ~SteadyNSEQPROM() {}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};

class SteadyNSSolver : public StokesSolver
{

friend class ParameterizedProblem;
friend class SteadyNSOperator;

protected:
   enum OperType {
      BASE,
      LF,
      NUM_OPERTYPE
   } oper_type = BASE;

   double zeta = 1.0;
   ConstantCoefficient *zeta_coeff = NULL, *minus_zeta = NULL, *minus_half_zeta = NULL;

   // operator for nonlinear convection.
   Array<NonlinearForm *> hs;
   
   // integration rule for nonliear operator.
   const IntegrationRule *ir_nl = NULL;

   // interface integrator
   InterfaceForm *nl_itf = NULL;
   mutable BlockVector xu_temp, yu_temp;

   // component ROM element for nonlinear convection.
   Array<DenseTensor *> comp_tensors, subdomain_tensors;
   Array<ROMNonlinearForm *> comp_eqps, subdomain_eqps;

   Solver *J_solver = NULL;
   GMRESSolver *J_gmres = NULL;
   NewtonSolver *newton_solver = NULL;

public:
   SteadyNSSolver();

   virtual ~SteadyNSSolver();

   using StokesSolver::GetVariableNames;

   void InitVariables() override;

   void BuildDomainOperators() override;

   void SetupDomainBCOperators() override;

   void AssembleOperator() override;

   void SaveROMOperator(const std::string input_prefix="") override;
   void LoadROMOperatorFromFile(const std::string input_prefix="") override;

   bool Solve(SampleGenerator *sample_generator = NULL) override;

   void ProjectOperatorOnReducedBasis() override;

   void SolveROM() override;

   void AllocateROMTensorElems() override;
   void BuildROMTensorElems() override;
   void SaveROMTensorElems(const std::string &filename) override;
   void LoadROMTensorElems(const std::string &filename) override;
   void AssembleROMTensorOper() override;

   void AllocateROMEQPElems() override;
   void TrainEQPElems(SampleGenerator *sample_generator) override;
   void SaveEQPElems(const std::string &filename) override;
   void LoadEQPElems(const std::string &filename) override;
   void AssembleROMEQPOper() override;

private:
   DenseTensor* GetReducedTensor(DenseMatrix *basis, FiniteElementSpace *fespace);
};

#endif
