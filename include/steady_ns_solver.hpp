// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_STEADY_NS_SOLVER_HPP
#define SCALEUPROM_STEADY_NS_SOLVER_HPP

#include "stokes_solver.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

// A proxy Operator used for Newton Solver.
class SteadyNSOperator : public Operator
{
protected:
   bool direct_solve;

   Array<int> u_offsets, vblock_offsets;
   Array<NonlinearForm *> hs;
   BlockOperator *Hop = NULL;

   BlockMatrix *linearOp = NULL;
   SparseMatrix *M = NULL, *B = NULL, *Bt = NULL;

   // Jacobian matrix objects
   mutable BlockMatrix *system_jac = NULL;
   mutable Array<SparseMatrix *> hs_mats;
   mutable BlockMatrix *hs_jac = NULL;
   mutable SparseMatrix *uu_mono = NULL;
   mutable SparseMatrix *mono_jac = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;

   HYPRE_BigInt sys_glob_size;
   mutable HYPRE_BigInt sys_row_starts[2];
public:
   SteadyNSOperator(BlockMatrix *linearOp_, Array<NonlinearForm *> &hs_, Array<int> &u_offsets_, const bool direct_solve_=true);

   virtual ~SteadyNSOperator();

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};

class SteadyNSSolver : public StokesSolver
{

friend class ParameterizedProblem;

protected:
   double zeta = 1.0;
   ConstantCoefficient zeta_coeff;

   // operator for nonlinear convection.
   Array<NonlinearForm *> hs;
   
   // integration rule for nonliear operator.
   const IntegrationRule *ir_nl = NULL;

   // component ROM element for nonlinear convection.
   Array<DenseTensor *> comp_tensors;

   Solver *J_solver = NULL;
   GMRESSolver *J_gmres = NULL;
   NewtonSolver *newton_solver = NULL;

public:
   SteadyNSSolver();

   virtual ~SteadyNSSolver();

   // virtual void SetupBCVariables() override;
   // virtual void AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr = -1);
   // virtual void AddBCFunction(const Vector &F, const int battr = -1);
   // virtual void InitVariables();

   // void DeterminePressureDirichlet();

   virtual void BuildOperators() override;
   // virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();
   
   // virtual bool BCExistsOnBdr(const int &global_battr_idx);
   // virtual void SetupBCOperators() override;
   // virtual void SetupRHSBCOperators();
   // virtual void SetupDomainBCOperators();

   virtual void Assemble();
   // virtual void AssembleRHS();
   // virtual void AssembleOperator();
   // // For bilinear case.
   // // system-specific.
   // virtual void AssembleInterfaceMatrixes();

   // Component-wise assembly
   virtual void BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp);
   // virtual void BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp);
   // virtual void BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp);
   virtual void SaveCompBdrROMElement(hid_t &file_id);
   virtual void LoadCompBdrROMElement(hid_t &file_id);

   virtual void Solve();

   virtual void ProjectOperatorOnReducedBasis();

   // void SanityCheckOnCoeffs();

   // virtual void SetParameterizedProblem(ParameterizedProblem *problem) override;

   // // to ensure incompressibility for the problems with all velocity dirichlet bc.
   // void SetComplementaryFlux(const Array<bool> nz_dbcs);

};

#endif
