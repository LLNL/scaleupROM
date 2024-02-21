// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_LINELAST_SOLVER_HPP
#define SCALEUPROM_LINELAST_SOLVER_HPP

#include "multiblock_solver.hpp"
#include "interfaceinteg.hpp"
#include "mfem.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class LinElastSolver : public MultiBlockSolver
{

   friend class ParameterizedProblem;

protected:
   // interface integrator
   InterfaceForm *a_itf = NULL;
   // int skip_zeros = 1;

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;
   // For nonlinear problem
   // BlockOperator *globalMat;
   BlockMatrix *globalMat = NULL;
   SparseMatrix *globalMat_mono = NULL;

   // variables needed for direct solve
   HYPRE_BigInt sys_glob_size;
   HYPRE_BigInt sys_row_starts[2];
   HypreParMatrix *globalMat_hypre = NULL;
   MUMPSSolver *mumps = NULL;

   // operators
   Array<LinearForm *> bs;
   Array<BilinearForm *> as;

   // Lame constants for each subdomain, global boundary attribute ordering
   Array<VectorConstantCoefficient *> lambda_c;
   Array<VectorConstantCoefficient *> mu_c;
   Array<VectorFunctionCoefficient *> bdr_coeffs;
   Array<VectorFunctionCoefficient *> rhs_coeffs;

   // DG parameters specific to linear elasticity equation.
   double alpha = -1.0;
   double kappa = -1.0;

   // Initial positions
   VectorFunctionCoefficient *init_x;

public:
   LinElastSolver();

   virtual ~LinElastSolver();

   static const std::vector<std::string> GetVariableNames()
   {
      std::vector<std::string> varnames(1);
      varnames[0] = "solution";
      return varnames;
   }

   virtual void InitVariables();

   virtual void BuildOperators();
   virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();

   virtual void Assemble();
   virtual void AssembleRHS();
   virtual void AssembleOperator();
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrixes();

   virtual bool Solve();

   virtual void SetupBCVariables() override;
   virtual void SetupIC(std::function<void(const Vector &, Vector &)> F);
   virtual void AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr = -1);
   virtual void AddRHSFunction(std::function<void(const Vector &, Vector &)> F);
   virtual bool BCExistsOnBdr(const int &global_battr_idx);
   virtual void SetupBCOperators();
   virtual void SetupRHSBCOperators();
   virtual void SetupDomainBCOperators();

   // Component-wise assembly
   virtual void BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp);
   virtual void BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp);
   virtual void BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp);

   virtual void ProjectOperatorOnReducedBasis();

   void SanityCheckOnCoeffs();

   virtual void SetParameterizedProblem(ParameterizedProblem *problem);
};

#endif
