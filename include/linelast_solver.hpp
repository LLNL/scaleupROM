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
   Array<PWConstCoefficient *> lambda_cs;
   Array<PWConstCoefficient *> mu_cs;

   // DG parameters specific to linear elasticity equation.
   double sigma = -1.0;
   double kappa = -1.0;

   // Initial positions
   VectorFunctionCoefficient init_x;

public:
   LinElastSolver();

   virtual ~LinElastSolver();

   static const std::vector<std::string> GetVariableNames()
   {
      std::vector<std::string> varnames(1);
      varnames[0] = "solution";
      return varnames;
   }

   static void InitDisplacement(const Vector &x, Vector &u) // Making this static for now
   {
      u = 0.0;
      u(u.Size() - 1) = -0.2 * x(0);
   }

   virtual void SetupBCVariables() override;
   virtual void AddBCFunction(std::function<double(const Vector &)> F, const int battr = -1);
   virtual void AddBCFunction(const double &F, const int battr = -1);
   virtual void InitVariables();

   virtual void BuildOperators();
   virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();

   virtual bool BCExistsOnBdr(const int &global_battr_idx);
   virtual void SetupBCOperators();
   virtual void SetupRHSBCOperators();
   virtual void SetupDomainBCOperators();

   virtual void AddRHSFunction(std::function<double(const Vector &)> F)
   {
      rhs_coeffs.Append(new FunctionCoefficient(F));
   }
   virtual void AddRHSFunction(const double F)
   {
      rhs_coeffs.Append(new ConstantCoefficient(F));
   }

   virtual void Assemble();
   virtual void AssembleRHS();
   virtual void AssembleOperator();
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrixes();

   // Component-wise assembly
   virtual void BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp);
   virtual void BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp);
   virtual void BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp);

   virtual bool Solve();

   virtual void ProjectOperatorOnReducedBasis();

   void SanityCheckOnCoeffs();

   virtual void SetParameterizedProblem(ParameterizedProblem *problem);
};

#endif
