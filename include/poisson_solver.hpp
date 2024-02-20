// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_POISSON_SOLVER_HPP
#define SCALEUPROM_POISSON_SOLVER_HPP

#include "multiblock_solver.hpp"
#include "interfaceinteg.hpp"
#include "mfem.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class PoissonSolver : public MultiBlockSolver
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

   // rhs coefficients
   // The solution dimension is 1 by default, for which using VectorCoefficient is not allowed. (in LinearForm Assemble.)
   // For a derived class for vector solution, this is the first one needs to be changed to Array<VectorCoefficient*>.
   Array<Coefficient *> rhs_coeffs;
   Array<Coefficient *> bdr_coeffs;

   // DG parameters specific to Poisson equation.
   double sigma = -1.0;
   double kappa = -1.0;

public:
   PoissonSolver();

   virtual ~PoissonSolver();

   static const std::vector<std::string> GetVariableNames()
   {
      std::vector<std::string> varnames(1);
      varnames[0] = "solution";
      return varnames;
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
   { rhs_coeffs.Append(new FunctionCoefficient(F)); }
   virtual void AddRHSFunction(const double F)
   { rhs_coeffs.Append(new ConstantCoefficient(F)); }

   virtual void Assemble();
   virtual void AssembleRHS();
   virtual void AssembleOperator();
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrices();

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
