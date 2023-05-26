// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef SCALEUPROM_STOKES_SOLVER_HPP
#define SCALEUPROM_STOKES_SOLVER_HPP

#include "multiblock_solver.hpp"
#include "interfaceinteg.hpp"
#include "dg_mixed_bilin.hpp"
#include "mfem.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class SchurOperator : public Operator
{
protected:
   Operator *A, *B;//, *Bt;
   CGSolver *solver = NULL;

   int maxIter = 10000;
   double rtol = 1.0e-15;
   double atol = 1.0e-15;

public:
   SchurOperator(Operator* const A_, Operator* const B_)
      : Operator(B_->Height()), A(A_), B(B_)
   {
      solver = new CGSolver();
      solver->SetRelTol(rtol);
      solver->SetMaxIter(maxIter);
      solver->SetOperator(*A);
      solver->SetPrintLevel(0);
   };

   virtual ~SchurOperator()
   {
      delete solver;
   }
   
   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector x1(A->NumCols());
      B->MultTranspose(x, x1);

      Vector y1(x1.Size());
      solver->Mult(x1, y1);

      B->Mult(y1, y);
   }
};

class StokesSolver : public MultiBlockSolver
{

friend class ParameterizedProblem;

protected:
   double nu;
   Coefficient *nu_coeff;
   ConstantCoefficient minus_one;

   int porder, uorder;

   // Finite element collection for all fe spaces.
   FiniteElementCollection *ufec;   // for velocity
   FiniteElementCollection *pfec;   // for pressure
   // Finite element spaces
   Array<FiniteElementSpace *> ufes, pfes;

   // GridFunction for pressure
   Array<GridFunction *> ps;
   // Will use us for velocity GridFunction.

   // interface integrator
   InterfaceNonlinearFormIntegrator *vec_diff, *norm_flux;

   // System matrix for Bilinear case.
   Array<int> u_offsets, p_offsets;
   Array2D<SparseMatrix *> m_mats, b_mats;
   BlockMatrix *mMat, *bMat;
   SparseMatrix *M, *B;

   // operators
   Array<LinearForm *> fs, gs;
   Array<BilinearForm *> ms;
   Array<MixedBilinearFormDGExtension *> bs;

   // rhs coefficients
   // The solution dimension is 1 by default, for which using VectorCoefficient is not allowed. (in LinearForm Assemble.)
   // For a derived class for vector solution, this is the first one needs to be changed to Array<VectorCoefficient*>.
   Array<VectorCoefficient *> f_coeffs;
   // Velocity Dirichlet condition
   Array<VectorCoefficient *> ud_coeffs;
   // Stress Neumann condition
   Array<VectorCoefficient *> sn_coeffs;
   bool pres_dbc = false;

   // DG parameters for interior penalty method.
   double sigma = -1.0;
   double kappa = -1.0;

   // // Used for bottom-up building, only with ComponentTopologyHandler.
   // Array<DenseMatrix *> comp_mats;
   // // boundary condition is enforced via forcing term.
   // Array<Array<DenseMatrix *> *> bdr_mats;
   // Array<Array2D<DenseMatrix *> *> port_mats;   // reference ports.

public:
   StokesSolver();

   virtual ~StokesSolver();

   GridFunction* GetVelGridFunction(const int k) const { return us[k]; }
   GridFunction* GetPresGridFunction(const int k) const { return ps[k]; }
   const int GetVelFEOrder() const { return uorder; }
   const int GetPresFEOrder() const { return porder; }

   virtual void SetupBCVariables() override;
   virtual void AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr = -1);
   virtual void AddBCFunction(const Vector &F, const int battr = -1);
   virtual void InitVariables();

   virtual void BuildOperators();
   virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();
   
   virtual void SetupBCOperators() override;
   virtual void SetupRHSBCOperators();
   virtual void SetupDomainBCOperators();

   virtual void AddRHSFunction(std::function<void(const Vector &, Vector &)> F)
   { f_coeffs.Append(new VectorFunctionCoefficient(vdim[0], F)); }
   virtual void AddRHSFunction(const Vector &F)
   { f_coeffs.Append(new VectorConstantCoefficient(F)); }

   virtual void Assemble();
   virtual void AssembleRHS();
   virtual void AssembleOperator();
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrixes();

   // Component-wise assembly
   virtual void AllocateROMElements() {}
   virtual void BuildROMElements() {}
   virtual void SaveROMElements(const std::string &filename) {}
   virtual void LoadROMElements(const std::string &filename) {}
   virtual void AssembleROM() {}

   virtual void Solve();

   void InitUnifiedParaview(const std::string &file_prefix) override
   { mfem_error("StokesSolver::InitUnifiedParaview is not implemented yet!\n"); }
   void InitIndividualParaview(const std::string& file_prefix) override;

   virtual void ProjectOperatorOnReducedBasis() {}
   // { rom_handler->ProjectOperatorOnReducedBasis(mats); }
   virtual double CompareSolution() { return -1.; }
   virtual void SaveBasisVisualization() {}
   // { rom_handler->SaveBasisVisualization(fes); }

   void SanityCheckOnCoeffs();

   virtual void SetParameterizedProblem(ParameterizedProblem *problem) {}
};

#endif