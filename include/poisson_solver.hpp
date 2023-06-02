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
   InterfaceNonlinearFormIntegrator *interface_integ;
   // int skip_zeros = 1;

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;
   // For nonlinear problem
   // BlockOperator *globalMat;
   BlockMatrix *globalMat;
   SparseMatrix *globalMat_mono;

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

   virtual void SetupBCVariables() override;
   virtual void AddBCFunction(std::function<double(const Vector &)> F, const int battr = -1);
   virtual void AddBCFunction(const double &F, const int battr = -1);
   virtual void InitVariables();

   virtual void BuildOperators();
   virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();
   
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
   virtual void AssembleInterfaceMatrixes();

   // Component-wise assembly
   virtual void BuildROMElements();
   virtual void SaveROMElements(const std::string &filename);
   virtual void LoadROMElements(const std::string &filename);
   virtual void AssembleROM();

   virtual void Solve();

   virtual void ProjectOperatorOnReducedBasis();

   void SanityCheckOnCoeffs();

   virtual void SetParameterizedProblem(ParameterizedProblem *problem);
};

#endif
