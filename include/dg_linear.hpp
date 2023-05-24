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

#ifndef SCALEUPROM_DG_LINEAR_HPP
#define SCALEUPROM_DG_LINEAR_HPP

#include "mfem.hpp"

namespace mfem
{

class DGVectorDirichletLFIntegrator : public DGElasticityDirichletLFIntegrator
{
public:
   DGVectorDirichletLFIntegrator(VectorCoefficient &uD_, Coefficient &mu_,
                                 double alpha_, double kappa_)
      : DGElasticityDirichletLFIntegrator(uD_, mu_, mu_, alpha_, kappa_) { lambda = NULL; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect)
   { mfem_error("DGVectorDirichletLFIntegrator::AssembleRHSElementVect not implemented!\n"); }
   
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};


/// Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
class DGBoundaryNormalLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape;
   VectorCoefficient &Q;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   DGBoundaryNormalLFIntegrator(VectorCoefficient &QG)
      : Q(QG) { }

   virtual bool SupportsDevice() { return false; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   using LinearFormIntegrator::AssembleRHSElementVect;
};

class BoundaryNormalStressLFIntegrator : public LinearFormIntegrator
{
private:
   VectorCoefficient &F;
   Vector shape, nor, Fvec, Fn;
   DenseMatrix Fmat;

public:
   BoundaryNormalStressLFIntegrator(VectorCoefficient &f, const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir), F(f) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

} // namespace mfem

#endif
