// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

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

/// Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
class DGBdrLaxFriedrichsLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape;
   VectorCoefficient &Q;
   DenseMatrix ELV;
   Coefficient *Z = NULL;

   double w;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   DGBdrLaxFriedrichsLFIntegrator(VectorCoefficient &QG, Coefficient *ZG = NULL)
      : Q(QG), Z(ZG) { }

   virtual bool SupportsDevice() { return false; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   using LinearFormIntegrator::AssembleRHSElementVect;
};

class DGBdrTemamLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape;
   Coefficient *Z = NULL;
   VectorCoefficient &Q;
   DenseMatrix ELV;

   double w;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   DGBdrTemamLFIntegrator(VectorCoefficient &QG, Coefficient *ZG = NULL)
      : Q(QG), Z(ZG) { }

   virtual bool SupportsDevice() { return false; }

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
