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

#ifndef SCALEUPROM_HYPERREDUCTION_INTEG_HPP
#define SCALEUPROM_HYPERREDUCTION_INTEG_HPP

#include "mfem.hpp"

namespace mfem
{

class HyperReductionIntegrator : virtual public NonlinearFormIntegrator
{
public:
   virtual void AssembleQuadratureVector(const FiniteElement &el,
                                          ElementTransformation &T,
                                          const IntegrationPoint &ip,
                                          const double &iw,
                                          const Vector &eltest,
                                          Vector &elquad);

   virtual void AssembleQuadratureVector(const FiniteElement &el1,
                                          const FiniteElement &el2,
                                          FaceElementTransformations &T,
                                          const IntegrationPoint &ip,
                                          const double &iw,
                                          const Vector &eltest,
                                          Vector &elquad);
};

class VectorConvectionTrilinearFormIntegrator : virtual public HyperReductionIntegrator
{
private:
   int dim;
   Coefficient *Q{};
   VectorCoefficient *vQ{};
   DenseMatrix dshape, dshapex, elmat_comp, EF, gradEF, ELV;
   Vector shape;

public:
   VectorConvectionTrilinearFormIntegrator(Coefficient &q, VectorCoefficient *vq = NULL)
      : Q(&q), vQ(vq), HyperReductionIntegrator() { }

   VectorConvectionTrilinearFormIntegrator() = default;

   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleQuadratureVector(const FiniteElement &el,
                                          ElementTransformation &T,
                                          const IntegrationPoint &ip,
                                          const double &iw,
                                          const Vector &eltest,
                                          Vector &elquad) override;

   // void AssembleElementQuadrature(const FiniteElement &el,
   //                               ElementTransformation &T,
   //                               const Vector &eltest,
   //                               DenseMatrix &elquad);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
};

} // namespace mfem

#endif
