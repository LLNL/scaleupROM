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

#ifndef SCALEUPROM_INTERFACE_INTEGRATOR_HPP
#define SCALEUPROM_INTERFACE_INTEGRATOR_HPP

// #include <fem/bilininteg.hpp>
#include "mfem.hpp"

namespace mfem
{

class InterfaceNonlinearFormIntegrator : public NonlinearFormIntegrator
{
protected:
  InterfaceNonlinearFormIntegrator(const IntegrationRule *ir = NULL)
      : NonlinearFormIntegrator(ir) {}
public:
   // FaceElementTransformations belongs to one mesh (having mesh pointer).
   // In order to extract element/transformation from each mesh,
   // two FaceElementTransformations are needed.
   virtual void AssembleInterfaceVector(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr1,
                                       FaceElementTransformations &Tr2,
                                       const Vector &elfun, Vector &elvect);

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   virtual void AssembleInterfaceGrad(const FiniteElement &el1,
                                      const FiniteElement &el2,
                                      FaceElementTransformations &Tr1,
                                      FaceElementTransformations &Tr2,
                                      const Vector &elfun, DenseMatrix &elmat);

   virtual void AssembleInterfaceMatrix(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Trans1,
                                       FaceElementTransformations &Trans2,
                                       Array2D<DenseMatrix*> &elmats);
};

class InterfaceDGDiffusionIntegrator : public InterfaceNonlinearFormIntegrator
{
protected:
  Coefficient *Q;
  MatrixCoefficient *MQ;
  double sigma, kappa;

  // these are not thread-safe!
  Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
  DenseMatrix dshape1, dshape2, mq, adjJ;
  Vector nor2;
  Array2D<DenseMatrix*> jmats;

public:
   InterfaceDGDiffusionIntegrator(const double s, const double k)
      : Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   InterfaceDGDiffusionIntegrator(Coefficient &q, const double s, const double k)
      : Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   InterfaceDGDiffusionIntegrator(MatrixCoefficient &q, const double s, const double k)
      : Q(NULL), MQ(&q), sigma(s), kappa(k) { }

   virtual void AssembleInterfaceMatrix(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Trans1,
                                        FaceElementTransformations &Trans2,
                                        Array2D<DenseMatrix*> &elmats);
};

} // namespace mfem

#endif
