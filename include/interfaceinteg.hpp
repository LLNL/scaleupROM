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

   /** Abstract method used for assembling InteriorFaceIntegrators in a
       MixedBilinearFormDGExtension. */
   virtual void AssembleInterfaceMatrix(const FiniteElement &trial_fe1,
                                       const FiniteElement &trial_fe2,
                                       const FiniteElement &test_fe1,
                                       const FiniteElement &test_fe2,
                                       FaceElementTransformations &Trans1,
                                       FaceElementTransformations &Trans2,
                                       Array2D<DenseMatrix*> &elmats)
   { mfem_error("Abstract method InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix!\n"); }
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

// DGDiffusionFaceIntegrator
class InterfaceDGVectorDiffusionIntegrator : public InterfaceNonlinearFormIntegrator
{
public:
   InterfaceDGVectorDiffusionIntegrator(double alpha_, double kappa_)
      : mu(NULL), alpha(alpha_), kappa(kappa_) { }

   InterfaceDGVectorDiffusionIntegrator(Coefficient &mu_,
                              double alpha_, double kappa_)
      : mu(&mu_), alpha(alpha_), kappa(kappa_) { }

   virtual void AssembleInterfaceMatrix(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Trans1,
                                        FaceElementTransformations &Trans2,
                                        Array2D<DenseMatrix*> &elmats);

   using InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix;

protected:
   Coefficient *mu;
   double alpha, kappa;

   // values of all scalar basis functions for one component of u (which is a
   // vector) at the integration point in the reference space
   Vector shape1, shape2;
   // values of derivatives of all scalar basis functions for one component
   // of u (which is a vector) at the integration point in the reference space
   DenseMatrix dshape1, dshape2;
   // Adjugate of the Jacobian of the transformation: adjJ = det(J) J^{-1}
   DenseMatrix adjJ;
   // gradient of shape functions in the real (physical, not reference)
   // coordinates, scaled by det(J):
   //    dshape_ps(jdof,jm) = sum_{t} adjJ(t,jm)*dshape(jdof,t)
   DenseMatrix dshape1_ps, dshape2_ps;
   Vector nor;  // nor = |weight(J_face)| n
   Vector nM1, nM2;  // nM1 = (mu1     * ip.weight / detJ1) nor
   Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
   // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
   Array2D<DenseMatrix*> jmats;

   // Since elmats are already blocked out, we do not need the offsets.
   // offsets are used only for jmat, to determine the lower-triangular part.
   static void AssembleBlock(
      const int dim, const int row_ndofs, const int col_ndofs,
      const int row_offset, const int col_offset, const double jmatcoef,
      const Vector &row_shape, const Vector &col_shape, const Vector &col_dshape_dnM,
      DenseMatrix &elmat, DenseMatrix &jmat);
};

class InterfaceDGNormalFluxIntegrator : public InterfaceNonlinearFormIntegrator
{
private:
   int dim;
   int order;
   int p;

   int trial_dof1, trial_dof2, test_dof1, test_dof2;
   int trial_vdof1, trial_vdof2;

   double w, wn;
   int i, j, idof, jdof, jm;

   Vector nor, wnor;
   Vector shape1, shape2;
   // Vector divshape;
   Vector trshape1, trshape2;
   // DenseMatrix vshape1, vshape2;
   // Vector vshape1_n, vshape2_n;

public:
   InterfaceDGNormalFluxIntegrator() {};
   virtual ~InterfaceDGNormalFluxIntegrator() {};

   virtual void AssembleInterfaceMatrix(const FiniteElement &trial_fe1,
                                       const FiniteElement &trial_fe2,
                                       const FiniteElement &test_fe1,
                                       const FiniteElement &test_fe2,
                                       FaceElementTransformations &Trans1,
                                       FaceElementTransformations &Trans2,
                                       Array2D<DenseMatrix*> &elmats);

   using InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix;
};

} // namespace mfem

#endif
