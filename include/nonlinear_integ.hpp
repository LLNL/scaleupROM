// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_NONLINEAR_INTEG_HPP
#define SCALEUPROM_NONLINEAR_INTEG_HPP

#include "mfem.hpp"

namespace mfem
{

class IncompressibleInviscidFluxNLFIntegrator :
   public VectorConvectionNLFIntegrator
{
private:
   int dim;
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, uu, ELV, elmat_comp;
   Vector shape;

public:
   IncompressibleInviscidFluxNLFIntegrator(Coefficient &q): Q(&q) { }

   IncompressibleInviscidFluxNLFIntegrator() = default;

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
};

class DGLaxFriedrichsFluxIntegrator : public NonlinearFormIntegrator
{
private:
   int dim, ndofs1, ndofs2, nvdofs;
   double w, un1, un2;
   Coefficient *Q{};

   Vector nor, nh, shape1, shape2, u1, u2, un;
   DenseMatrix udof1, udof2, elv1, elv2, uu;
   DenseMatrix elmat_comp11, elmat_comp12, elmat_comp21, elmat_comp22;
public:
   DGLaxFriedrichsFluxIntegrator(Coefficient &q) : Q(&q) {}

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat);

};

class DGTemamFluxIntegrator : public NonlinearFormIntegrator
{
private:
   int dim, ndofs1, ndofs2, nvdofs;
   double w, un;
   Coefficient *Q{};

   Vector nor, shape1, shape2, u1, u2, flux;
   DenseMatrix udof1, udof2, elv1, elv2;
   DenseMatrix elmat_comp11, elmat_comp12, elmat_comp21;
public:
   DGTemamFluxIntegrator(Coefficient &q) : Q(&q) {}

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat);

};

} // namespace mfem

#endif
