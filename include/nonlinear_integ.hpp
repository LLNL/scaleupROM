// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_NONLINEAR_INTEG_HPP
#define SCALEUPROM_NONLINEAR_INTEG_HPP

#include "mfem.hpp"

namespace mfem
{

/*
   TrilinearForm domain integrator that computes Temam's domain integral:
   For the trial velocity u and the test velocity v,
      0.5 * (u_l * (d/dx_l) u_m * v_m - u_l * d/dx_l v_m * u_m)
*/
class TemamTrilinearFormIntegrator :
   public NonlinearFormIntegrator
{
private:
   int dim;
   Coefficient *Q{};
   VectorCoefficient *vQ{};
   DenseMatrix dshape, dshapex, EF, gradEF, ELV, elmat_comp, elmat_comp2;
   Vector shape;

public:
   TemamTrilinearFormIntegrator(Coefficient &q, VectorCoefficient *vq = NULL)
      : Q(&q), vQ(vq) { }

   TemamTrilinearFormIntegrator() = default;

   static const IntegrationRule& GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
};

/*
   TrilinearForm face integrator that computes Temam's flux:
   For the trial velocity u^1, u^2 and the test velocity v^1, v^2 on the interior face with the normal vector n^1,
      0.5 * (u^1_l * n^1_l * (u^1_m - u^2_m) * v^1_m - u^1_l * n^1_l * (v^1_m - v^2_m) * u^1_m)
   =  0.5 * u^1_l * n^1_l * (u^1_m * v^2_m - u^2_m * v^1_m)
   If it's used on the boundary, then
      0.5 * u^1_l * n^1_l * u^1_m * v^1_m.
   A scalar coefficient can be multiplied if specified via Coefficient *Q.
*/
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
