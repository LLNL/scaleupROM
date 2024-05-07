// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_HYPERREDUCTION_INTEG_HPP
#define SCALEUPROM_HYPERREDUCTION_INTEG_HPP

#include "mfem.hpp"

namespace mfem
{

struct SampleInfo {
   int el;         // element index (used for DomainIntegrator)
   int face;       // face index (used for InteriorFaceIntegrator)
   int be;         // boundary element index (used for BdrFaceIntegrator)
   int qp;         // quadrature point
   double qw;      // quadrature weight
   // can add dofs for other hyper reductions.
};

class HyperReductionIntegrator : virtual public NonlinearFormIntegrator
{
public:
   const bool precomputable;

protected:
   HyperReductionIntegrator(const bool precomputable_ = false, const IntegrationRule *ir = NULL)
      : precomputable(precomputable_), NonlinearFormIntegrator(ir) {}

   // removed const qualifier for basis in order to use its column view vector.
   void GetBasisElement(DenseMatrix &basis, const int col,
                        const Array<int> vdofs, Vector &basis_el,
                        DofTransformation *dof_trans = NULL);

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

   virtual void AssembleQuadratureGrad(const FiniteElement &el,
                                       ElementTransformation &T,
                                       const IntegrationPoint &ip,
                                       const double &iw,
                                       const Vector &eltest,
                                       DenseMatrix &quadmat);

   virtual void AssembleQuadratureGrad(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &T,
                                       const IntegrationPoint &ip,
                                       const double &iw,
                                       const Vector &eltest,
                                       DenseMatrix &quadmat);

   virtual void AppendPrecomputeCoefficients(const FiniteElementSpace *fes,
                                             DenseMatrix &basis,
                                             const SampleInfo &sample);

   virtual void AddAssembleVector_Fast(const int s, const double qw,
                                       ElementTransformation &T, const IntegrationPoint &ip,
                                       const Vector &x, Vector &y);
   virtual void AddAssembleVector_Fast(const int s, const double qw,
                                       FaceElementTransformations &T, const IntegrationPoint &ip,
                                       const Vector &x, Vector &y);
   virtual void AddAssembleGrad_Fast(const int s, const double qw,
                                     ElementTransformation &T, const IntegrationPoint &ip,
                                     const Vector &x, DenseMatrix &jac);
   virtual void AddAssembleGrad_Fast(const int s, const double qw,
                                     FaceElementTransformations &T, const IntegrationPoint &ip,
                                     const Vector &x, DenseMatrix &jac);
};

class VectorConvectionTrilinearFormIntegrator : virtual public HyperReductionIntegrator
{
private:
   int dim;
   Coefficient *Q{};
   VectorCoefficient *vQ{};
   DenseMatrix dshape, dshapex, elmat_comp, EF, gradEF, ELV;
   Vector shape;

   // // DenseTensor is column major and i is the fastest index. 
   // // For fast iteration, we set k to be the test function index.
   Array<DenseTensor *> coeffs;
   Array<DenseMatrix *> shapes;
   Array<Array<DenseMatrix *> *> dshapes;

   // Tensor precomputation is more expensive.
   bool tensor = false;

public:
   VectorConvectionTrilinearFormIntegrator(Coefficient &q, VectorCoefficient *vq = NULL)
      // : HyperReductionIntegrator(true), Q(&q), vQ(vq), coeffs(0) { }
      : HyperReductionIntegrator(true), Q(&q), vQ(vq), shapes(0), dshapes(0), coeffs(0) { }

   VectorConvectionTrilinearFormIntegrator() = default;

   ~VectorConvectionTrilinearFormIntegrator()
   // { for (int k = 0; k < coeffs.Size(); k++) delete coeffs[k]; }
   {
      for (int k = 0; k < shapes.Size(); k++) delete shapes[k];
      for (int k = 0; k < dshapes.Size(); k++)
      {
         for (int kk = 0; kk < dshapes[k]->Size(); kk++)
            delete (*dshapes[k])[kk];
         delete dshapes[k];
      }
   }

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

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);

   virtual void AssembleQuadratureGrad(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const IntegrationPoint &ip,
                                       const double &iw,
                                       const Vector &elfun,
                                       DenseMatrix &elmat) override;

   virtual void AppendPrecomputeCoefficients(const FiniteElementSpace *fes,
                                             DenseMatrix &basis,
                                             const SampleInfo &sample) override;

   virtual void AddAssembleVector_Fast(const int s, const double qw,
                                       ElementTransformation &T, const IntegrationPoint &ip,
                                       const Vector &x, Vector &y) override;
   virtual void AddAssembleGrad_Fast(const int s, const double qw,
                                     ElementTransformation &T, const IntegrationPoint &ip,
                                     const Vector &x, DenseMatrix &jac) override;
};

/*
   < \nabla v, uu > domain integrator
*/
class IncompressibleInviscidFluxNLFIntegrator :
   public HyperReductionIntegrator
{
private:
   int dim;
   Coefficient *Q{};
   DenseMatrix dshape, dshapex, EF, uu, ELV, elmat_comp;
   Vector shape;

public:
   IncompressibleInviscidFluxNLFIntegrator(Coefficient &q)
      : HyperReductionIntegrator(true), Q(&q) { }

   IncompressibleInviscidFluxNLFIntegrator() = default;

   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &trans,
                              const Vector &elfun,
                              Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                           ElementTransformation &trans,
                           const Vector &elfun,
                           DenseMatrix &elmat) override;
};

/*
   Interior face, interface integrator
   < [v], {uu \dot n} + \Lambda * [u] >
   \Lambda = max( | u- \dot n |, | u+ \dot n | )
*/
class DGLaxFriedrichsFluxIntegrator : public HyperReductionIntegrator
{
private:
   int dim, ndofs1, ndofs2, nvdofs;
   double w, un1, un2, un;
   Coefficient *Q{};

   Vector nor, flux, shape1, shape2, u1, u2;
   DenseMatrix udof1, udof2, elv1, elv2;
   DenseMatrix elmat_comp11, elmat_comp12, elmat_comp21, elmat_comp22;
public:
   DGLaxFriedrichsFluxIntegrator(Coefficient &q)
      : HyperReductionIntegrator(true), Q(&q) {}

   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
    void AssembleFaceGrad(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, DenseMatrix &elmat) override;

};

} // namespace mfem

#endif
