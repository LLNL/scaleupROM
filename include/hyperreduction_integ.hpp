// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_HYPERREDUCTION_INTEG_HPP
#define SCALEUPROM_HYPERREDUCTION_INTEG_HPP

#include "mfem.hpp"
#include "hdf5_utils.hpp"
#include "etc.hpp"

namespace mfem
{

struct EQPSample {
   SampleInfo info;
   /* shape and gradient of shape function on element 1 */
   DenseMatrix *shape1 = NULL;
   Array<DenseMatrix *> dshape1;
   /* shape and gradient of shape function on element 2 */
   DenseMatrix *shape2 = NULL;
   Array<DenseMatrix *> dshape2;

   EQPSample(const SampleInfo &info_)
      : info(info_), shape1(NULL), shape2(NULL), dshape1(0), dshape2(0) {}

   ~EQPSample()
   {
      delete shape1;
      delete shape2;
      DeletePointers(dshape1);
      DeletePointers(dshape2);
   }
};

class EQPElement
{
public:
   Array<EQPSample *> samples;

public:
   EQPElement() : samples(0) {}

   EQPElement(const Array<SampleInfo> &samples_)
   {
      samples.SetSize(samples_.Size());
      for (int s = 0; s < samples.Size(); s++)
         samples[s] = new EQPSample(samples_[s]);
   }

   ~EQPElement()
   {
      DeletePointers(samples);
   }

   const int Size() { return samples.Size(); }

   EQPSample* GetSample(const int s)
   {
      assert((s >= 0) && (s < samples.Size()));
      return samples[s];
   }

   void Save(hid_t file_id, const std::string &dsetname, const IntegratorType type);
   void Load(hid_t file_id, const std::string &dsetname, const IntegratorType type);
};

class HyperReductionIntegrator : virtual public NonlinearFormIntegrator
{
protected:
   HyperReductionIntegrator(const IntegrationRule *ir = NULL)
      : NonlinearFormIntegrator(ir) {}

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

   virtual void AppendPrecomputeDomainCoeffs(const FiniteElementSpace *fes,
                                             DenseMatrix &basis,
                                             const SampleInfo &sample);
   virtual void AppendPrecomputeInteriorFaceCoeffs(const FiniteElementSpace *fes,
                                                   DenseMatrix &basis,
                                                   const SampleInfo &sample);
   virtual void AppendPrecomputeBdrFaceCoeffs(const FiniteElementSpace *fes,
                                             DenseMatrix &basis,
                                             const SampleInfo &sample);

   virtual void AddAssembleVector_Fast(const int s, const EQPSample &eqp_sample,
                                       ElementTransformation &T, const Vector &x, Vector &y);
   virtual void AddAssembleVector_Fast(const int s, const EQPSample &eqp_sample,
                                       FaceElementTransformations &T, const Vector &x, Vector &y);
   virtual void AddAssembleGrad_Fast(const int s, const EQPSample &eqp_sample,
                                     ElementTransformation &T, const Vector &x, DenseMatrix &jac);
   virtual void AddAssembleGrad_Fast(const int s, const EQPSample &eqp_sample,
                                     FaceElementTransformations &T, const Vector &x, DenseMatrix &jac);
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
      // : HyperReductionIntegrator(true), Q(&q), vQ(vq), coeffs(0) { }
      : HyperReductionIntegrator(), Q(&q), vQ(vq) { }

   VectorConvectionTrilinearFormIntegrator() = default;

   ~VectorConvectionTrilinearFormIntegrator() {}

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

   void AddAssembleVector_Fast(const int s, const EQPSample &eqp_sample, 
                              ElementTransformation &T, const Vector &x, Vector &y) override;
   void AddAssembleGrad_Fast(const int s, const EQPSample &eqp_sample, 
                              ElementTransformation &T, const Vector &x, DenseMatrix &jac) override;
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
      : HyperReductionIntegrator(), Q(&q) { }

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

   void AssembleQuadratureVector(const FiniteElement &el,
                                 ElementTransformation &T,
                                 const IntegrationPoint &ip,
                                 const double &iw,
                                 const Vector &eltest,
                                 Vector &elquad) override;

   void AssembleQuadratureGrad(const FiniteElement &el,
                              ElementTransformation &trans,
                              const IntegrationPoint &ip,
                              const double &iw,
                              const Vector &elfun,
                              DenseMatrix &elmat);

   void AddAssembleVector_Fast(const int s, const EQPSample &eqp_sample, 
                                 ElementTransformation &T, const Vector &x, Vector &y);
   void AddAssembleGrad_Fast(const int s, const EQPSample &eqp_sample, 
                              ElementTransformation &T, const Vector &x, DenseMatrix &jac);
};

} // namespace mfem

#endif
