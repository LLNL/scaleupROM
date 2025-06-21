// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_INTERFACE_INTEGRATOR_HPP
#define SCALEUPROM_INTERFACE_INTEGRATOR_HPP

#include "mfem.hpp"
#include "hyperreduction_integ.hpp"
#include "etc.hpp"

namespace mfem
{

class InterfaceNonlinearFormIntegrator : virtual public HyperReductionIntegrator
{
protected:
   InterfaceNonlinearFormIntegrator(const IntegrationRule *ir = NULL)
      : HyperReductionIntegrator(ir) {}
public:
   // FaceElementTransformations belongs to one mesh (having mesh pointer).
   // In order to extract element/transformation from each mesh,
   // two FaceElementTransformations are needed.
   virtual void AssembleInterfaceVector(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr1,
                                       FaceElementTransformations &Tr2,
                                       const Vector &elfun1, const Vector &elfun2,
                                       Vector &elvect1, Vector &elvect2);

   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   virtual void AssembleInterfaceGrad(const FiniteElement &el1,
                                      const FiniteElement &el2,
                                      FaceElementTransformations &Tr1,
                                      FaceElementTransformations &Tr2,
                                      const Vector &elfun1, const Vector &elfun2,
                                      Array2D<DenseMatrix*> &elmats);

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

   virtual void AssembleQuadratureVector(const FiniteElement &el1,
                                          const FiniteElement &el2,
                                          FaceElementTransformations &Tr1,
                                          FaceElementTransformations &Tr2,
                                          const IntegrationPoint &ip,
                                          const double &iw,
                                          const Vector &eltest1, const Vector &eltest2,
                                          Vector &elquad1, Vector &elquad2);

   virtual void AssembleQuadratureGrad(const FiniteElement &el1,
                                          const FiniteElement &el2,
                                          FaceElementTransformations &Tr1,
                                          FaceElementTransformations &Tr2,
                                          const IntegrationPoint &ip,
                                          const double &iw,
                                          const Vector &eltest1, const Vector &eltest2,
                                          Array2D<DenseMatrix*> &quadmats);

   virtual void AddAssembleVector_Fast(const EQPSample &eqp_sample,
                                       FaceElementTransformations &Tr1,
                                       FaceElementTransformations &Tr2,
                                       const Vector &x1, const Vector &x2,
                                       Vector &y1, Vector &y2);

   virtual void AddAssembleGrad_Fast(const EQPSample &eqp_sample,
                                     FaceElementTransformations &Tr1,
                                     FaceElementTransformations &Tr2,
                                     const Vector &x1, const Vector &x2,
                                     Array2D<SparseMatrix *> &jac);
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

class InterfaceDGTemamFluxIntegrator : public InterfaceNonlinearFormIntegrator
{
private:
   int dim, ndofs1, ndofs2, nvdofs1, nvdofs2;
   double w, un;
   Coefficient *Q{};

   Vector nor, shape1, shape2, u1, u2, flux;
   DenseMatrix udof1, udof2, elv1, elv2;
   DenseMatrix elmat_comp11, elmat_comp12, elmat_comp21;

public:
   InterfaceDGTemamFluxIntegrator(Coefficient &q) : Q(&q) {};
   virtual ~InterfaceDGTemamFluxIntegrator() {};

   virtual void AssembleInterfaceVector(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr1,
                                       FaceElementTransformations &Tr2,
                                       const Vector &elfun1, const Vector &elfun2,
                                       Vector &elvect1, Vector &elvect2);

   virtual void AssembleInterfaceGrad(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr1,
                                       FaceElementTransformations &Tr2,
                                       const Vector &elfun1, const Vector &elfun2,
                                       Array2D<DenseMatrix *> &elmats);
};

class InterfaceDGElasticityIntegrator : public InterfaceNonlinearFormIntegrator
{
protected:
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
   Vector nor; // nor = |weight(J_face)| n
   Vector nM1, nM2;   // nM1 = (mu1     * ip.weight / detJ1) nor
   Vector nL1, nL2;
   Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
   // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
   Array2D<DenseMatrix *> jmats;

   Coefficient *lambda, *mu;

   static void AssembleBlock(const int dim, const int row_ndofs,
                              const int col_ndofs, const int row_offset, const int col_offset, const double jmatcoef,
                              const Vector &col_nL, const Vector &col_nM, const Vector &row_shape, const Vector &col_shape,
                              const Vector &col_dshape_dnM, const DenseMatrix &col_dshape, DenseMatrix &elmat, DenseMatrix &jmat);
public:
   InterfaceDGElasticityIntegrator(Coefficient *lambda_, Coefficient *mu_, double alpha_, double kappa_) : lambda(lambda_), mu(mu_), alpha(alpha_), kappa(kappa_) {};


         virtual void AssembleInterfaceMatrix(const FiniteElement &el1,
                                             const FiniteElement &el2,
                                             FaceElementTransformations &Trans1,
                                             FaceElementTransformations &Trans2,
                                             Array2D<DenseMatrix *> &elmats);

   using InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix;
};

/*
   Interior face, interface integrator
   < [v], {uu \dot n} + \Lambda / 2 * [u] >
   \Lambda = max( 2 * | u- \dot n |, 2 * | u+ \dot n | )

   For boundary face,
   (i) Neumann condition (UD == NULL)
   < v-, (u-)(u-) \dot n >
   (ii) Dirichlet condition (UD != NULL)
   Same formulation, u+ = UD
*/
class DGLaxFriedrichsFluxIntegrator : public InterfaceNonlinearFormIntegrator
{
private:
   int dim;
   double un1, un2, un;
   double u1mag, u2mag, normag;
   Vector flux, u1, u2, nor;

   int ndofs1, ndofs2, nvdofs;
   double w;
   Coefficient *Q{};
   VectorCoefficient *UD = NULL;

   Vector shape1, shape2;
   DenseMatrix udof1, udof2, elv1, elv2;
   DenseMatrix elmat_comp11, elmat_comp12, elmat_comp21, elmat_comp22;

   mutable TimeProfiler timer;

public:
   DGLaxFriedrichsFluxIntegrator(Coefficient &q, VectorCoefficient *ud = NULL, const IntegrationRule *ir = NULL)
      : InterfaceNonlinearFormIntegrator(ir), Q(&q), UD(ud), timer("DGLaxFriedrichsFluxIntegrator") {}

   virtual ~DGLaxFriedrichsFluxIntegrator() {};

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

   void AssembleQuadratureVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &T,
                                 const IntegrationPoint &ip,
                                 const double &iw,
                                 const Vector &eltest,
                                 Vector &elquad) override;

   void AssembleQuadratureGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &T,
                              const IntegrationPoint &ip,
                              const double &iw,
                              const Vector &eltest,
                              DenseMatrix &quadmat) override;

   void AddAssembleVector_Fast(const EQPSample &eqp_sample,
                              FaceElementTransformations &T,
                              const Vector &x, Vector &y) override;
   void AddAssembleGrad_Fast(const EQPSample &eqp_sample,
                           FaceElementTransformations &T,
                           const Vector &x, DenseMatrix &jac) override;

   void AssembleInterfaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr1,
                                 FaceElementTransformations &Tr2,
                                 const Vector &elfun1, const Vector &elfun2,
                                 Vector &elvect1, Vector &elvect2) override;

   void AssembleInterfaceGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Tr1,
                              FaceElementTransformations &Tr2,
                              const Vector &elfun1, const Vector &elfun2,
                              Array2D<DenseMatrix *> &elmats) override;

   void AssembleQuadratureVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr1,
                                 FaceElementTransformations &Tr2,
                                 const IntegrationPoint &ip,
                                 const double &iw,
                                 const Vector &eltest1, const Vector &eltest2,
                                 Vector &elquad1, Vector &elquad2) override;

   void AssembleQuadratureGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr1,
                                 FaceElementTransformations &Tr2,
                                 const IntegrationPoint &ip,
                                 const double &iw,
                                 const Vector &eltest1, const Vector &eltest2,
                                 Array2D<DenseMatrix*> &quadmats) override;

   void AddAssembleVector_Fast(const EQPSample &eqp_sample,
                              FaceElementTransformations &Tr1,
                              FaceElementTransformations &Tr2,
                              const Vector &x1, const Vector &x2,
                              Vector &y1, Vector &y2) override;

   void AddAssembleGrad_Fast(const EQPSample &eqp_sample,
                           FaceElementTransformations &Tr1,
                           FaceElementTransformations &Tr2,
                           const Vector &x1, const Vector &x2,
                           Array2D<SparseMatrix *> &jac) override;

private:
   void ComputeFluxDotN(const Vector &u1, const Vector &u2, const Vector &nor,
                        const bool &eval2, Vector &flux);

   void ComputeGradFluxDotN(const Vector &u1, const Vector &u2, const Vector &nor,
                            const bool &eval2, const bool &ndofs2,
                            DenseMatrix &gradu1, DenseMatrix &gradu2);

   void AssembleQuadVectorBase(const FiniteElement &el1,
                               const FiniteElement &el2,
                               FaceElementTransformations *Tr1,
                               FaceElementTransformations *Tr2,
                               const IntegrationPoint &ip, const double &iw, const int &ndofs2,
                               const DenseMatrix &elfun1, const DenseMatrix &elfun2,
                               DenseMatrix &elvect1, DenseMatrix &elvect2);

   void AssembleQuadGradBase(const FiniteElement &el1, const FiniteElement &el2,
                             FaceElementTransformations *Tr1, FaceElementTransformations *Tr2,
                             const IntegrationPoint &ip, const double &iw, const int &ndofs2,
                             const DenseMatrix &elfun1, const DenseMatrix &elfun2,
                             double &w, DenseMatrix &gradu1, DenseMatrix &gradu2,
                             DenseMatrix &elmat11, DenseMatrix &elmat12, DenseMatrix &elmat21, DenseMatrix &elmat22);

};

} // namespace mfem

#endif
