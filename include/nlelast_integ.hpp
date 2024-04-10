// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_NLELAST_INTEG_HPP
#define SCALEUPROM_NLELAST_INTEG_HPP

#include "mfem.hpp"
#include "hyperreduction_integ.hpp"

namespace mfem
{

 class TestLinModel //: public HyperelasticModel
 {
 protected:
    mutable double mu, lambda;
    Coefficient *c_mu, *c_lambda;
    ElementTransformation *Ttr;
 
 public:
    TestLinModel(double mu_, double lambda_)
       : mu(mu_), lambda(lambda_) { c_mu = new ConstantCoefficient(mu), c_lambda = new ConstantCoefficient(lambda); }
 
    void SetTransformation(ElementTransformation &Ttr_) { Ttr = &Ttr_; }
    double EvalW(const DenseMatrix &J);
 
    void EvalP(const FiniteElement &el, const IntegrationPoint &ip, const DenseMatrix &PMatI, FaceElementTransformations &Trans, DenseMatrix &P);
    void EvalP(const FiniteElement &el, const IntegrationPoint &ip, const DenseMatrix &PMatI, ElementTransformation &Trans, DenseMatrix &P);
    
    void EvalDmat(const int dim, const int dof, const IntegrationPoint ip, FaceElementTransformations &Trans, const DenseMatrix gshape, DenseMatrix &Dmat);
    void EvalDmat(const int dim, const int dof, const IntegrationPoint ip, ElementTransformation &Trans, const DenseMatrix gshape, DenseMatrix &Dmat);
    
    void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                     const double w, DenseMatrix &elmat, const FiniteElement &el, const IntegrationPoint &ip,ElementTransformation &Trans);
 };
 
// DG boundary integrator for nonlinear elastic DG.
// For this is just DGElasticityIntegrator with a different name
class DGHyperelasticNLFIntegrator : virtual public HyperReductionIntegrator 
 {

 public:
    DGHyperelasticNLFIntegrator(TestLinModel *m, double alpha_, double kappa_)
       : HyperReductionIntegrator(false), model(m), lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) { }
 
   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   const Vector &elfun, Vector &elvect);

   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat);
        // values of all scalar basis functions for one component of u (which is a
    protected:
   TestLinModel *model;
    Coefficient *lambda, *mu;
    double alpha, kappa;// vector) at the integration point in the reference space
 #ifndef MFEM_THREAD_SAFE
    Vector elvect1, elvect2;
    Vector elfun1, elfun2;

   DenseMatrix Jrt;
    DenseMatrix PMatI1, PMatO1, NorMat1, DSh1, DS1, Jpt1, P1;
    DenseMatrix PMatI2, PMatO2, NorMat2, DSh2, DS2, Jpt2, P2;
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
    Vector nL1, nL2;  // nL1 = (lambda1 * ip.weight / detJ1) nor
    Vector nM1, nM2;  // nM1 = (mu1     * ip.weight / detJ1) nor
    Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
    // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
    DenseMatrix jmat;
 #endif
 
    static void AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset, const Vector &row_shape,
       const Vector &wnor, const DenseMatrix &Dmat, DenseMatrix &elmat);

 };

// Domain integrator for nonlinear elastic DG.
class HyperelasticNLFIntegratorHR : virtual public HyperReductionIntegrator 
 {
 
 private:
 TestLinModel *model;
    //   Jrt: the Jacobian of the target-to-reference-element transformation.
    //   Jpr: the Jacobian of the reference-to-physical-element transformation.
    //   Jpt: the Jacobian of the target-to-physical-element transformation.
    //     P: represents dW_d(Jtp) (dim x dim).
    //   DSh: gradients of reference shape functions (dof x dim).
    //    DS: gradients of the shape functions in the target (stress-free)
    //        configuration (dof x dim).
    // PMatI: coordinates of the deformed configuration (dof x dim).
    // PMatO: reshaped view into the local element contribution to the operator
    //        output - the result of AssembleElementVector() (dof x dim).
    DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;
 
 public:
    HyperelasticNLFIntegratorHR(TestLinModel *m): HyperReductionIntegrator(false), model(m)
    { }

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);

 };

// RHS integrator for nonlinear elastic DG.
// For this is just DGElasticityDirichletLFIntegrator with a different name
  class DGHyperelasticDirichletNLFIntegrator : public LinearFormIntegrator // Should this be a nonlinear form later?
 {
 protected:
    VectorCoefficient &uD;
    Coefficient *lambda, *mu;
    double alpha, kappa;
 
 #ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix adjJ;
    DenseMatrix dshape_ps;
    Vector nor;
    Vector dshape_dn;
    Vector dshape_du;
    Vector u_dir;
 #endif
 
 public:
    DGHyperelasticDirichletNLFIntegrator(VectorCoefficient &uD_,
                                      Coefficient &lambda_, Coefficient &mu_,
                                      double alpha_, double kappa_)
       : uD(uD_), lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }
 
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
