// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_NLELAST_INTEG_HPP
#define SCALEUPROM_NLELAST_INTEG_HPP

#include "mfem.hpp"
#include "hyperreduction_integ.hpp"

namespace mfem
{
// DG boundary integrator for nonlinear elastic DG.
// For this is just DGElasticityIntegrator with a different name
class DGHyperelasticNLFIntegrator : virtual public HyperReductionIntegrator 
 {
 public:
    DGHyperelasticNLFIntegrator(double alpha_, double kappa_)
       : HyperReductionIntegrator(false), lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) { }
 
    DGHyperelasticNLFIntegrator(Coefficient &lambda_, Coefficient &mu_,
                           double alpha_, double kappa_)
       : HyperReductionIntegrator(false), lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);

   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat);
 protected:
    Coefficient *lambda, *mu;
    double alpha, kappa;
 
 #ifndef MFEM_THREAD_SAFE
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
    Vector nL1, nL2;  // nL1 = (lambda1 * ip.weight / detJ1) nor
    Vector nM1, nM2;  // nM1 = (mu1     * ip.weight / detJ1) nor
    Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
    // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
    DenseMatrix jmat;
 #endif
 
    static void AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset,
       const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
       const Vector &row_shape, const Vector &col_shape,
       const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
       DenseMatrix &elmat, DenseMatrix &jmat);

    //using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat);
 };

// Domain integrator for nonlinear elastic DG.
// For this is just ElasticityIntegrator with a different name
// Later, this will just be HyperelasticNLFIntegrator
class HyperelasticNLFIntegratorHR : virtual public HyperReductionIntegrator 
 {
 protected:
    double q_lambda, q_mu;
    Coefficient *lambda, *mu;
    virtual void AssembleElementMatrix(const FiniteElement &,
                                       ElementTransformation &,
                                       DenseMatrix &);
 
 private:
 #ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape, gshape, pelmat;
    Vector divshape;
 #endif
 
 public:
    HyperelasticNLFIntegratorHR(Coefficient &l, Coefficient &m): HyperReductionIntegrator(false)
    { lambda = &l; mu = &m; }

    /** With this constructor lambda = q_l * m and mu = q_m * m;
        if dim * q_l + 2 * q_m = 0 then trace(sigma) = 0. */
    HyperelasticNLFIntegratorHR(Coefficient &m, double q_l, double q_m): HyperReductionIntegrator(false)
    { lambda = NULL; mu = &m; q_lambda = q_l; q_mu = q_m; }
 

   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat);
 
    virtual void ComputeElementFlux(const FiniteElement &el,
                                    ElementTransformation &Trans,
                                    Vector &u,
                                    const FiniteElement &fluxelem,
                                    Vector &flux, bool with_coef = true,
                                    const IntegrationRule *ir = NULL);
 
    virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                     ElementTransformation &Trans,
                                     Vector &flux, Vector *d_energy = NULL);
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
