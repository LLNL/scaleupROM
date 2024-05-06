// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_NLELAST_INTEG_HPP
#define SCALEUPROM_NLELAST_INTEG_HPP

#include "mfem.hpp"
#include "hyperreduction_integ.hpp"

namespace mfem
{

   /// Abstract class for hyperelastic models that work with DG methods
   class DGHyperelasticModel
   {
   protected:
      ElementTransformation *Ttr; /**< Reference-element to target-element
                                       transformation. */

   public:
      DGHyperelasticModel() : Ttr(NULL) {}
      virtual ~DGHyperelasticModel() {}

      void SetTransformation(ElementTransformation &Ttr_) { Ttr = &Ttr_; }
      virtual double EvalW(const DenseMatrix &Jpt) const = 0;
      virtual void EvalP(const DenseMatrix &Jpt, DenseMatrix &P) const = 0;
      virtual void SetMatParam(ElementTransformation &Trans, const IntegrationPoint &ip)const = 0;
      virtual void SetMatParam(FaceElementTransformations &Trans, const IntegrationPoint &ip)const = 0;
      virtual void GetMatParam(Vector &params) const = 0;

      virtual double EvalDGWeight(const double w, ElementTransformation &Ttr, const IntegrationPoint &ip) const = 0;

      virtual void EvalDmat(const int dim, const int dof, const DenseMatrix &DS, const DenseMatrix &J, DenseMatrix &Dmat)const = 0;

   };

   class LinElastMaterialModel : public DGHyperelasticModel
   {
   protected:
      Coefficient *c_mu, *c_lambda;
      mutable double mu, lambda;

   public:
      LinElastMaterialModel(double mu_, double lambda_)
          : mu(mu_), lambda(lambda_) { c_mu = new ConstantCoefficient(mu), c_lambda = new ConstantCoefficient(lambda); }

      virtual void SetMatParam(ElementTransformation &Trans, const IntegrationPoint &ip) const;
      virtual void SetMatParam(FaceElementTransformations &Trans, const IntegrationPoint &ip) const;
      virtual void GetMatParam(Vector &params) const;
      virtual double EvalW(const DenseMatrix &J) const;

      virtual double EvalDGWeight(const double w, ElementTransformation &Ttr, const IntegrationPoint &ip) const;
      virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

      virtual void EvalDmat(const int dim, const int dof, const DenseMatrix &DS, const DenseMatrix &J, DenseMatrix &Dmat) const;


   };

   class NeoHookeanHypModel : public DGHyperelasticModel
   {
   protected:
      mutable double mu, K, g;
      Coefficient *c_mu, *c_K, *c_g;
      ElementTransformation *Ttr;
      //DenseMatrix E, S;
      mutable DenseMatrix Z;    // dim x dim
      mutable DenseMatrix G, C; // dof x dim

   public:
      NeoHookeanHypModel(double mu_, double K_, double g_ = 1.0)
          : mu(mu_), K(K_), g(g_) { c_mu = new ConstantCoefficient(mu), c_K = new ConstantCoefficient(K), c_g = new ConstantCoefficient(g); }

     virtual void SetMatParam(ElementTransformation &Trans, const IntegrationPoint &ip) const;
      virtual void SetMatParam(FaceElementTransformations &Trans, const IntegrationPoint &ip) const;
      virtual void GetMatParam(Vector &params) const;
      virtual double EvalW(const DenseMatrix &J) const;

      virtual double EvalDGWeight(const double w, ElementTransformation &Ttr, const IntegrationPoint &ip) const;
      virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const;

      virtual void EvalDmat(const int dim, const int dof, const DenseMatrix &DS, const DenseMatrix &J, DenseMatrix &Dmat) const;
   };

   // DG boundary integrator for nonlinear elastic DG.
   // For this is just DGElasticityIntegrator with a different name
   class DGHyperelasticNLFIntegrator : virtual public HyperReductionIntegrator
   {

   public:
      DGHyperelasticNLFIntegrator(DGHyperelasticModel *m, double alpha_, double kappa_)
          : HyperReductionIntegrator(false), model(m), lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) {}

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
      DGHyperelasticModel *model;
      Coefficient *lambda, *mu;
      double alpha, kappa; // vector) at the integration point in the reference space
      Vector elvect1, elvect2;
      Vector elfun1, elfun2;

      DenseMatrix Jrt;
      DenseMatrix PMatI1, PMatO1, DSh1, DS1, Jpt1, adjJ1, P1, Dmat1;
      DenseMatrix PMatI2, PMatO2, DSh2, DS2, Jpt2, adjJ2, P2, Dmat2;
      Vector shape1, tau1,wnor1;
      Vector shape2, tau2,wnor2;
      
      Vector nor;                      // nor = |weight(J_face)| n
      // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
      DenseMatrix jmat;

      static void AssembleBlock(
          const int dim, const int row_ndofs, const int col_ndofs,
          const int row_offset, const int col_offset, const Vector &row_shape,
          const Vector &col_shape, const double jmatcoef,
          const Vector &wnor, const DenseMatrix &Dmat, DenseMatrix &elmat, DenseMatrix &jmat);

      static void AssembleJmat(
          const int dim, const int row_ndofs, const int col_ndofs,
          const int row_offset, const int col_offset, const Vector &row_shape,
          const Vector &col_shape, const double jmatcoef, DenseMatrix &jmat);
   };

   // Domain integrator for nonlinear elastic DG.
   class HyperelasticNLFIntegratorHR : virtual public HyperReductionIntegrator
   {

   private:
      DGHyperelasticModel *model;
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
      DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO, Dmat;

   public:
      HyperelasticNLFIntegratorHR(DGHyperelasticModel *m) : HyperReductionIntegrator(false), model(m)
      {
      }

      virtual void AssembleElementVector(const FiniteElement &el,
                                         ElementTransformation &trans,
                                         const Vector &elfun,
                                         Vector &elvect);

      virtual void AssembleElementGrad(const FiniteElement &el,
                                       ElementTransformation &trans,
                                       const Vector &elfun,
                                       DenseMatrix &elmat);

      virtual void AssembleH(const int dim, const int dof, const double w,
                             const DenseMatrix &J, DenseMatrix &elmat);
   };

   // RHS integrator for nonlinear elastic DG.
   // For this is just DGElasticityDirichletLFIntegrator with a different name
   class DGHyperelasticDirichletLFIntegrator : public LinearFormIntegrator // Should this be a nonlinear form later?
   {
   protected:
      VectorCoefficient &uD;
      DGHyperelasticModel *model;
      Coefficient *lambda, *mu;
      double alpha, kappa;
      Vector shape, nor, u_dir;

   public:
      DGHyperelasticDirichletLFIntegrator(VectorCoefficient &uD_,
                                           DGHyperelasticModel *m,
                                           double alpha_, double kappa_)
          : uD(uD_), model(m), lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) {}

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
