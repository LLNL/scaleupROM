// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nlelast_integ.hpp"

using namespace std;
namespace mfem
{
 // Boundary integrator
void DGHyperelasticNLFIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, Vector &elvect){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Tr,
                              const Vector &elfun, DenseMatrix &elmat){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset,
       const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
       const Vector &row_shape, const Vector &col_shape,
       const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
       DenseMatrix &elmat, DenseMatrix &jmat){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat){MFEM_ABORT("not implemented")};

// Domain integrator
void HyperelasticNLFIntegratorHR::AssembleElementMatrix(const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
       int dof = el.GetDof();
       int dim = el.GetDim();
       double w, L, M;

       MFEM_ASSERT(dim == Trans.GetSpaceDim(), "");

#ifdef MFEM_THREAD_SAFE
       DenseMatrix dshape(dof, dim), gshape(dof, dim), pelmat(dof);
       Vector divshape(dim * dof);
#else
       dshape.SetSize(dof, dim);
       gshape.SetSize(dof, dim);
       pelmat.SetSize(dof);
       divshape.SetSize(dim * dof);
#endif

       elmat.SetSize(dof * dim);

       const IntegrationRule *ir = IntRule;
       if (ir == NULL)
       {
              int order = 2 * Trans.OrderGrad(&el); // correct order?
              ir = &IntRules.Get(el.GetGeomType(), order);
       }

       elmat = 0.0;

       for (int i = 0; i < ir->GetNPoints(); i++)
       {
              const IntegrationPoint &ip = ir->IntPoint(i);

              el.CalcDShape(ip, dshape);

              Trans.SetIntPoint(&ip);
              w = ip.weight * Trans.Weight();
              Mult(dshape, Trans.InverseJacobian(), gshape);
              MultAAt(gshape, pelmat);
              gshape.GradToDiv(divshape);

              M = mu->Eval(Trans, ip);
              if (lambda)
              {
                     L = lambda->Eval(Trans, ip);
              }
              else
              {
                     L = q_lambda * M;
                     M = q_mu * M;
              }

              if (L != 0.0)
              {
                     AddMult_a_VVt(L * w, divshape, elmat);
              }

              if (M != 0.0)
              {
                     for (int d = 0; d < dim; d++)
                     {
                            for (int k = 0; k < dof; k++)
                                   for (int l = 0; l < dof; l++)
                                   {
                                          elmat(dof * d + k, dof * d + l) += (M * w) * pelmat(k, l);
                                   }
                     }
                     for (int ii = 0; ii < dim; ii++)
                            for (int jj = 0; jj < dim; jj++)
                            {
                                   for (int kk = 0; kk < dof; kk++)
                                          for (int ll = 0; ll < dof; ll++)
                                          {
                                                 elmat(dof * ii + kk, dof * jj + ll) +=
                                                     (M * w) * gshape(kk, jj) * gshape(ll, ii);
                                          }
                            }
              }
       }
};

void HyperelasticNLFIntegratorHR::AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect){
       DenseMatrix elmat(elfun.Size());
       AssembleElementMatrix(el, trans, elmat);
       elmat.Mult(elfun, elvect);
       }

void HyperelasticNLFIntegratorHR::AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat){
AssembleElementMatrix(el, trans, elmat);
                                    };
 
void HyperelasticNLFIntegratorHR::ComputeElementFlux(const FiniteElement &el,
                                    ElementTransformation &Trans,
                                    Vector &u,
                                    const FiniteElement &fluxelem,
                                    Vector &flux, bool with_coef,
                                    const IntegrationRule *ir){
    const int dof = el.GetDof();
    const int dim = el.GetDim();
    const int tdim = dim*(dim+1)/2; // num. entries in a symmetric tensor
    double L, M;
 
    MFEM_ASSERT(dim == 2 || dim == 3,
                "dimension is not supported: dim = " << dim);
    MFEM_ASSERT(dim == Trans.GetSpaceDim(), "");
    MFEM_ASSERT(fluxelem.GetMapType() == FiniteElement::VALUE, "");
    MFEM_ASSERT(dynamic_cast<const NodalFiniteElement*>(&fluxelem), "");
 
 #ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape(dof, dim);
 #else
    dshape.SetSize(dof, dim);
 #endif
 
    double gh_data[9], grad_data[9];
    DenseMatrix gh(gh_data, dim, dim);
    DenseMatrix grad(grad_data, dim, dim);
 
    if (!ir)
    {
       ir = &fluxelem.GetNodes();
    }
    const int fnd = ir->GetNPoints();
    flux.SetSize(fnd * tdim);
 
    DenseMatrix loc_data_mat(u.GetData(), dof, dim);
    for (int i = 0; i < fnd; i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       el.CalcDShape(ip, dshape);
       MultAtB(loc_data_mat, dshape, gh);
 
       Trans.SetIntPoint(&ip);
       Mult(gh, Trans.InverseJacobian(), grad);
 
       M = mu->Eval(Trans, ip);
       if (lambda)
       {
          L = lambda->Eval(Trans, ip);
       }
       else
       {
          L = q_lambda * M;
          M = q_mu * M;
       }
 
       // stress = 2*M*e(u) + L*tr(e(u))*I, where
       //   e(u) = (1/2)*(grad(u) + grad(u)^T)
       const double M2 = 2.0*M;
       if (dim == 2)
       {
          L *= (grad(0,0) + grad(1,1));
          // order of the stress entries: s_xx, s_yy, s_xy
          flux(i+fnd*0) = M2*grad(0,0) + L;
          flux(i+fnd*1) = M2*grad(1,1) + L;
          flux(i+fnd*2) = M*(grad(0,1) + grad(1,0));
       }
       else if (dim == 3)
       {
          L *= (grad(0,0) + grad(1,1) + grad(2,2));
          // order of the stress entries: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
          flux(i+fnd*0) = M2*grad(0,0) + L;
          flux(i+fnd*1) = M2*grad(1,1) + L;
          flux(i+fnd*2) = M2*grad(2,2) + L;
          flux(i+fnd*3) = M*(grad(0,1) + grad(1,0));
          flux(i+fnd*4) = M*(grad(0,2) + grad(2,0));
          flux(i+fnd*5) = M*(grad(1,2) + grad(2,1));
       }
    }
};
 
double HyperelasticNLFIntegratorHR::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                     ElementTransformation &Trans,
                                     Vector &flux, Vector *d_energy){
    const int dof = fluxelem.GetDof();
    const int dim = fluxelem.GetDim();
    const int tdim = dim*(dim+1)/2; // num. entries in a symmetric tensor
    double L, M;
 
    // The MFEM_ASSERT constraints in ElasticityIntegrator::ComputeElementFlux
    // are assumed here too.
    MFEM_ASSERT(d_energy == NULL, "anisotropic estimates are not supported");
    MFEM_ASSERT(flux.Size() == dof*tdim, "invalid 'flux' vector");
 
 #ifndef MFEM_THREAD_SAFE
    shape.SetSize(dof);
 #else
    Vector shape(dof);
 #endif
    double pointstress_data[6];
    Vector pointstress(pointstress_data, tdim);
 
    // View of the 'flux' vector as a (dof x tdim) matrix
    DenseMatrix flux_mat(flux.GetData(), dof, tdim);
 
    // Use the same integration rule as in AssembleElementMatrix, replacing 'el'
    // with 'fluxelem' when 'IntRule' is not set.
    // Should we be using a different (more accurate) rule here?
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       int order = 2 * Trans.OrderGrad(&fluxelem);
       ir = &IntRules.Get(fluxelem.GetGeomType(), order);
    }
 
    double energy = 0.0;
 
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       fluxelem.CalcShape(ip, shape);
 
       flux_mat.MultTranspose(shape, pointstress);
 
       Trans.SetIntPoint(&ip);
       double w = Trans.Weight() * ip.weight;
 
       M = mu->Eval(Trans, ip);
       if (lambda)
       {
          L = lambda->Eval(Trans, ip);
       }
       else
       {
          L = q_lambda * M;
          M = q_mu * M;
       }
 
       // The strain energy density at a point is given by (1/2)*(s : e) where s
       // and e are the stress and strain tensors, respectively. Since we only
       // have the stress, we need to compute the strain from the stress:
       //    s = 2*mu*e + lambda*tr(e)*I
       // Taking trace on both sides we find:
       //    tr(s) = 2*mu*tr(e) + lambda*tr(e)*dim = (2*mu + dim*lambda)*tr(e)
       // which gives:
       //    tr(e) = tr(s)/(2*mu + dim*lambda)
       // Then from the first identity above we can find the strain:
       //    e = (1/(2*mu))*(s - lambda*tr(e)*I)
 
       double pt_e; // point strain energy density
       const double *s = pointstress_data;
       if (dim == 2)
       {
          // s entries: s_xx, s_yy, s_xy
          const double tr_e = (s[0] + s[1])/(2*(M + L));
          L *= tr_e;
          pt_e = (0.25/M)*(s[0]*(s[0] - L) + s[1]*(s[1] - L) + 2*s[2]*s[2]);
       }
       else // (dim == 3)
       {
          // s entries: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
          const double tr_e = (s[0] + s[1] + s[2])/(2*M + 3*L);
          L *= tr_e;
          pt_e = (0.25/M)*(s[0]*(s[0] - L) + s[1]*(s[1] - L) + s[2]*(s[2] - L) +
                           2*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]));
       }
 
       energy += w * pt_e;
    }
 
    return energy;
};
//RHS
void DGHyperelasticDirichletNLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect){MFEM_ABORT("not implemented")};

void DGHyperelasticDirichletNLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect){MFEM_ABORT("not implemented")};

} // namespace mfem
