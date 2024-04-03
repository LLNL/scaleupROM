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
                                 const Vector &elfun, Vector &elvect){
       DenseMatrix elmat(elfun.Size());
       AssembleFaceMatrix(el1, el2, Tr, elmat);
       elmat.Mult(elfun, elvect);
                                 };

void DGHyperelasticNLFIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Tr,
                              const Vector &elfun, DenseMatrix &elmat){
       AssembleFaceMatrix(el1, el2, Tr, elmat);
                              };

void DGHyperelasticNLFIntegrator::AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset,
       const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
       const Vector &row_shape, const Vector &col_shape,
       const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
       DenseMatrix &elmat, DenseMatrix &jmat){
for (int jm = 0, j = col_offset; jm < dim; ++jm)
    {
       for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
       {
          const double t2 = col_dshape_dnM(jdof);
          for (int im = 0, i = row_offset; im < dim; ++im)
          {
             const double t1 = col_dshape(jdof, jm) * col_nL(im);
             const double t3 = col_dshape(jdof, im) * col_nM(jm);
             const double tt = t1 + ((im == jm) ? t2 : 0.0) + t3;
             for (int idof = 0; idof < row_ndofs; ++idof, ++i)
             {
                elmat(i, j) += row_shape(idof) * tt;
             }
          }
       }
    }
 
    if (jmatcoef == 0.0) { return; }
 
    for (int d = 0; d < dim; ++d)
    {
       const int jo = col_offset + d*col_ndofs;
       const int io = row_offset + d*row_ndofs;
       for (int jdof = 0, j = jo; jdof < col_ndofs; ++jdof, ++j)
       {
          const double sj = jmatcoef * col_shape(jdof);
          for (int i = max(io,j), idof = i - io; idof < row_ndofs; ++idof, ++i)
          {
             jmat(i, j) += row_shape(idof) * sj;
          }
       }
    }
};

void DGHyperelasticNLFIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat){
#ifdef MFEM_THREAD_SAFE
    // For descriptions of these variables, see the class declaration.
    Vector shape1, shape2;
    DenseMatrix dshape1, dshape2;
    DenseMatrix adjJ;
    DenseMatrix dshape1_ps, dshape2_ps;
    Vector nor;
    Vector nL1, nL2;
    Vector nM1, nM2;
    Vector dshape1_dnM, dshape2_dnM;
    DenseMatrix jmat;
 #endif
 
    const int dim = el1.GetDim();
    const int ndofs1 = el1.GetDof();
    const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;
    const int nvdofs = dim*(ndofs1 + ndofs2);
 
    // Initially 'elmat' corresponds to the term:
    //    < { sigma(u) . n }, [v] > =
    //    < { (lambda div(u) I + mu (grad(u) + grad(u)^T)) . n }, [v] >
    // But eventually, it's going to be replaced by:
    //    elmat := -elmat + alpha*elmat^T + jmat
    elmat.SetSize(nvdofs);
    elmat = 0.;
 
    const bool kappa_is_nonzero = (kappa != 0.0);
    if (kappa_is_nonzero)
    {
       jmat.SetSize(nvdofs);
       jmat = 0.;
    }
 
    adjJ.SetSize(dim);
    shape1.SetSize(ndofs1);
    dshape1.SetSize(ndofs1, dim);
    dshape1_ps.SetSize(ndofs1, dim);
    nor.SetSize(dim);
    nL1.SetSize(dim);
    nM1.SetSize(dim);
    dshape1_dnM.SetSize(ndofs1);
 
    if (ndofs2)
    {
       shape2.SetSize(ndofs2);
       dshape2.SetSize(ndofs2, dim);
       dshape2_ps.SetSize(ndofs2, dim);
       nL2.SetSize(dim);
       nM2.SetSize(dim);
       dshape2_dnM.SetSize(ndofs2);
    }
 
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       // a simple choice for the integration order; is this OK?
       const int order = 2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0);
       ir = &IntRules.Get(Trans.GetGeometryType(), order);
    }
 
    for (int pind = 0; pind < ir->GetNPoints(); ++pind)
    {
       const IntegrationPoint &ip = ir->IntPoint(pind);
 
       // Set the integration point in the face and the neighboring elements
       Trans.SetAllIntPoints(&ip);
 
       // Access the neighboring elements' integration points
       // Note: eip2 will only contain valid data if Elem2 exists
       const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
       const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();
 
       el1.CalcShape(eip1, shape1);
       el1.CalcDShape(eip1, dshape1);
 
       CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
       Mult(dshape1, adjJ, dshape1_ps);
 
       if (dim == 1)
       {
          nor(0) = 2*eip1.x - 1.0;
       }
       else
       {
          CalcOrtho(Trans.Jacobian(), nor);
       }
 
       double w, wLM;
       if (ndofs2)
       {
          el2.CalcShape(eip2, shape2);
          el2.CalcDShape(eip2, dshape2);
          CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
          Mult(dshape2, adjJ, dshape2_ps);
 
          w = ip.weight/2;
          const double w2 = w / Trans.Elem2->Weight();
          const double wL2 = w2 * lambda->Eval(*Trans.Elem2, eip2);
          const double wM2 = w2 * mu->Eval(*Trans.Elem2, eip2);
          nL2.Set(wL2, nor);
          nM2.Set(wM2, nor);
          wLM = (wL2 + 2.0*wM2);
          dshape2_ps.Mult(nM2, dshape2_dnM);
       }
       else
       {
          w = ip.weight;
          wLM = 0.0;
       }
 
       {
          const double w1 = w / Trans.Elem1->Weight();
          const double wL1 = w1 * lambda->Eval(*Trans.Elem1, eip1);
          const double wM1 = w1 * mu->Eval(*Trans.Elem1, eip1);
          nL1.Set(wL1, nor);
          nM1.Set(wM1, nor);
          wLM += (wL1 + 2.0*wM1);
          dshape1_ps.Mult(nM1, dshape1_dnM);
       }
 
       const double jmatcoef = kappa * (nor*nor) * wLM;
 
       // (1,1) block
       AssembleBlock(
          dim, ndofs1, ndofs1, 0, 0, jmatcoef, nL1, nM1,
          shape1, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);
 
       if (ndofs2 == 0) { continue; }
 
       // In both elmat and jmat, shape2 appears only with a minus sign.
       shape2.Neg();
 
       // (1,2) block
       AssembleBlock(
          dim, ndofs1, ndofs2, 0, dim*ndofs1, jmatcoef, nL2, nM2,
          shape1, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
       // (2,1) block
       AssembleBlock(
          dim, ndofs2, ndofs1, dim*ndofs1, 0, jmatcoef, nL1, nM1,
          shape2, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);
       // (2,2) block
       AssembleBlock(
          dim, ndofs2, ndofs2, dim*ndofs1, dim*ndofs1, jmatcoef, nL2, nM2,
          shape2, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
    }
 
    // elmat := -elmat + alpha*elmat^t + jmat
    if (kappa_is_nonzero)
    {
       for (int i = 0; i < nvdofs; ++i)
       {
          for (int j = 0; j < i; ++j)
          {
             double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
             elmat(i,j) = alpha*aji - aij + mij;
             elmat(j,i) = alpha*aij - aji + mij;
          }
          elmat(i,i) = (alpha - 1.)*elmat(i,i) + jmat(i,i);
       }
    }
    else
    {
       for (int i = 0; i < nvdofs; ++i)
       {
          for (int j = 0; j < i; ++j)
          {
             double aij = elmat(i,j), aji = elmat(j,i);
             elmat(i,j) = alpha*aji - aij;
             elmat(j,i) = alpha*aij - aji;
          }
          elmat(i,i) *= (alpha - 1.);
       }
    }
};

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
      elvect.SetSize(elfun.Size());
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
                                        Vector &elvect){
       mfem_error("DGElasticityDirichletLFIntegrator::AssembleRHSElementVect");
       };

void DGHyperelasticDirichletNLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect){
MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");
 
 #ifdef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix adjJ;
    DenseMatrix dshape_ps;
    Vector nor;
    Vector dshape_dn;
    Vector dshape_du;
    Vector u_dir;
 #endif
 
    const int dim = el.GetDim();
    const int ndofs = el.GetDof();
    const int nvdofs = dim*ndofs;
 
    elvect.SetSize(nvdofs);
    elvect = 0.0;
 
    adjJ.SetSize(dim);
    shape.SetSize(ndofs);
    dshape.SetSize(ndofs, dim);
    dshape_ps.SetSize(ndofs, dim);
    nor.SetSize(dim);
    dshape_dn.SetSize(ndofs);
    dshape_du.SetSize(ndofs);
    u_dir.SetSize(dim);
 
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       const int order = 2*el.GetOrder(); // <-----
       ir = &IntRules.Get(Tr.GetGeometryType(), order);
    }
 
    for (int pi = 0; pi < ir->GetNPoints(); ++pi)
    {
       const IntegrationPoint &ip = ir->IntPoint(pi);
 
       // Set the integration point in the face and the neighboring element
       Tr.SetAllIntPoints(&ip);
 
       // Access the neighboring element's integration point
       const IntegrationPoint &eip = Tr.GetElement1IntPoint();
 
       // Evaluate the Dirichlet b.c. using the face transformation.
       uD.Eval(u_dir, Tr, ip);
 
       el.CalcShape(eip, shape);
       el.CalcDShape(eip, dshape);
 
       CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
       Mult(dshape, adjJ, dshape_ps);
 
       if (dim == 1)
       {
          nor(0) = 2*eip.x - 1.0;
       }
       else
       {
          CalcOrtho(Tr.Jacobian(), nor);
       }
 
       double wL, wM, jcoef;
       {
          const double w = ip.weight / Tr.Elem1->Weight();
          wL = w * lambda->Eval(*Tr.Elem1, eip);
          wM = w * mu->Eval(*Tr.Elem1, eip);
          jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
          dshape_ps.Mult(nor, dshape_dn);
          dshape_ps.Mult(u_dir, dshape_du);
       }
 
       // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
       //   + kappa < h^{-1} (lambda + 2 mu) uD, v >
 
       // i = idof + ndofs * im
       // v_phi(i,d) = delta(im,d) phi(idof)
       // div(v_phi(i)) = dphi(idof,im)
       // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
       //
       // term 1:
       //   alpha < uD, lambda div(v_phi(i)) n >
       //   alpha lambda div(v_phi(i)) (uD.n) =
       //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
       //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
       //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
       // term 2:
       //   < alpha uD, mu grad(v_phi(i)).n > =
       //   alpha mu uD^T grad(v_phi(i)) n =
       //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
       //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
       //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
       //   alpha * wM * u_dir(im) * dshape_dn(idof)
       // term 3:
       //   < alpha uD, mu (grad(v_phi(i)))^T n > =
       //   alpha mu n^T grad(v_phi(i)) uD =
       //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
       //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
       //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
       //   alpha * wM * nor(im) * dshape_du(idof)
       // term j:
       //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
       //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
       //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
       //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
       //      [ 1/h = |nor|/det(J1) ]
       //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
       //   jcoef * u_dir(im) * shape(idof)
 
       wM *= alpha;
       const double t1 = alpha * wL * (u_dir*nor);
       for (int im = 0, i = 0; im < dim; ++im)
       {
          const double t2 = wM * u_dir(im);
          const double t3 = wM * nor(im);
          const double tj = jcoef * u_dir(im);
          for (int idof = 0; idof < ndofs; ++idof, ++i)
          {
             elvect(i) += (t1*dshape_ps(idof,im) + t2*dshape_dn(idof) +
                           t3*dshape_du(idof) + tj*shape(idof));
          }
       }
    }
};

} // namespace mfem
