// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "dg_linear.hpp"

using namespace std;

namespace mfem
{

/*
    DGVectorDirichletLFIntegrator
*/

void DGVectorDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
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
         // wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         // jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         jcoef = kappa * wM * (nor*nor);
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
      // const double t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const double t2 = wM * u_dir(im);
         // const double t3 = wM * nor(im);
         const double tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t2*dshape_dn(idof) + tj*shape(idof));
         }
      }
   }
}

/*
    DGBoundaryNormalLFIntegrator
*/

void DGBoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int intorder = oa * el.GetOrder() + ob;  // <----------
      int intorder = 2 * el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      if (dim > 1)
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      else
      {
         nor[0] = 1.0;
      }
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

void DGBoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();  // <----------
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      Q.Eval(Qvec, *Tr.Elem1, eip);

      el.CalcShape(eip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

/*
    BoundaryNormalStressLFIntegrator
*/

void BoundaryNormalStressLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   Fvec.SetSize (dim * dim);
   Fn.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      if (dim > 1)
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      else
      {
         nor[0] = 1.0;
      }

      el.CalcShape (ip, shape);
      F.Eval(Fvec, Tr, ip);
      // column-major reshaping.
      Fmat.UseExternalData(Fvec.ReadWrite(), dim, dim);
      Fmat.Mult(nor, Fn);
      Fn *= ip.weight;
      for (int k = 0, j = 0; k < dim; k++)
         for (int jdof = 0; jdof < dof; jdof++, j++)
            elvect(j) += Fn(k) * shape(jdof);
   }
}

void BoundaryNormalStressLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   nor.SetSize (dim);
   Fvec.SetSize (dim * dim);
   Fn.SetSize (dim);

   elvect.SetSize (dim*dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      el.CalcShape (eip, shape);
      F.Eval(Fvec, *Tr.Face, ip);
      // column-major reshaping.
      Fmat.UseExternalData(Fvec.ReadWrite(), dim, dim);
      Fmat.Mult(nor, Fn);
      Fn *= ip.weight;
      for (int k = 0, j = 0; k < dim; k++)
         for (int jdof = 0; jdof < dof; jdof++, j++)
            elvect(j) += Fn(k) * shape(jdof);
   }
}

void DGBdrLaxFriedrichsLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec, gn(dim);
   double un;

   shape.SetSize(dof);
   elvect.SetSize(dim * dof);
   elvect = 0.0;

   ELV.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = (int) (ceil(1.5 * (2 * el.GetOrder() - 1)));  // <----------
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      Q.Eval(Qvec, *(Tr.Face), ip);

      el.CalcShape(eip, shape);

      un = 0.5 * (Qvec * nor);
      gn.Set(un, Qvec);

      // Need to check the signs.
      AddMult_a_VWt(-ip.weight, shape, gn, ELV);
      AddMult_a_VWt(ip.weight * abs(un), shape, Qvec, ELV);
   }
}

void DGBdrLaxFriedrichsLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec, gn(dim);
   double un;

   shape.SetSize(dof);
   elvect.SetSize(dim * dof);
   elvect = 0.0;

   ELV.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = (int) (ceil(1.5 * (2 * el.GetOrder() - 1)));  // <----------
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      if (dim == 1)
      {
         nor[0] = 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      un = 0.5 * (Qvec * nor);
      gn.Set(un, Qvec);

      // Need to check the signs.
      AddMult_a_VWt(-ip.weight, shape, gn, ELV);
      AddMult_a_VWt(ip.weight * abs(un), shape, Qvec, ELV);
   }
}

void DGBdrTemamLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec, gn(dim);
   double un;

   shape.SetSize(dof);
   elvect.SetSize(dim * dof);
   elvect = 0.0;

   ELV.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = (int) (ceil(1.5 * (2 * el.GetOrder() - 1)));  // <----------
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      w = ip.weight;
      if (Z) w *= Z->Eval(*(Tr.Face), eip);

      Q.Eval(Qvec, *(Tr.Face), eip);

      el.CalcShape(eip, shape);

      un = (Qvec * nor);
      gn.Set(un, Qvec);

      // Need to check the signs.
      AddMult_a_VWt(w, shape, gn, ELV);
   }
}

void DGBdrTemamLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec, gn(dim);
   double un;

   shape.SetSize(dof);
   elvect.SetSize(dim * dof);
   elvect = 0.0;

   ELV.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = (int) (ceil(1.5 * (2 * el.GetOrder() - 1)));  // <----------
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      if (dim == 1)
      {
         nor[0] = 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      w = ip.weight;
      if (Z) w *= Z->Eval(Tr, ip);

      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      un = (Qvec * nor);
      gn.Set(un, Qvec);

      // Need to check the signs.
      AddMult_a_VWt(w, shape, gn, ELV);
   }
}

}
