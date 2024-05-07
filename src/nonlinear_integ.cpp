// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nonlinear_integ.hpp"

using namespace std;

namespace mfem
{

const IntegrationRule&
TemamTrilinearFormIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void TemamTrilinearFormIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &T,
   const Vector &elfun, Vector &elvect)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector u1(dim), udu(dim), vec1(nd);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      double w = 0.5 * ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }

      EF.MultTranspose(shape, u1);
      MultAtB(EF, dshape, gradEF);
      gradEF.Mult(u1, udu);
      udu *= w;
      AddMultVWt(shape, udu, ELV);

      dshape.Mult(u1, vec1);
      u1 *= -w;
      AddMultVWt(vec1, u1, ELV);
   }
}

void TemamTrilinearFormIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &trans,
   const Vector &elfun, DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   elmat_comp2.SetSize(nd);
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector u1(dim), vec2(nd), vec3(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      // el.CalcDShape(ip, dshape);
      el.CalcPhysDShape(trans, dshape);

      EF.MultTranspose(shape, u1);
      MultAtB(EF, dshape, gradEF);

      double w = 0.5 * ip.weight * trans.Weight();
      if (Q) { w *= Q->Eval(trans, ip); }

      dshape.Mult(u1, vec2);
      MultVWt(vec2, shape, elmat_comp);
      for (int d = 0; d < dim; d++)
         elmat.AddMatrix(-w, elmat_comp, d * nd, d * nd);

      MultVWt(shape, vec2, elmat_comp);
      for (int d = 0; d < dim; d++)
         elmat.AddMatrix(w, elmat_comp, d * nd, d * nd);

      MultVVt(shape, elmat_comp2);
      for (int di = 0; di < dim; di++)
      {
         dshapex.Set(u1(di), dshape);
         for (int dj = 0; dj < dim; dj++)
         {
            dshapex.GetColumnReference(dj, vec3);
            MultVWt(vec3, shape, elmat_comp);
            elmat.AddMatrix(-w, elmat_comp, di * nd, dj * nd);

            elmat.AddMatrix(w * gradEF(di, dj), elmat_comp2, di * nd, dj * nd);
         }
      }
   }
}

/*
   DGTemamFluxIntegrator
*/

void DGTemamFluxIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                                const FiniteElement &el2,
                                                FaceElementTransformations &Tr,
                                                const Vector &elfun, Vector &elvect)
{
   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = (Tr.Elem2No >= 0) ? el2.GetDof() : 0;
   nvdofs = dim * (ndofs1 + ndofs2);
   elvect.SetSize(nvdofs);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(elfun.GetData(), ndofs1, dim);
   elv1.UseExternalData(elvect.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);
   
   if (ndofs2)
   {
      udof2.UseExternalData(elfun.GetData() + ndofs1 * dim, ndofs2, dim);
      elv2.UseExternalData(elvect.GetData() + ndofs1 * dim, ndofs2, dim);
      shape2.SetSize(ndofs2);
      u2.SetSize(dim);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0) - 1)));
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      udof1.MultTranspose(shape1, u1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         udof2.MultTranspose(shape2, u2);
      }

      // 0.5 factor remains whether boundary or interior face.
      w = 0.5 * ip.weight;
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= w;
      un = nor * u1;
      flux.Set(un, u1);

      // If interior face, two flux terms cancel out.
      if (ndofs2)
      {
         AddMultVWt(shape2, flux, elv2);
         flux.Set(-un, u2);
         AddMultVWt(shape1, flux, elv1);
      }
      else
      {
         AddMultVWt(shape1, flux, elv1);
      }
   }
}

void DGTemamFluxIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                                             const FiniteElement &el2,
                                             FaceElementTransformations &Tr,
                                             const Vector &elfun, DenseMatrix &elmat)
{
   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = (Tr.Elem2No >= 0) ? el2.GetDof() : 0;
   nvdofs = dim * (ndofs1 + ndofs2);
   elmat.SetSize(nvdofs);
   elmat_comp11.SetSize(ndofs1);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(elfun.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);  

   if (ndofs2)
   {
      udof2.UseExternalData(elfun.GetData() + ndofs1 * dim, ndofs2, dim);
      shape2.SetSize(ndofs2);
      u2.SetSize(dim);

      elmat_comp12.SetSize(ndofs1, ndofs2);
      elmat_comp21.SetSize(ndofs2, ndofs1);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0) - 1)));
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   } 

   elmat = 0.0;
   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      udof1.MultTranspose(shape1, u1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         udof2.MultTranspose(shape2, u2);
      }

      // 0.5 factor remains whether boundary or interior face.
      w = 0.5 * ip.weight;
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= w;
      un = nor * u1;

      MultVVt(shape1, elmat_comp11);
      if (ndofs2)
      {
         MultVWt(shape1, shape2, elmat_comp12);
         elmat_comp21.Transpose(elmat_comp12);
      }

      if (ndofs2)
      {
         for (int di = 0; di < dim; di++)
         {
            elmat.AddMatrix(un, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
            elmat.AddMatrix(-un, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);

            for (int dj = 0; dj < dim; dj++)
            {
               elmat.AddMatrix(u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
               elmat.AddMatrix(-u2(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
            }
         }
      }
      else
      {
         for (int di = 0; di < dim; di++)
         {
            elmat.AddMatrix(un, elmat_comp11, di * ndofs1, di * ndofs1);

            for (int dj = 0; dj < dim; dj++)
               elmat.AddMatrix(u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
         }
      }
   }  // for (int pind = 0; pind < ir->GetNPoints(); ++pind)
}

}
