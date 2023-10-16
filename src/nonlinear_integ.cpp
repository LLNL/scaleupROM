// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "nonlinear_integ.hpp"

using namespace std;

namespace mfem
{

void IncompressibleInviscidFluxNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &T,
   const Vector &elfun,
   Vector &elvect)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elvect.SetSize(nd * dim);
   uu.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector u1(dim);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      double w = ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }

      // MultAtB(EF, dshape, gradEF);
      EF.MultTranspose(shape, u1);
      MultVVt(u1, uu);
      
      AddMult_a(w, dshape, uu, ELV);
   }
}

void IncompressibleInviscidFluxNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   DenseMatrix &elmat)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   dshapex.SetSize(nd, dim);
   elmat.SetSize(nd * dim);
   elmat_comp.SetSize(nd);
   // gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(nd), vec3(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      // el.CalcDShape(ip, dshape);
      el.CalcPhysDShape(trans, dshape);

      EF.MultTranspose(shape, vec1);

      double w = ip.weight * trans.Weight();
      if (Q) { w *= Q->Eval(trans, ip); }

      vec1 *= w;

      dshape.Mult(vec1, vec2);
      MultVWt(vec2, shape, elmat_comp);
      for (int d = 0; d < dim; d++)
      {
         elmat.AddMatrix(elmat_comp, d * nd, d * nd);
      }

      for (int di = 0; di < dim; di++)
      {
         dshapex.Set(vec1(di), dshape);
         for (int dj = 0; dj < dim; dj++)
         {
            dshapex.GetColumnReference(dj, vec3);
            MultVWt(vec3, shape, elmat_comp);
            elmat.AddMatrix(elmat_comp, di * nd, dj * nd);
         }
      }
   }
}

void DGLaxFriedrichsFluxIntegrator::AssembleFaceVector(const FiniteElement &el1,
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
   nh.SetSize(dim);
   un.SetSize(dim);
   uu.SetSize(dim);

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

      w = ip.weight;
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= 0.5 * w;
      // if (ndofs2)
      // nh.Set(0.5, nor);
      // else
      nh = nor;

      MultVVt(u1, uu);
      if (ndofs2)
         AddMultVVt(u2, uu);

      uu.Mult(nh, un);
      AddMultVWt(shape1, un, elv1);
      if (ndofs2)
         AddMult_a_VWt(-1.0, shape2, un, elv2);

      un1 = nor * u1;
      un2 = (ndofs2) ? nor * u2 : 0.0;
      un1 = max(abs(un1), abs(un2));

      u1 *= un1;
      if (ndofs2)
         u1.Add(-un1, u2);

      AddMultVWt(shape1, u1, elv1);
      if (ndofs2)
         AddMult_a_VWt(-1.0, shape2, u1, elv2);
   }
}

void DGLaxFriedrichsFluxIntegrator::AssembleFaceGrad(const FiniteElement &el1,
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
   nh.SetSize(dim);
   un.SetSize(dim);
   uu.SetSize(dim);

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
      elmat_comp22.SetSize(ndofs2);
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

      w = ip.weight;
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= 0.5 * w;
      // if (ndofs2)
      // nh.Set(0.5, nor);
      // else
      nh = nor;

      un1 = nh * u1;
      un2 = (ndofs2) ? nh * u2 : 0.0;

      MultVVt(shape1, elmat_comp11);
      if (ndofs2)
      {
         MultVWt(shape1, shape2, elmat_comp12);
         elmat_comp21.Transpose(elmat_comp12);
         // MultVWt(shape2, shape1, elmat_comp21);
         MultVVt(shape2, elmat_comp22);
      }

      for (int di = 0; di < dim; di++)
      {
         elmat.AddMatrix(un1, elmat_comp11, di * ndofs1, di * ndofs1);
         if (ndofs2)
         {
            elmat.AddMatrix(-un1, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
            elmat.AddMatrix(un2, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);
            elmat.AddMatrix(-un2, elmat_comp22, di * ndofs2 + dim * ndofs1, di * ndofs2 + dim * ndofs1);
         }

         for (int dj = 0; dj < dim; dj++)
         {
            elmat.AddMatrix(u1(di) * nh(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
            if (ndofs2)
            {
               elmat.AddMatrix(-u1(di) * nh(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
               elmat.AddMatrix(u2(di) * nh(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
               elmat.AddMatrix(-u2(di) * nh(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
            }
         }
      }

      // un1 as maximum absolute eigenvalue, un2 as the sign.
      un1 = nor * u1;
      un2 = (ndofs2) ? nor * u2 : 0.0;
      bool u1_lg_u2 = (abs(un1) >= abs(un2));
      if (u1_lg_u2)
      {
         un2 = (un1 >= 0.0) ? 1.0 : -1.0;
         un1 = abs(un1);
      }
      else
      {
         un1 = abs(un2);
         un2 = (un2 >= 0.0) ? 1.0 : -1.0;
      }

      // [ u ] = u1 - u2
      if (ndofs2)
         u1.Add(-1.0, u2);

      for (int di = 0; di < dim; di++)
      {
         elmat.AddMatrix(un1, elmat_comp11, di * ndofs1, di * ndofs1);
         if (ndofs2)
         {
            elmat.AddMatrix(-un1, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
            elmat.AddMatrix(-un1, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);
            elmat.AddMatrix(un1, elmat_comp22, di * ndofs2 + dim * ndofs1, di * ndofs2 + dim * ndofs1);
         }

         for (int dj = 0; dj < dim; dj++)
         {
            if (u1_lg_u2)
            {
               elmat.AddMatrix(un2 * u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
               if (ndofs2)
                  elmat.AddMatrix(-un2 * u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
            }
            else
            {
               assert(ndofs2);
               elmat.AddMatrix(un2 * u1(di) * nor(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
               elmat.AddMatrix(-un2 * u1(di) * nor(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
            }
         }
      }
   }
}

}
