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

   //    Mult(dshape, trans.InverseJacobian(), dshapex);

   //    w = ip.weight;

   //    if (Q)
   //    {
   //       w *= Q->Eval(trans, ip);
   //    }

   //    MultAtB(EF, dshapex, gradEF);
      

   //    trans.AdjugateJacobian().Mult(vec1, vec2);

   //    vec2 *= w;
   //    dshape.Mult(vec2, vec3);
   //    MultVWt(shape, vec3, elmat_comp);

   //    for (int ii = 0; ii < dim; ii++)
   //    {
   //       elmat.AddMatrix(elmat_comp, ii * nd, ii * nd);
   //    }

   //    MultVVt(shape, elmat_comp);
   //    w = ip.weight * trans.Weight();
   //    if (Q)
   //    {
   //       w *= Q->Eval(trans, ip);
   //    }
   //    for (int ii = 0; ii < dim; ii++)
   //    {
   //       for (int jj = 0; jj < dim; jj++)
   //       {
   //          elmat.AddMatrix(w * gradEF(ii, jj), elmat_comp, ii * nd, jj * nd);
   //       }
   //    }
   }
}

}
