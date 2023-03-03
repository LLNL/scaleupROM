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

#include "interfaceinteg.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;

namespace mfem
{

void InterfaceNonlinearFormIntegrator::AssembleInterfaceVector(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
  const Vector &elfun, Vector &elvect)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceVector\n"
             "   is not implemented for this class.");
}

void InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
  const Vector &elfun, DenseMatrix &elmat)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad\n"
             "   is not implemented for this class.");
}

void InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
  DenseMatrix &elmat)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad\n"
             "   is not implemented for this class.");
}

void InterfaceDGDiffusionIntegrator::AssembleInterfaceMatrix(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
  DenseMatrix &elmat)
{
   bool boundary = false;

   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;

   dim = el1.GetDim();

   nor.SetSize(dim);
   nor2.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   ndof1 = el1.GetDof();
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);

   // TODO: different boolean handling is needed (Elem2No will be zero for both Trans1 and Trans2).
   if (boundary) {
   // if (Trans.Elem2No >= 0)
     ndof2 = 0;
   } else {
     ndof2 = el2.GetDof();
     shape2.SetSize(ndof2);
     dshape2.SetSize(ndof2, dim);
     dshape2dn.SetSize(ndof2);
   }

   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2)
      {
         order = 2*max(el1.GetOrder(), el2.GetOrder());
      }
      else
      {
         order = 2*el1.GetOrder();
      }
      ir = &IntRules.Get(Trans1.GetGeometryType(), order);

      assert(Trans1.GetGeometryType() == Trans2.GetGeometryType());
   }

   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      // printf("ip\t%.3f\t%.3f\t%.3f\n", ip.x, ip.y, ip.z);

      // Set the integration point in the face and the neighboring elements
      Trans1.SetAllIntPoints(&ip);
      Trans2.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip1 and eip2 come from Element1 of Trans1 and Trans2 respectively.
      const IntegrationPoint &eip1 = Trans1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans2.GetElement1IntPoint();
      // printf("\tx\ty\tz\n");
      // printf("eip1\t%.3f\t%.3f\t%.3f\n", eip1.x, eip1.y, eip1.z);
      // printf("eip2\t%.3f\t%.3f\t%.3f\n", eip2.x, eip2.y, eip2.z);

      // computing outward normal vectors.
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
         nor2(0) = 2*eip2.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans1.Jacobian(), nor);
         CalcOrtho(Trans2.Jacobian(), nor2);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);
      w = ip.weight/Trans1.Elem1->Weight();
      if (ndof2)
      {
         w /= 2;
      }
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Trans1.Elem1, eip1);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Trans1.Elem1, eip1);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Trans1.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);
      if (kappa_is_nonzero)
      {
         wq = ni * nor;
      }
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      dshape1.Mult(nh, dshape1dn);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            elmat(i, j) += shape1(i) * dshape1dn(j);
         }

      if (ndof2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         // Trans2 is also boundary face and only has Elem1.
         w = ip.weight/2/Trans2.Elem1->Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans2.Elem1, eip2);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans2.Elem1, eip2);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans2.Elem1->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq += ni * nor;
         }

         dshape2.Mult(nh, dshape2dn);

         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
            }

         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa;
         for (int i = 0; i < ndof1; i++)
         {
            const double wsi = wq*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               jmat(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const int i2 = ndof1 + i;
               const double wsi = wq*shape2(i);
               for (int j = 0; j < ndof1; j++)
               {
                  jmat(i2, j) -= wsi * shape1(j);
               }
               for (int j = 0; j <= i; j++)
               {
                  jmat(i2, ndof1 + j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
            elmat(i,j) = sigma*aji - aij + mij;
            elmat(j,i) = sigma*aij - aji + mij;
         }
         elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
      }
   }
   else
   {
      for (int i = 0; i < ndofs; i++)
      {
         for (int j = 0; j < i; j++)
         {
            double aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = sigma*aji - aij;
            elmat(j,i) = sigma*aij - aji;
         }
         elmat(i,i) *= (sigma - 1.);
      }
   }
}

}
