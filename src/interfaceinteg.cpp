// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "interfaceinteg.hpp"
#include "etc.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;

namespace mfem
{

void InterfaceNonlinearFormIntegrator::AssembleInterfaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
   const Vector &elfun1, const Vector &elfun2,
   Vector &elvect1, Vector &elvect2)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceVector\n"
             "   is not implemented for this class.");
}

void InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
   const Vector &elfun1, const Vector &elfun2,
   Array2D<DenseMatrix*> &elmats)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad\n"
             "   is not implemented for this class.");
}

void InterfaceNonlinearFormIntegrator::AssembleInterfaceMatrix(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
  Array2D<DenseMatrix*> &elmats)
{
   mfem_error("InterfaceNonlinearFormIntegrator::AssembleInterfaceGrad\n"
             "   is not implemented for this class.");
}

void InterfaceDGDiffusionIntegrator::AssembleInterfaceMatrix(
  const FiniteElement &el1, const FiniteElement &el2,
  FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
  Array2D<DenseMatrix*> &elmats)
{
   assert(elmats.NumRows() == 2);
   assert(elmats.NumCols() == 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) assert(elmats(i, j));

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
   // do not allow 0 ndof2 for now.
   assert(ndof2 > 0);

   ndofs = ndof1 + ndof2;
   elmats(0,0)->SetSize(ndof1, ndof1);
   elmats(0,1)->SetSize(ndof1, ndof2);
   elmats(1,0)->SetSize(ndof2, ndof1);
   elmats(1,1)->SetSize(ndof2, ndof2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) *elmats(i,j) = 0.0;
   // elmat.SetSize(ndofs);
   // elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmats.SetSize(2,2);
      jmats(0,0) = new DenseMatrix(ndof1, ndof1);
      // only the lower-triangular part of jmat is assembled.
      jmats(0,1) = NULL;
      jmats(1,0) = new DenseMatrix(ndof2, ndof1);
      jmats(1,1) = new DenseMatrix(ndof2, ndof2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j <= i; j++) *jmats(i,j) = 0.0;
      // jmat.SetSize(ndofs);
      // jmat = 0.;
   }

   DenseMatrix *elmat = NULL, *jmat = NULL;

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
      elmat = elmats(0,0);
      for (int i = 0; i < ndof1; i++)
         for (int j = 0; j < ndof1; j++)
         {
            (*elmat)(i, j) += shape1(i) * dshape1dn(j);
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

         elmat = elmats(0,1);
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof2; j++)
            {
               (*elmat)(i, j) += shape1(i) * dshape2dn(j);
            }

         elmat = elmats(1,0);
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof1; j++)
            {
               (*elmat)(i, j) -= shape2(i) * dshape1dn(j);
            }

         elmat = elmats(1,1);
         for (int i = 0; i < ndof2; i++)
            for (int j = 0; j < ndof2; j++)
            {
               (*elmat)(i, j) -= shape2(i) * dshape2dn(j);
            }
      }

      if (kappa_is_nonzero)
      {
         // only assemble the lower triangular part of jmat
         wq *= kappa;
         jmat = jmats(0,0);
         for (int i = 0; i < ndof1; i++)
         {
            const double wsi = wq*shape1(i);
            for (int j = 0; j <= i; j++)
            {
               (*jmat)(i, j) += wsi * shape1(j);
            }
         }
         if (ndof2)
         {
            for (int i = 0; i < ndof2; i++)
            {
               const double wsi = wq*shape2(i);
               jmat = jmats(1,0);
               for (int j = 0; j < ndof1; j++)
               {
                  (*jmat)(i, j) -= wsi * shape1(j);
               }
               jmat = jmats(1,1);
               for (int j = 0; j <= i; j++)
               {
                  (*jmat)(i, j) += wsi * shape2(j);
               }
            }
         }
      }
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   Array<int> ndof_array(2);
   ndof_array[0] = ndof1;
   ndof_array[1] = ndof2;
   DenseMatrix *elmat12 = NULL;
   if (kappa_is_nonzero)
   {
      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I,I);
         jmat = jmats(I,I); 
         for (int i = 0; i < ndof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i,j), aji = (*elmat)(j,i), mij = (*jmat)(i,j);
               (*elmat)(i,j) = sigma*aji - aij + mij;
               (*elmat)(j,i) = sigma*aij - aji + mij;
            }
            (*elmat)(i,i) = (sigma - 1.) * (*elmat)(i,i) + (*jmat)(i,i);
         }  // for (int i = 0; i < ndofs_array[I]; i++)
      }  // for (int I = 0; I < 2; I++)
      elmat = elmats(1,0);
      jmat = jmats(1,0);
      assert(jmat != NULL);
      elmat12 = elmats(0,1);
      for (int i = 0; i < ndof2; i++)
      {
         for (int j = 0; j < ndof1; j++)
         {
            double aij = (*elmat)(i,j), aji = (*elmat12)(j,i), mij = (*jmat)(i,j);
            (*elmat)(i,j) = sigma*aji - aij + mij;
            (*elmat12)(j,i) = sigma*aij - aji + mij;
         }  // for (int j = 0; j < ndofs1; j++)
      }  // for (int i = 0; i < ndofs2; i++)
   }  // if (kappa_is_nonzero)
   else
   {
      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I,I);
         for (int i = 0; i < ndof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i,j), aji = (*elmat)(j,i);
               (*elmat)(i,j) = sigma*aji - aij;
               (*elmat)(j,i) = sigma*aij - aji;
            }
            (*elmat)(i,i) *= (sigma - 1.);
         }  // for (int i = 0; i < ndofs_array[I]; i++)
      }  // for (int I = 0; I < 2; I++)
      elmat = elmats(1,0);
      elmat12 = elmats(0,1);
      for (int i = 0; i < ndof2; i++)
      {
         for (int j = 0; j < ndof1; j++)
         {
            double aij = (*elmat)(i,j), aji = (*elmat12)(j,i);
            (*elmat)(i,j) = sigma*aji - aij;
            (*elmat12)(j,i) = sigma*aij - aji;
         }  // for (int j = 0; j < ndofs1; j++)
      }  // for (int i = 0; i < ndofs2; i++)
   }  // not if (kappa_is_nonzero)

   if (kappa_is_nonzero)
      DeletePointers(jmats);
}

void InterfaceDGVectorDiffusionIntegrator::AssembleInterfaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
   Array2D<DenseMatrix*> &elmats)
{
   assert(elmats.NumRows() == 2);
   assert(elmats.NumCols() == 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) assert(elmats(i, j));

   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (Trans2.Elem1No >= 0) ? el2.GetDof() : 0;
   const int nvdofs1 = dim * ndofs1;
   const int nvdofs2 = dim * ndofs2;
   assert(ndofs2 > 0);

   // Initially 'elmat' corresponds to the term:
   //    < { mu grad(u) . n }, [v] >
   // But eventually, it's going to be replaced by:
   //    elmat := -elmat + alpha*elmat^T + jmat
   elmats(0,0)->SetSize(nvdofs1, nvdofs1);
   elmats(0,1)->SetSize(nvdofs1, nvdofs2);
   elmats(1,0)->SetSize(nvdofs2, nvdofs1);
   elmats(1,1)->SetSize(nvdofs2, nvdofs2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) *elmats(i,j) = 0.0;

   const bool kappa_is_nonzero = (kappa != 0.0);
   jmats.SetSize(2,2);
   if (kappa_is_nonzero)
   {
      jmats(0,0) = new DenseMatrix(nvdofs1, nvdofs1);
      // only the lower-triangular part of jmat is assembled.
      jmats(0,1) = NULL;
      jmats(1,0) = new DenseMatrix(nvdofs2, nvdofs1);
      jmats(1,1) = new DenseMatrix(nvdofs2, nvdofs2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j <= i; j++) *jmats(i,j) = 0.0;
      // jmat.SetSize(nvdofs);
      // jmat = 0.;
   }

   adjJ.SetSize(dim);
   shape1.SetSize(ndofs1);
   dshape1.SetSize(ndofs1, dim);
   dshape1_ps.SetSize(ndofs1, dim);
   nor.SetSize(dim);
   // nL1.SetSize(dim);
   nM1.SetSize(dim);
   dshape1_dnM.SetSize(ndofs1);

   if (ndofs2)
   {
      shape2.SetSize(ndofs2);
      dshape2.SetSize(ndofs2, dim);
      dshape2_ps.SetSize(ndofs2, dim);
      // nL2.SetSize(dim);
      nM2.SetSize(dim);
      dshape2_dnM.SetSize(ndofs2);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans1.GetGeometryType(), order);

      assert(Trans1.GetGeometryType() == Trans2.GetGeometryType());
   }

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans1.SetAllIntPoints(&ip);
      Trans2.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans2.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);

      CalcAdjugate(Trans1.Elem1->Jacobian(), adjJ);
      Mult(dshape1, adjJ, dshape1_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans1.Jacobian(), nor);
      }

      double w, wLM;
      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         CalcAdjugate(Trans2.Elem1->Jacobian(), adjJ);
         Mult(dshape2, adjJ, dshape2_ps);

         w = ip.weight/2;
         const double w2 = w / Trans2.Elem1->Weight();
         // const double wL2 = w2 * lambda->Eval(*Trans.Elem2, eip2);
         const double wM2 = w2 * mu->Eval(*Trans2.Elem1, eip2);
         // nL2.Set(wL2, nor);
         nM2.Set(wM2, nor);
         // wLM = (wL2 + 2.0*wM2);
         wLM = wM2;
         dshape2_ps.Mult(nM2, dshape2_dnM);
      }
      else
      {
         w = ip.weight;
         wLM = 0.0;
      }

      {
         const double w1 = w / Trans1.Elem1->Weight();
         // const double wL1 = w1 * lambda->Eval(*Trans.Elem1, eip1);
         const double wM1 = w1 * mu->Eval(*Trans1.Elem1, eip1);
         // nL1.Set(wL1, nor);
         nM1.Set(wM1, nor);
         // wLM += (wL1 + 2.0*wM1);
         wLM += wM1;
         dshape1_ps.Mult(nM1, dshape1_dnM);
      }

      const double jmatcoef = kappa * (nor*nor) * wLM;

      // (1,1) block
      AssembleBlock(
         dim, ndofs1, ndofs1, 0, 0, jmatcoef,
         shape1, shape1, dshape1_dnM, *elmats(0,0), *jmats(0,0));

      if (ndofs2 == 0) { continue; }

      // In both elmat and jmat, shape2 appears only with a minus sign.
      shape2.Neg();

      // (1,2) block
      // jmats(0,1) is never used. set jmatcoef = 0 for this case only.
      AssembleBlock(
         dim, ndofs1, ndofs2, 0, nvdofs1, 0.0,
         shape1, shape2, dshape2_dnM, *elmats(0,1), *jmats(0,1));
      // (2,1) block
      AssembleBlock(
         dim, ndofs2, ndofs1, nvdofs1, 0, jmatcoef,
         shape2, shape1, dshape1_dnM, *elmats(1,0), *jmats(1,0));
      // (2,2) block
      AssembleBlock(
         dim, ndofs2, ndofs2, nvdofs1, nvdofs1, jmatcoef,
         shape2, shape2, dshape2_dnM, *elmats(1,1), *jmats(1,1));
   }

   // elmat := -elmat + alpha*elmat^t + jmat
   Array<int> nvdof_array(2);
   nvdof_array[0] = nvdofs1;
   nvdof_array[1] = nvdofs2;
   DenseMatrix *elmat = NULL, *jmat = NULL;
   DenseMatrix *elmat12 = NULL;
   if (kappa_is_nonzero)
   {
      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I,I);
         jmat = jmats(I,I); 
         for (int i = 0; i < nvdof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i,j), aji = (*elmat)(j,i), mij = (*jmat)(i,j);
               (*elmat)(i,j) = alpha*aji - aij + mij;
               (*elmat)(j,i) = alpha*aij - aji + mij;
            }
            (*elmat)(i,i) = (alpha - 1.) * (*elmat)(i,i) + (*jmat)(i,i);
         }  // for (int i = 0; i < nvdofs_array[I]; i++)
      }  // for (int I = 0; I < 2; I++)
      elmat = elmats(1,0);
      jmat = jmats(1,0);
      assert(jmat != NULL);
      elmat12 = elmats(0,1);
      for (int i = 0; i < nvdofs2; i++)
      {
         for (int j = 0; j < nvdofs1; j++)
         {
            double aij = (*elmat)(i,j), aji = (*elmat12)(j,i), mij = (*jmat)(i,j);
            (*elmat)(i,j) = alpha*aji - aij + mij;
            (*elmat12)(j,i) = alpha*aij - aji + mij;
         }  // for (int j = 0; j < nvdofs1; j++)
      }  // for (int i = 0; i < nvdofs2; i++)
   }  // if (kappa_is_nonzero)
   else
   {
      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I,I);
         for (int i = 0; i < nvdof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i,j), aji = (*elmat)(j,i);
               (*elmat)(i,j) = alpha*aji - aij;
               (*elmat)(j,i) = alpha*aij - aji;
            }
            (*elmat)(i,i) *= (alpha - 1.);
         }  // for (int i = 0; i < nvdofs_array[I]; i++)
      }  // for (int I = 0; I < 2; I++)
      elmat = elmats(1,0);
      elmat12 = elmats(0,1);
      for (int i = 0; i < nvdofs2; i++)
      {
         for (int j = 0; j < nvdofs1; j++)
         {
            double aij = (*elmat)(i,j), aji = (*elmat12)(j,i);
            (*elmat)(i,j) = alpha*aji - aij;
            (*elmat12)(j,i) = alpha*aij - aji;
         }  // for (int j = 0; j < nvdofs1; j++)
      }  // for (int i = 0; i < nvdofs2; i++)
   }  // not if (kappa_is_nonzero)

   if (kappa_is_nonzero)
      DeletePointers(jmats);
}

// static method
void InterfaceDGVectorDiffusionIntegrator::AssembleBlock(
   const int dim, const int row_ndofs, const int col_ndofs,
   const int row_offset, const int col_offset, const double jmatcoef,
   const Vector &row_shape, const Vector &col_shape, const Vector &col_dshape_dnM,
   DenseMatrix &elmat, DenseMatrix &jmat)
{
   // row_offset and col_offset are not needed for elmat.
   for (int d = 0; d < dim; ++d)
   {
      int j = d * col_ndofs;
      for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
      {
         int i = d * row_ndofs;
         const double t2 = col_dshape_dnM(jdof);
         for (int idof = 0; idof < row_ndofs; ++idof, ++i)
            elmat(i, j) += row_shape(idof) * t2;
      }
   }

   if (jmatcoef == 0.0) { return; }

   // row_offset and col_offset are only needed to determine the lower-triangular part.
   for (int d = 0; d < dim; ++d)
   {
      const int jo = d*col_ndofs;
      const int io = d*row_ndofs;
      for (int jdof = 0, j = jo; jdof < col_ndofs; ++jdof, ++j)
      {
         const double sj = jmatcoef * col_shape(jdof);
         int i_start = (io + row_offset > j + col_offset) ? io : j;
         for (int i = i_start, idof = i - io; idof < row_ndofs; ++idof, ++i)
         {
            jmat(i, j) += row_shape(idof) * sj;
         }
      }
   }
}

void InterfaceDGNormalFluxIntegrator::AssembleInterfaceMatrix(
   const FiniteElement &trial_fe1, const FiniteElement &trial_fe2,
   const FiniteElement &test_fe1, const FiniteElement &test_fe2,
   FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
   Array2D<DenseMatrix *> &elmats)
{
   assert(elmats.NumRows() == 2);
   assert(elmats.NumCols() == 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) assert(elmats(i, j));

   dim = trial_fe1.GetDim();
   trial_dof1 = trial_fe1.GetDof();
   trial_vdof1 = dim * trial_dof1;
   test_dof1 = test_fe1.GetDof();

   nor.SetSize(dim);
   wnor.SetSize(dim);

   // vshape1.SetSize(trial_dof1, dim);
   // vshape1_n.SetSize(trial_dof1);
   trshape1.SetSize(trial_dof1);
   shape1.SetSize(test_dof1);

   if (Trans2.Elem1No >= 0)
   {
      trial_dof2 = trial_fe2.GetDof();
      trial_vdof2 = dim * trial_dof2;
      test_dof2 = test_fe2.GetDof();

      // vshape2.SetSize(trial_dof2, dim);
      // vshape2_n.SetSize(trial_dof2);
      trshape2.SetSize(trial_dof2);
      shape2.SetSize(test_dof2);
   }
   else
   {
      trial_dof2 = 0;
      test_dof2 = 0;
   }
   assert((trial_dof2 > 0) && (test_dof2 > 0));

   elmats(0,0)->SetSize(test_dof1, trial_vdof1);
   elmats(0,1)->SetSize(test_dof1, trial_vdof2);
   elmats(1,0)->SetSize(test_dof2, trial_vdof1);
   elmats(1,1)->SetSize(test_dof2, trial_vdof2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) *elmats(i,j) = 0.0;

   // TODO: need to revisit this part for proper convergence rate.
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      if (Trans2.Elem1No >= 0)
         order = (min(Trans1.Elem1->OrderW(), Trans2.Elem1->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans1.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans1.GetGeometryType(), order);

      assert(Trans1.GetGeometryType() == Trans2.GetGeometryType());
   }  // if (ir == NULL)

   for (p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans1.SetAllIntPoints(&ip);
      Trans2.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans2.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans1.Jacobian(), nor);
      }

      // trial_fe1.CalcVShape(eip1, vshape1);
      trial_fe1.CalcShape(eip1, trshape1);
      test_fe1.CalcShape(eip1, shape1);
      // vshape1.Mult(nor, vshape1_n);

      w = ip.weight;
      if (trial_dof2)
         w *= 0.5;

      wnor.Set(w, nor);

      DenseMatrix *elmat = NULL;
      
      elmat = elmats(0,0);
      for (jm = 0, j = 0; jm < dim; jm++)
      {
         wn = wnor(jm);
         for (jdof = 0; jdof < trial_dof1; jdof++, j++)
            for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
               (*elmat)(i, j) += wn * shape1(idof) * trshape1(jdof);
      }

      if (trial_dof2)
      {
         // trial_fe2.CalcVShape(eip2, vshape2);
         trial_fe2.CalcShape(eip2, trshape2);
         test_fe2.CalcShape(eip2, shape2);
         // vshape2.Mult(nor, vshape2_n);

         elmat = elmats(1,0);
         for (jm = 0, j = 0; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof1; jdof++, j++)
               for (idof = 0, i = 0; idof < test_dof2; idof++, i++)
                  (*elmat)(i, j) += wn * shape2(idof) * trshape1(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof1; j++)
         //    {
         //       elmat(test_dof1+i, j) += w * shape2(i) * vshape1_n(j);
         //    }

         elmat = elmats(1,1);
         for (jm = 0, j = 0; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = 0; idof < test_dof2; idof++, i++)
                  (*elmat)(i, j) -= wn * shape2(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(test_dof1+i, trial_dof1+j) -= w * shape2(i) * vshape2_n(j);
         //    }

         elmat = elmats(0,1);
         for (jm = 0, j = 0; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
                  (*elmat)(i, j) -= wn * shape1(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof1; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(i, trial_dof1+j) -= w * shape1(i) * vshape2_n(j);
         //    }
      }  // if (trial_dof2)
   }  // for (p = 0; p < ir->GetNPoints(); p++)
}

void InterfaceDGTemamFluxIntegrator::AssembleInterfaceVector(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
   const Vector &elfun1, const Vector &elfun2,
   Vector &elvect1, Vector &elvect2)
{
   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = el2.GetDof();
   nvdofs1 = dim * ndofs1;
   nvdofs2 = dim * ndofs2;
   elvect1.SetSize(nvdofs1);
   elvect2.SetSize(nvdofs2);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(elfun1.GetData(), ndofs1, dim);
   elv1.UseExternalData(elvect1.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);
   
   udof2.UseExternalData(elfun2.GetData(), ndofs2, dim);
   elv2.UseExternalData(elvect2.GetData(), ndofs2, dim);
   shape2.SetSize(ndofs2);
   u2.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(el1.GetOrder(), el2.GetOrder()) - 1)));
      ir = &IntRules.Get(Tr1.GetGeometryType(), order);
   }

   elvect1 = 0.0; elvect2 = 0.0;
   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Tr1.SetAllIntPoints(&ip);
      Tr2.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Tr1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr2.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);
      udof1.MultTranspose(shape1, u1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr1.Jacobian(), nor);
      }

      el2.CalcShape(eip2, shape2);
      udof2.MultTranspose(shape2, u2);

      // 0.5 factor remains whether boundary or interior face.
      w = 0.5 * ip.weight;
      if (Q) { w *= Q->Eval(*Tr1.Elem1, eip1); }

      nor *= w;
      un = nor * u1;
      flux.Set(un, u1);

      // If interior face, two flux terms cancel out.
      AddMultVWt(shape2, flux, elv2);
      flux.Set(-un, u2);
      AddMultVWt(shape1, flux, elv1);
   }
}

void InterfaceDGTemamFluxIntegrator::AssembleInterfaceGrad(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr1, FaceElementTransformations &Tr2,
   const Vector &elfun1, const Vector &elfun2,
   Array2D<DenseMatrix*> &elmats)
{
   assert(elmats.NumRows() == 2);
   assert(elmats.NumCols() == 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) assert(elmats(i, j));

   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = el2.GetDof();
   nvdofs1 = dim * ndofs1;
   nvdofs2 = dim * ndofs2;
   elmats(0, 0)->SetSize(nvdofs1);
   elmats(0, 1)->SetSize(nvdofs1, nvdofs2);
   elmats(1, 0)->SetSize(nvdofs2, nvdofs1);
   elmats(1, 1)->SetSize(nvdofs2);
   elmat_comp11.SetSize(ndofs1);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(elfun1.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);  

   udof2.UseExternalData(elfun2.GetData(), ndofs2, dim);
   shape2.SetSize(ndofs2);
   u2.SetSize(dim);

   elmat_comp12.SetSize(ndofs1, ndofs2);
   elmat_comp21.SetSize(ndofs2, ndofs1);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(el1.GetOrder(), el2.GetOrder()) - 1)));
      ir = &IntRules.Get(Tr1.GetGeometryType(), order);
   } 

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) *(elmats(i, j)) = 0.0;

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Tr1.SetAllIntPoints(&ip);
      Tr2.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Tr1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr2.GetElement1IntPoint();

      el1.CalcShape(eip1, shape1);
      udof1.MultTranspose(shape1, u1);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr1.Jacobian(), nor);
      }

      el2.CalcShape(eip2, shape2);
      udof2.MultTranspose(shape2, u2);

      // 0.5 factor remains whether boundary or interior face.
      w = 0.5 * ip.weight;
      if (Q) { w *= Q->Eval(*Tr1.Elem1, eip1); }

      nor *= w;
      un = nor * u1;

      MultVVt(shape1, elmat_comp11);
      MultVWt(shape1, shape2, elmat_comp12);
      elmat_comp21.Transpose(elmat_comp12);

      for (int di = 0; di < dim; di++)
      {
         elmats(1, 0)->AddMatrix(un, elmat_comp21, di * ndofs2, di * ndofs1);
         elmats(0, 1)->AddMatrix(-un, elmat_comp12, di * ndofs1, di * ndofs2);

         for (int dj = 0; dj < dim; dj++)
         {
            elmats(1, 0)->AddMatrix(u1(di) * nor(dj), elmat_comp21, di * ndofs2, dj * ndofs1);
            elmats(0, 0)->AddMatrix(-u2(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
         }
      }
   } // for (int pind = 0; pind < ir->GetNPoints(); ++pind)
}

void InterfaceDGElasticityIntegrator::AssembleInterfaceMatrix(
      const FiniteElement &el1, const FiniteElement &el2,
      FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
      Array2D<DenseMatrix *> &elmats)
{
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

   bool boundary = false;
   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (boundary) ? 0 : el2.GetDof();
   const int nvdofs1 = dim * ndofs1;
   const int nvdofs2 = dim * ndofs2;
   // Initially 'elmat' corresponds to the term:
   //    < { sigma(u) . n }, [v] > =
   //    < { (lambda div(u) I + mu (grad(u) + grad(u)^T)) . n }, [v] >
   // But eventually, it's going to be replaced by:
   //    elmat := -elmat + alpha*elmat^T + jmat
   elmats(0, 0)->SetSize(nvdofs1, nvdofs1);
   elmats(0, 1)->SetSize(nvdofs1, nvdofs2);
   elmats(1, 0)->SetSize(nvdofs2, nvdofs1);
   elmats(1, 1)->SetSize(nvdofs2, nvdofs2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         *elmats(i, j) = 0.0;

   const bool kappa_is_nonzero = (kappa != 0.0);
   jmats.SetSize(2, 2);
   if (kappa_is_nonzero)
   {
      jmats(0, 0) = new DenseMatrix(nvdofs1, nvdofs1);
      // only the lower-triangular part of jmat is assembled.
      jmats(0, 1) = NULL;
      jmats(1, 0) = new DenseMatrix(nvdofs2, nvdofs1);
      jmats(1, 1) = new DenseMatrix(nvdofs2, nvdofs2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j <= i; j++)
            *jmats(i, j) = 0.0;
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
      ir = &IntRules.Get(Trans1.GetGeometryType(), order);
      assert(Trans1.GetGeometryType() == Trans2.GetGeometryType());
   }

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans1.SetAllIntPoints(&ip);
      Trans2.SetAllIntPoints(&ip);
      // Access the neighboring elements' integration points
      // Note: eip1 and eip2 come from Element1 of Trans1 and Trans2 respectively.
      const IntegrationPoint &eip1 = Trans1.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans2.GetElement1IntPoint();
      //  computing outward normal vectors.
      if (dim == 1)
      {
         nor(0) = 2 * eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans1.Jacobian(), nor);
      }
      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);

      CalcAdjugate(Trans1.Elem1->Jacobian(), adjJ);
      Mult(dshape1, adjJ, dshape1_ps);

      double w, wLM;
      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         CalcAdjugate(Trans2.Elem1->Jacobian(), adjJ);
         Mult(dshape2, adjJ, dshape2_ps);

         w = ip.weight / 2;
         const double w2 = w / Trans2.Elem1->Weight();
         const double wL2 = w2 * lambda->Eval(*Trans2.Elem1, eip2);
         const double wM2 = w2 * mu->Eval(*Trans2.Elem1, eip2);
         nL2.Set(wL2, nor);
         nM2.Set(wM2, nor);
         wLM = (wL2 + 2.0 * wM2);
         dshape2_ps.Mult(nM2, dshape2_dnM);
      }
      else
      {
         w = ip.weight;
         wLM = 0.0;
      }

      {
         const double w1 = w / Trans1.Elem1->Weight();
         const double wL1 = w1 * lambda->Eval(*Trans1.Elem1, eip1);
         const double wM1 = w1 * mu->Eval(*Trans1.Elem1, eip1);
         nL1.Set(wL1, nor);
         nM1.Set(wM1, nor);
         wLM += (wL1 + 2.0 * wM1);
         dshape1_ps.Mult(nM1, dshape1_dnM);
      }
      const double jmatcoef = kappa * (nor * nor) * wLM;

      // (1,1) block
      AssembleBlock(
            dim, ndofs1, ndofs1, 0, 0, jmatcoef, nL1, nM1,
            shape1, shape1, dshape1_dnM, dshape1_ps, *elmats(0, 0), *jmats(0, 0));
      if (ndofs2 == 0)
      {
         continue;
      }

      // In both elmat and jmat, shape2 appears only with a minus sign.
      shape2.Neg();

      // (1,2) block
      AssembleBlock(
            dim, ndofs1, ndofs2, 0, dim * ndofs1, 0.0, nL2, nM2,
            shape1, shape2, dshape2_dnM, dshape2_ps, *elmats(0, 1), *jmats(0, 1));
      // (2,1) block
      AssembleBlock(
            dim, ndofs2, ndofs1, dim * ndofs1, 0, jmatcoef, nL1, nM1,
            shape2, shape1, dshape1_dnM, dshape1_ps, *elmats(1, 0), *jmats(1, 0));
      // (2,2) block
      AssembleBlock(
            dim, ndofs2, ndofs2, dim * ndofs1, dim * ndofs1, jmatcoef, nL2, nM2,
            shape2, shape2, dshape2_dnM, dshape2_ps, *elmats(1, 1), *jmats(1, 1));
   }

   // elmat := -elmat + sigma*elmat^t + jmat
   Array<int> ndof_array(2);
   ndof_array[0] = nvdofs1;
   ndof_array[1] = nvdofs2;
   DenseMatrix *elmat = NULL;
   DenseMatrix *jmat = NULL;
   DenseMatrix *elmat12 = NULL;
   if (kappa_is_nonzero)
   {
      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I, I);
         jmat = jmats(I, I);
         for (int i = 0; i < ndof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i, j), aji = (*elmat)(j, i), mij = (*jmat)(i, j);
               (*elmat)(i, j) = alpha * aji - aij + mij;
               (*elmat)(j, i) = alpha * aij - aji + mij;
            }
            (*elmat)(i, i) = (alpha - 1.) * (*elmat)(i, i) + (*jmat)(i, i);
         } // for (int i = 0; i < ndofs_array[I]; i++)
      }    // for (int I = 0; I < 2; I++)
      elmat = elmats(1, 0);
      jmat = jmats(1, 0);
      assert(jmat != NULL);
      elmat12 = elmats(0, 1);
      for (int i = 0; i < nvdofs2; i++)
      {
         for (int j = 0; j < nvdofs1; j++)
         {
            double aij = (*elmat)(i, j), aji = (*elmat12)(j, i), mij = (*jmat)(i, j);
            (*elmat)(i, j) = alpha * aji - aij + mij;
            (*elmat12)(j, i) = alpha * aij - aji + mij;
         } // for (int j = 0; j < ndofs1; j++)
      }    // for (int i = 0; i < ndofs2; i++)
   }       // if (kappa_is_nonzero)
   else
   {

      for (int I = 0; I < 2; I++)
      {
         elmat = elmats(I, I);
         for (int i = 0; i < ndof_array[I]; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = (*elmat)(i, j), aji = (*elmat)(j, i);
               (*elmat)(i, j) = alpha * aji - aij;
               (*elmat)(j, i) = alpha * aij - aji;
            }
            (*elmat)(i, i) *= (alpha - 1.);
         } // for (int i = 0; i < ndofs_array[I]; i++)
      }    // for (int I = 0; I < 2; I++)

      elmat = elmats(1, 0);
      elmat12 = elmats(0, 1);
      for (int i = 0; i < nvdofs2; i++)
      {
         for (int j = 0; j < nvdofs1; j++)
         {
            double aij = (*elmat)(i, j), aji = (*elmat12)(j, i);
            (*elmat)(i, j) = alpha * aji - aij;
            (*elmat12)(j, i) = alpha * aij - aji;
         } // for (int j = 0; j < ndofs1; j++)
      }    // for (int i = 0; i < ndofs2; i++)
   }       // not if (kappa_is_nonzero)

   if (kappa_is_nonzero)
      DeletePointers(jmats);
}

// static method
void InterfaceDGElasticityIntegrator::AssembleBlock(
      const int dim, const int row_ndofs, const int col_ndofs,
      const int row_offset, const int col_offset,
      const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
      const Vector &row_shape, const Vector &col_shape,
      const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
      DenseMatrix &elmat, DenseMatrix &jmat)
{
   for (int jm = 0, j = 0; jm < dim; ++jm)
   {
      for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
      {
         const double t2 = col_dshape_dnM(jdof);
         for (int im = 0, i = 0; im < dim; ++im)
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

   if (jmatcoef == 0.0)
   {
      return;
   }

   for (int d = 0; d < dim; ++d)
   {
      const int jo = d * col_ndofs;
      const int io = d * row_ndofs;
      for (int jdof = 0, j = jo; jdof < col_ndofs; ++jdof, ++j)
      {
         const double sj = jmatcoef * col_shape(jdof);
         int i_start = (io + row_offset > j + col_offset) ? io : j;
         for (int i = i_start, idof = i - io; idof < row_ndofs; ++idof, ++i)
         {
            jmat(i, j) += row_shape(idof) * sj;
         }
      }
   }
}
}
