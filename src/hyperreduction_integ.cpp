// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "hyperreduction_integ.hpp"
#include "linalg_utils.hpp"

using namespace std;

namespace mfem
{

void HyperReductionIntegrator::AssembleQuadratureVector(
   const FiniteElement &el, ElementTransformation &T, const IntegrationPoint &ip,
   const double &iw, const Vector &eltest, Vector &elquad)
{
   mfem_error ("HyperReductionIntegrator::AssembleQuadratureVector(...)\n"
               "for element is not implemented for this class.");
}

void HyperReductionIntegrator::AssembleQuadratureVector(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &T,
   const IntegrationPoint &ip, const double &iw, const Vector &eltest, Vector &elquad)
{
   mfem_error ("HyperReductionIntegrator::AssembleQuadratureVector(...)\n"
               "for face is not implemented for this class.");
}

void HyperReductionIntegrator::AssembleQuadratureGrad(
   const FiniteElement &el, ElementTransformation &T, const IntegrationPoint &ip,
   const double &iw, const Vector &eltest, DenseMatrix &quadmat)
{
   mfem_error ("HyperReductionIntegrator::AssembleQuadratureGrad(...)\n"
               "for element is not implemented for this class.");
}

void HyperReductionIntegrator::AssembleQuadratureGrad(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &T,
   const IntegrationPoint &ip, const double &iw, const Vector &eltest, DenseMatrix &quadmat)
{
   mfem_error ("HyperReductionIntegrator::AssembleQuadratureGrad(...)\n"
               "for face is not implemented for this class.");
}

void HyperReductionIntegrator::AppendPrecomputeDomainCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   mfem_error ("HyperReductionIntegrator::AppendPrecomputeDomainCoeffs(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AppendPrecomputeInteriorFaceCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   mfem_error ("HyperReductionIntegrator::AppendPrecomputeInteriorFaceCoeffs(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AppendPrecomputeBdrFaceCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   mfem_error ("HyperReductionIntegrator::AppendPrecomputeBdrFaceCoeffs(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleVector_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, Vector &y)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleVector_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleVector_Fast(
   const int s, const double qw, FaceElementTransformations &T, const IntegrationPoint &ip, const Vector &x, Vector &y)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleVector_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleGrad_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, DenseMatrix &jac)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleGrad_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleGrad_Fast(
   const int s, const double qw, FaceElementTransformations &T, const IntegrationPoint &ip, const Vector &x, DenseMatrix &jac)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleGrad_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::GetBasisElement(
   DenseMatrix &basis, const int col, const Array<int> vdofs, Vector &basis_el, DofTransformation *dof_trans)
{
   Vector tmp;
   basis.GetColumnReference(col, tmp);
   tmp.GetSubVector(vdofs, basis_el);   // this involves a copy.
   if (dof_trans) {dof_trans->InvTransformPrimal(basis_el); }
}

const IntegrationRule&
VectorConvectionTrilinearFormIntegrator::GetRule(const FiniteElement &fe,
                                                ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void VectorConvectionTrilinearFormIntegrator::AssembleElementVector(
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
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);
   ELV.UseExternalData(elvect.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // // NOTE: this is for a test of AssembleQuadratureVector,
      // //       which should return the equivalent answer.
      // Vector tmp(nd * dim);
      // AssembleQuadratureVector(el, T, ip, ip.weight, elfun, tmp);
      // elvect += tmp;
      // continue;

      T.SetIntPoint(&ip);
      el.CalcShape(ip, shape);
      el.CalcPhysDShape(T, dshape);
      double w = ip.weight * T.Weight();
      if (Q) { w *= Q->Eval(T, ip); }

      MultAtB(EF, dshape, gradEF);
      if (vQ)
         vQ->Eval(vec1, T, ip);
      else
         EF.MultTranspose(shape, vec1);
      gradEF.Mult(vec1, vec2);
      vec2 *= w;
      AddMultVWt(shape, vec2, ELV);
   }
}

// void VectorConvectionTrilinearFormIntegrator::AssembleElementQuadrature(
//    const FiniteElement &el,
//    ElementTransformation &T,
//    const Vector &eltest,
//    DenseMatrix &elquad)
// {
//    const int nd = el.GetDof();
//    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
//    const int nq = ir->GetNPoints();
//    dim = el.GetDim();

//    shape.SetSize(nd);
//    dshape.SetSize(nd, dim);
//    elquad.SetSize(nd * dim, nq);
//    gradEF.SetSize(dim);

//    EF.UseExternalData(eltest.GetData(), nd, dim);

//    Vector vec1(dim), vec2(dim), vec_tr(dim);

//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       ELV.UseExternalData(elquad.GetColumn(i), nd, dim);

//       const IntegrationPoint &ip = ir->IntPoint(i);
//       T.SetIntPoint(&ip);
//       el.CalcShape(ip, shape);
//       el.CalcPhysDShape(T, dshape);
//       double w = ip.weight * T.Weight();
//       if (Q) { w *= Q->Eval(T, ip); }

//       MultAtB(EF, dshape, gradEF);
//       if (vQ)
//          vQ->Eval(vec1, T, ip);
//       else
//          EF.MultTranspose(shape, vec1);
//       gradEF.Mult(vec1, vec2);
//       vec2 *= w;

//       MultVWt(shape, vec2, ELV);
//    }
// }

void VectorConvectionTrilinearFormIntegrator::AssembleQuadratureVector(
   const FiniteElement &el,
   ElementTransformation &T,
   const IntegrationPoint &ip,
   const double &iw,
   const Vector &eltest,
   Vector &elquad)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elquad.SetSize(nd * dim);
   gradEF.SetSize(dim);

   EF.UseExternalData(eltest.GetData(), nd, dim);
   ELV.UseExternalData(elquad.GetData(), nd, dim);

   Vector vec1(dim), vec2(dim), vec_tr(dim);

   // const IntegrationPoint &ip = ir->IntPoint(i);
   T.SetIntPoint(&ip);
   el.CalcShape(ip, shape);
   el.CalcPhysDShape(T, dshape);
   // double w = ip.weight * T.Weight();
   double w = iw * T.Weight();
   if (Q) { w *= Q->Eval(T, ip); }

   MultAtB(EF, dshape, gradEF);
   if (vQ)
      vQ->Eval(vec1, T, ip);
   else
      EF.MultTranspose(shape, vec1);
   gradEF.Mult(vec1, vec2);
   vec2 *= w;

   MultVWt(shape, vec2, ELV);
}

void VectorConvectionTrilinearFormIntegrator::AssembleElementGrad(
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
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);

      Mult(dshape, trans.InverseJacobian(), dshapex);

      w = ip.weight;

      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }

      MultAtB(EF, dshapex, gradEF);
      EF.MultTranspose(shape, vec1);

      trans.AdjugateJacobian().Mult(vec1, vec2);

      vec2 *= w;
      dshape.Mult(vec2, vec3);
      MultVWt(shape, vec3, elmat_comp);

      for (int ii = 0; ii < dim; ii++)
      {
         elmat.AddMatrix(elmat_comp, ii * nd, ii * nd);
      }

      MultVVt(shape, elmat_comp);
      w = ip.weight * trans.Weight();
      if (Q)
      {
         w *= Q->Eval(trans, ip);
      }
      for (int ii = 0; ii < dim; ii++)
      {
         for (int jj = 0; jj < dim; jj++)
         {
            elmat.AddMatrix(w * gradEF(ii, jj), elmat_comp, ii * nd, jj * nd);
         }
      }
   }
}

void VectorConvectionTrilinearFormIntegrator::AssembleQuadratureGrad(
   const FiniteElement &el,
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   const double &iw,
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
   gradEF.SetSize(dim);

   EF.UseExternalData(elfun.GetData(), nd, dim);

   double w;
   Vector vec1(dim), vec2(dim), vec3(nd);

   elmat = 0.0;
   trans.SetIntPoint(&ip);

   el.CalcShape(ip, shape);
   el.CalcDShape(ip, dshape);

   Mult(dshape, trans.InverseJacobian(), dshapex);

   // w = ip.weight;
   w = iw;

   if (Q)
   {
      w *= Q->Eval(trans, ip);
   }

   MultAtB(EF, dshapex, gradEF);
   EF.MultTranspose(shape, vec1);

   trans.AdjugateJacobian().Mult(vec1, vec2);

   vec2 *= w;
   dshape.Mult(vec2, vec3);
   MultVWt(shape, vec3, elmat_comp);

   for (int ii = 0; ii < dim; ii++)
   {
      elmat.AddMatrix(elmat_comp, ii * nd, ii * nd);
   }

   MultVVt(shape, elmat_comp);
   // w = ip.weight * trans.Weight();
   w = iw * trans.Weight();
   if (Q)
   {
      w *= Q->Eval(trans, ip);
   }
   for (int ii = 0; ii < dim; ii++)
   {
      for (int jj = 0; jj < dim; jj++)
      {
         elmat.AddMatrix(w * gradEF(ii, jj), elmat_comp, ii * nd, jj * nd);
      }
   }
}

void VectorConvectionTrilinearFormIntegrator::AppendPrecomputeDomainCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   const int nbasis = basis.NumCols();

   const int el = sample.el;
   const FiniteElement *fe = fes->GetFE(el);
   Array<int> vdofs;
   // TODO(kevin): not exactly sure what doftrans impacts.
   DofTransformation *doftrans = fes->GetElementVDofs(el, vdofs);
   ElementTransformation *T = fes->GetElementTransformation(el);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*fe, *T);
   const IntegrationPoint &ip = ir->IntPoint(sample.qp);

   const int nd = fe->GetDof();
   dim = fe->GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);

   T->SetIntPoint(&ip);
   fe->CalcShape(ip, shape);
   fe->CalcPhysDShape(*T, dshape);

   if (tensor)
   {
      // Not all nonlinear form integrators can have tensors as coefficients.
      // This is the special case of polynominally nonlinear operator.
      // For more general nonlinear operators, probably shape/dshape have to be stored.
      DenseTensor *elten = new DenseTensor(nbasis, nbasis, nbasis);

      gradEF.SetSize(dim);
      Vector vec1(dim), vec2(dim);
      Vector vec3(nd * dim);
      elmat_comp.UseExternalData(vec3.GetData(), nd, dim);
      Vector basis_i, basis_j, basis_k;

      for (int i = 0; i < nbasis; i++)
      {
         GetBasisElement(basis, i, vdofs, basis_i, doftrans);
         EF.UseExternalData(basis_i.GetData(), nd, dim);
         EF.MultTranspose(shape, vec1);

         for (int j = 0; j < nbasis; j++)
         {
            GetBasisElement(basis, j, vdofs, basis_j, doftrans);
            ELV.UseExternalData(basis_j.GetData(), nd, dim);
            MultAtB(ELV, dshape, gradEF);
            gradEF.Mult(vec1, vec2);
            MultVWt(shape, vec2, elmat_comp);
            if (doftrans) { doftrans->TransformDual(vec3); }

            for (int k = 0; k < nbasis; k++)
            {
               GetBasisElement(basis, k, vdofs, basis_k);   // doftrans is already applied above for test function.

               (*elten)(i, j, k) = (basis_k * vec3);
            }  // for (int k = 0; k < nbasis; k++)
         }  // for (int j = 0; j < nbasis; j++)
      }  // for (int i = 0; i < nbasis; i++)

      coeffs.Append(elten);
   }
   else
   {
      Vector basis_i;
      DenseMatrix *vec1s = new DenseMatrix(dim, nbasis);
      Array<DenseMatrix *> *gradEFs = new Array<DenseMatrix *>(0);
      for (int i = 0; i < nbasis; i++)
      {
         GetBasisElement(basis, i, vdofs, basis_i, doftrans);
         EF.UseExternalData(basis_i.GetData(), nd, dim);

         Vector vec1;
         vec1s->GetColumnReference(i, vec1);
         EF.MultTranspose(shape, vec1);

         DenseMatrix *gradEF1 = new DenseMatrix(dim);
         MultAtB(EF, dshape, *gradEF1);
         gradEFs->Append(gradEF1);
      }
      shapes.Append(vec1s);
      dshapes.Append(gradEFs);
   }
}

void VectorConvectionTrilinearFormIntegrator::AddAssembleVector_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, Vector &y)
{
   T.SetIntPoint(&ip);
   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   if (tensor)
   {
      const DenseTensor *tensor = coeffs[s];
      Vector tmp(tensor->SizeK());
      y.SetSize(tensor->SizeK());
      TensorAddScaledContract(*tensor, w, x, x, y);
   }
   else
   {
      dim = shapes[s]->NumRows();
      Vector vec1(dim), vec2(dim);
      shapes[s]->Mult(x, vec1);
      Array<DenseMatrix *> *gradEFs = dshapes[s];

      gradEF.SetSize(dim);
      gradEF = 0.0;
      for (int k = 0; k < gradEFs->Size(); k++)
         gradEF.Add(x(k), *((*gradEFs)[k]));
      gradEF.Mult(vec1, vec2);

      assert(y.Size() == x.Size());
      shapes[s]->AddMultTranspose(vec2, y, w);
   }
}

void VectorConvectionTrilinearFormIntegrator::AddAssembleGrad_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, DenseMatrix &jac)
{
   T.SetIntPoint(&ip);
   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   if (tensor)
   {
      const DenseTensor *tensor = coeffs[s];
      TensorAddScaledMultTranspose(*tensor, w, x, 0, jac);
      TensorAddScaledMultTranspose(*tensor, w, x, 1, jac);
   }
   else
   {
      dim = shapes[s]->NumRows();
      int nbasis = shapes[s]->NumCols();
      Vector vec1(dim), vec2(dim);
      shapes[s]->Mult(x, vec1);
      Array<DenseMatrix *> *gradEFs = dshapes[s];

      gradEF.SetSize(dim);
      gradEF = 0.0;
      for (int k = 0; k < gradEFs->Size(); k++)
         gradEF.Add(x(k), *((*gradEFs)[k]));
      gradEF *= w;

      ELV.SetSize(dim, nbasis);
      Mult(gradEF, *shapes[s], ELV);

      for (int k = 0; k < nbasis; k++)
      {
         ELV.GetColumnReference(k, vec2);
         (*gradEFs)[k]->AddMult(vec1, vec2, w);
      }

      Vector jac_col;
      for (int k = 0; k < nbasis; k++)
      {
         jac.GetColumnReference(k, jac_col);
         ELV.GetColumnReference(k, vec2);
         shapes[s]->AddMultTranspose(vec2, jac_col);
      }
   }
}

/*
   IncompressibleInviscidFluxNLFIntegrator
*/
const IntegrationRule&
IncompressibleInviscidFluxNLFIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

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

void IncompressibleInviscidFluxNLFIntegrator::AssembleQuadratureVector(
   const FiniteElement &el, ElementTransformation &T, const IntegrationPoint &ip,
   const double &iw, const Vector &eltest, Vector &elquad)
{
   const int nd = el.GetDof();
   dim = el.GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);
   elquad.SetSize(nd * dim);
   uu.SetSize(dim);

   EF.UseExternalData(eltest.GetData(), nd, dim);
   ELV.UseExternalData(elquad.GetData(), nd, dim);

   Vector u1(dim);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, T);
   ELV = 0.0;

   T.SetIntPoint(&ip);
   el.CalcShape(ip, shape);
   el.CalcPhysDShape(T, dshape);
   double w = iw * T.Weight();
   if (Q) { w *= Q->Eval(T, ip); }

   // MultAtB(EF, dshape, gradEF);
   EF.MultTranspose(shape, u1);
   MultVVt(u1, uu);
   
   AddMult_a(w, dshape, uu, ELV);
}

void IncompressibleInviscidFluxNLFIntegrator::AssembleQuadratureGrad(
   const FiniteElement &el, ElementTransformation &trans, const IntegrationPoint &ip,
   const double &iw, const Vector &elfun, DenseMatrix &elmat)
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

   Vector vec1(dim), vec2(nd), vec3(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, trans);

   elmat = 0.0;
   
   trans.SetIntPoint(&ip);

   el.CalcShape(ip, shape);
   // el.CalcDShape(ip, dshape);
   el.CalcPhysDShape(trans, dshape);

   EF.MultTranspose(shape, vec1);

   double w = iw * trans.Weight();
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

void IncompressibleInviscidFluxNLFIntegrator::AppendPrecomputeDomainCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   const int nbasis = basis.NumCols();

   const int el = sample.el;
   const FiniteElement *fe = fes->GetFE(el);
   Array<int> vdofs;
   // TODO(kevin): not exactly sure what doftrans impacts.
   DofTransformation *doftrans = fes->GetElementVDofs(el, vdofs);
   ElementTransformation *T = fes->GetElementTransformation(el);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(*fe, *T);
   const IntegrationPoint &ip = ir->IntPoint(sample.qp);

   const int nd = fe->GetDof();
   dim = fe->GetDim();

   shape.SetSize(nd);
   dshape.SetSize(nd, dim);

   T->SetIntPoint(&ip);
   fe->CalcShape(ip, shape);
   fe->CalcPhysDShape(*T, dshape);

   Vector basis_i;
   DenseMatrix *vec1s = new DenseMatrix(dim, nbasis);
   Array<DenseMatrix *> *gradEFs = new Array<DenseMatrix *>(0);
   for (int i = 0; i < nbasis; i++)
   {
      GetBasisElement(basis, i, vdofs, basis_i, doftrans);
      EF.UseExternalData(basis_i.GetData(), nd, dim);

      Vector vec1;
      vec1s->GetColumnReference(i, vec1);
      EF.MultTranspose(shape, vec1);

      DenseMatrix *gradEF1 = new DenseMatrix(dim);
      MultAtB(EF, dshape, *gradEF1);
      gradEFs->Append(gradEF1);
   }
   shapes.Append(vec1s);
   dshapes.Append(gradEFs);
}

void IncompressibleInviscidFluxNLFIntegrator::AddAssembleVector_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, Vector &y)
{
   T.SetIntPoint(&ip);
   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   dim = shapes[s]->NumRows();
   Vector u1(dim);
   shapes[s]->Mult(x, u1);
   Array<DenseMatrix *> *dshape = dshapes[s];

   Vector vec1(dim);
   assert(y.Size() == dshape->Size());
   for (int k = 0; k < dshape->Size(); k++)
   {
      (*dshape)[k]->Mult(u1, vec1);
      y(k) += w * (u1 * vec1);
   }
}

void IncompressibleInviscidFluxNLFIntegrator::AddAssembleGrad_Fast(
   const int s, const double qw, ElementTransformation &T, const IntegrationPoint &ip, const Vector &x, DenseMatrix &jac)
{
   T.SetIntPoint(&ip);
   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   dim = shapes[s]->NumRows();
   int nbasis = shapes[s]->NumCols();
   Vector u1(dim), vec1(dim), vec2(nbasis);
   shapes[s]->Mult(x, u1);
   Array<DenseMatrix *> *dshape = dshapes[s];

   u1 *= w;

   for (int i = 0; i < nbasis; i++)
   {
      (*dshape)[i]->Mult(u1, vec1);
      (*dshape)[i]->AddMultTranspose(u1, vec1);
      shapes[s]->MultTranspose(vec1, vec2);

      double *d_jac = jac.GetData() + i;
      for (int j = 0; j < nbasis; j++)
      {
         (*d_jac) += vec2(j);
         d_jac += nbasis;
      }
   }
}

/*
   DGLaxFriedrichsFluxIntegrator
*/

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

      w = ip.weight * Tr.Weight();
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= w;

      un1 = nor * u1;
      un2 = (ndofs2) ? nor * u2 : 0.0;

      flux.Set(un1, u1);
      if (ndofs2)
      {
         flux *= 0.5;
         flux.Add(0.5 * un2, u2);
      }

      un = max(abs(un1), abs(un2));
      flux.Add(un, u1);
      if (ndofs2)
         flux.Add(-un, u2);

      AddMultVWt(shape1, flux, elv1);
      if (ndofs2)
         AddMult_a_VWt(-1.0, shape2, flux, elv2);
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

      w = ip.weight * Tr.Weight();
      if (Q) { w *= Q->Eval(Tr, ip); }

      nor *= w;
      // Just for the average operator gradient.
      if (ndofs2)
         nor *= 0.5;

      un1 = nor * u1;
      un2 = (ndofs2) ? nor * u2 : 0.0;

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
            elmat.AddMatrix(u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
            if (ndofs2)
            {
               elmat.AddMatrix(-u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
               elmat.AddMatrix(u2(di) * nor(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
               elmat.AddMatrix(-u2(di) * nor(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
            }
         }
      }

      // Recover 1/2 factor on normal vector.
      if (ndofs2)
         nor *= 2.0;

      // un1 as maximum absolute eigenvalue, un2 as the sign.
      un = max( abs(un1), abs(un2) );
      if (ndofs2) un *= 2.0;
      
      double sgn = 0.0;
      bool u1_lg_u2 = (abs(un1) >= abs(un2));
      if (u1_lg_u2)
      {
         sgn = (un1 >= 0.0) ? 1.0 : -1.0;
      }
      else
      {
         sgn = (un2 >= 0.0) ? 1.0 : -1.0;
      }

      // [ u ] = u1 - u2
      if (ndofs2)
         u1.Add(-1.0, u2);

      for (int di = 0; di < dim; di++)
      {
         elmat.AddMatrix(un, elmat_comp11, di * ndofs1, di * ndofs1);
         if (ndofs2)
         {
            elmat.AddMatrix(-un, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
            elmat.AddMatrix(-un, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);
            elmat.AddMatrix(un, elmat_comp22, di * ndofs2 + dim * ndofs1, di * ndofs2 + dim * ndofs1);
         }

         // remember u1 in this loop is in fact [ u ] = u1 - u2.
         for (int dj = 0; dj < dim; dj++)
         {
            if (u1_lg_u2)
            {
               elmat.AddMatrix(sgn * u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
               if (ndofs2)
                  elmat.AddMatrix(-sgn * u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
            }
            else
            {
               assert(ndofs2);
               elmat.AddMatrix(sgn * u1(di) * nor(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
               elmat.AddMatrix(-sgn * u1(di) * nor(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
            }
         }
      }
   }
}

void DGLaxFriedrichsFluxIntegrator::AssembleQuadratureVector(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &T,
   const IntegrationPoint &ip, const double &iw, const Vector &eltest, Vector &elquad)
{
   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = (T.Elem2No >= 0) ? el2.GetDof() : 0;
   nvdofs = dim * (ndofs1 + ndofs2);
   elquad.SetSize(nvdofs);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(eltest.GetData(), ndofs1, dim);
   elv1.UseExternalData(elquad.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);
   
   if (ndofs2)
   {
      udof2.UseExternalData(eltest.GetData() + ndofs1 * dim, ndofs2, dim);
      elv2.UseExternalData(elquad.GetData() + ndofs1 * dim, ndofs2, dim);
      shape2.SetSize(ndofs2);
      u2.SetSize(dim);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0) - 1)));
      ir = &IntRules.Get(T.GetGeometryType(), order);
   }

   elquad = 0.0;

   // Set the integration point in the face and the neighboring elements
   T.SetAllIntPoints(&ip);

   // Access the neighboring elements' integration points
   // Note: eip2 will only contain valid data if Elem2 exists
   const IntegrationPoint &eip1 = T.GetElement1IntPoint();
   const IntegrationPoint &eip2 = T.GetElement2IntPoint();

   el1.CalcShape(eip1, shape1);
   udof1.MultTranspose(shape1, u1);

   if (dim == 1)
   {
      nor(0) = 2*eip1.x - 1.0;
   }
   else
   {
      CalcOrtho(T.Jacobian(), nor);
   }

   if (ndofs2)
   {
      el2.CalcShape(eip2, shape2);
      udof2.MultTranspose(shape2, u2);
   }

   w = iw * T.Weight();
   if (Q) { w *= Q->Eval(T, ip); }

   nor *= w;

   un1 = nor * u1;
   un2 = (ndofs2) ? nor * u2 : 0.0;

   flux.Set(un1, u1);
   if (ndofs2)
   {
      flux *= 0.5;
      flux.Add(0.5 * un2, u2);
   }

   un = max(abs(un1), abs(un2));
   flux.Add(un, u1);
   if (ndofs2)
      flux.Add(-un, u2);

   AddMultVWt(shape1, flux, elv1);
   if (ndofs2)
      AddMult_a_VWt(-1.0, shape2, flux, elv2);
}

void DGLaxFriedrichsFluxIntegrator::AssembleQuadratureGrad(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &T,
   const IntegrationPoint &ip, const double &iw, const Vector &eltest, DenseMatrix &quadmat)
{
   dim = el1.GetDim();
   ndofs1 = el1.GetDof();
   ndofs2 = (T.Elem2No >= 0) ? el2.GetDof() : 0;
   nvdofs = dim * (ndofs1 + ndofs2);
   quadmat.SetSize(nvdofs);
   elmat_comp11.SetSize(ndofs1);

   nor.SetSize(dim);
   flux.SetSize(dim);

   udof1.UseExternalData(eltest.GetData(), ndofs1, dim);
   shape1.SetSize(ndofs1);
   u1.SetSize(dim);  

   if (ndofs2)
   {
      udof2.UseExternalData(eltest.GetData() + ndofs1 * dim, ndofs2, dim);
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
      ir = &IntRules.Get(T.GetGeometryType(), order);
   } 

   quadmat = 0.0;

   // Set the integration point in the face and the neighboring elements
   T.SetAllIntPoints(&ip);

   // Access the neighboring elements' integration points
   // Note: eip2 will only contain valid data if Elem2 exists
   const IntegrationPoint &eip1 = T.GetElement1IntPoint();
   const IntegrationPoint &eip2 = T.GetElement2IntPoint();

   el1.CalcShape(eip1, shape1);
   udof1.MultTranspose(shape1, u1);

   if (dim == 1)
   {
      nor(0) = 2*eip1.x - 1.0;
   }
   else
   {
      CalcOrtho(T.Jacobian(), nor);
   }

   if (ndofs2)
   {
      el2.CalcShape(eip2, shape2);
      udof2.MultTranspose(shape2, u2);
   }

   w = iw * T.Weight();
   if (Q) { w *= Q->Eval(T, ip); }

   nor *= w;
   // Just for the average operator gradient.
   if (ndofs2)
      nor *= 0.5;

   un1 = nor * u1;
   un2 = (ndofs2) ? nor * u2 : 0.0;

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
      quadmat.AddMatrix(un1, elmat_comp11, di * ndofs1, di * ndofs1);
      if (ndofs2)
      {
         quadmat.AddMatrix(-un1, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
         quadmat.AddMatrix(un2, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);
         quadmat.AddMatrix(-un2, elmat_comp22, di * ndofs2 + dim * ndofs1, di * ndofs2 + dim * ndofs1);
      }

      for (int dj = 0; dj < dim; dj++)
      {
         quadmat.AddMatrix(u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
         if (ndofs2)
         {
            quadmat.AddMatrix(-u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
            quadmat.AddMatrix(u2(di) * nor(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
            quadmat.AddMatrix(-u2(di) * nor(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
         }
      }
   }

   // Recover 1/2 factor on normal vector.
   if (ndofs2)
      nor *= 2.0;

   // un1 as maximum absolute eigenvalue, un2 as the sign.
   un = max( abs(un1), abs(un2) );
   if (ndofs2) un *= 2.0;
   
   double sgn = 0.0;
   bool u1_lg_u2 = (abs(un1) >= abs(un2));
   if (u1_lg_u2)
   {
      sgn = (un1 >= 0.0) ? 1.0 : -1.0;
   }
   else
   {
      sgn = (un2 >= 0.0) ? 1.0 : -1.0;
   }

   // [ u ] = u1 - u2
   if (ndofs2)
      u1.Add(-1.0, u2);

   for (int di = 0; di < dim; di++)
   {
      quadmat.AddMatrix(un, elmat_comp11, di * ndofs1, di * ndofs1);
      if (ndofs2)
      {
         quadmat.AddMatrix(-un, elmat_comp21, di * ndofs2 + dim * ndofs1, di * ndofs1);
         quadmat.AddMatrix(-un, elmat_comp12, di * ndofs1, di * ndofs2 + dim * ndofs1);
         quadmat.AddMatrix(un, elmat_comp22, di * ndofs2 + dim * ndofs1, di * ndofs2 + dim * ndofs1);
      }

      // remember u1 in this loop is in fact [ u ] = u1 - u2.
      for (int dj = 0; dj < dim; dj++)
      {
         if (u1_lg_u2)
         {
            quadmat.AddMatrix(sgn * u1(di) * nor(dj), elmat_comp11, di * ndofs1, dj * ndofs1);
            if (ndofs2)
               quadmat.AddMatrix(-sgn * u1(di) * nor(dj), elmat_comp21, di * ndofs2 + dim * ndofs1, dj * ndofs1);
         }
         else
         {
            assert(ndofs2);
            quadmat.AddMatrix(sgn * u1(di) * nor(dj), elmat_comp12, di * ndofs1, dj * ndofs2 + dim * ndofs1);
            quadmat.AddMatrix(-sgn * u1(di) * nor(dj), elmat_comp22, di * ndofs2 + dim * ndofs1, dj * ndofs2 + dim * ndofs1);
         }
      }
   }
}

void DGLaxFriedrichsFluxIntegrator::AppendPrecomputeInteriorFaceCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   const int face = sample.face;
   FaceElementTransformations *T = fes->GetMesh()->GetInteriorFaceTransformations(face);
   assert(T != NULL);
   assert(T->Elem2No >= 0);

   AppendPrecomputeFaceCoeffs(fes, T, basis, sample);
}

void DGLaxFriedrichsFluxIntegrator::AppendPrecomputeBdrFaceCoeffs(
   const FiniteElementSpace *fes, DenseMatrix &basis, const SampleInfo &sample)
{
   const int be = sample.be;
   FaceElementTransformations *T = fes->GetMesh()->GetBdrFaceTransformations(be);
   assert(T != NULL);
   assert(T->Elem2No < 0);

   AppendPrecomputeFaceCoeffs(fes, T, basis, sample);
}

void DGLaxFriedrichsFluxIntegrator::AppendPrecomputeFaceCoeffs(
   const FiniteElementSpace *fes, FaceElementTransformations *T,
   DenseMatrix &basis, const SampleInfo &sample)
{
   const int nbasis = basis.NumCols();

   const bool el2 = (T->Elem2No >= 0);

   const FiniteElement *fe1 = fes->GetFE(T->Elem1No);
   const FiniteElement *fe2 = (el2) ? fes->GetFE(T->Elem2No) : fe1;

   Array<int> vdofs, vdofs2;
   fes->GetElementVDofs(T->Elem1No, vdofs);
   if (el2)
   {
      fes->GetElementVDofs(T->Elem2No, vdofs2);
      vdofs.Append(vdofs2);
   }

   dim = fe1->GetDim();
   ndofs1 = fe1->GetDof();
   ndofs2 = (T->Elem2No >= 0) ? fe2->GetDof() : 0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = (int)(ceil(1.5 * (2 * max(fe1->GetOrder(), ndofs2 ? fe2->GetOrder() : 0) - 1)));
      ir = &IntRules.Get(T->GetGeometryType(), order);
   }
   const IntegrationPoint &ip = ir->IntPoint(sample.qp);

   shape1.SetSize(ndofs1);
   if (el2) shape2.SetSize(ndofs2);

   T->SetAllIntPoints(&ip);

   const IntegrationPoint &eip1 = T->GetElement1IntPoint();
   fe1->CalcShape(eip1, shape1);
   if (el2)
   {
      const IntegrationPoint &eip2 = T->GetElement2IntPoint();
      fe2->CalcShape(eip2, shape2);
   }

   Vector basis_i;
   DenseMatrix *vec1s, *vec2s;
   vec1s = new DenseMatrix(dim, nbasis);
   if (el2) vec2s = new DenseMatrix(dim, nbasis);

   for (int i = 0; i < nbasis; i++)
   {
      GetBasisElement(basis, i, vdofs, basis_i);
      elv1.UseExternalData(basis_i.GetData(), ndofs1, dim);
      if (el2) elv2.UseExternalData(basis_i.GetData() + ndofs1 * dim, ndofs2, dim);

      Vector vec1, vec2;
      vec1s->GetColumnReference(i, vec1);
      elv1.MultTranspose(shape1, vec1);

      if (el2)
      {
         vec2s->GetColumnReference(i, vec2);
         elv2.MultTranspose(shape2, vec2);
      }
   }
   shapes1.Append(vec1s);
   if (el2) shapes2.Append(vec2s);
}

void DGLaxFriedrichsFluxIntegrator::AddAssembleVector_Fast(
   const int s, const double qw, FaceElementTransformations &T, const IntegrationPoint &ip, const Vector &x, Vector &y)
{
   const bool el2 = (T.Elem2No >= 0);
   T.SetAllIntPoints(&ip);
   // Access the neighboring elements' integration points
   // Note: eip2 will only contain valid data if Elem2 exists
   const IntegrationPoint &eip1 = T.GetElement1IntPoint();
   // const IntegrationPoint &eip2 = T.GetElement2IntPoint();

   dim = shapes1[s]->NumRows();
   nor.SetSize(dim);
   flux.SetSize(dim);
   u1.SetSize(dim);
   if (el2) u2.SetSize(dim);

   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   if (dim == 1)
   {
      nor(0) = 2*eip1.x - 1.0;
   }
   else
   {
      CalcOrtho(T.Jacobian(), nor);
   }

   nor *= w;

   shapes1[s]->Mult(x, u1);
   if (el2) shapes2[s]->Mult(x, u2);

   un1 = nor * u1;
   un2 = (el2) ? nor * u2 : 0.0;

   flux.Set(un1, u1);
   if (el2)
   {
      flux *= 0.5;
      flux.Add(0.5 * un2, u2);
   }

   un = max(abs(un1), abs(un2));
   flux.Add(un, u1);
   if (el2)
      flux.Add(-un, u2);

   assert(y.Size() == x.Size());
   shapes1[s]->AddMultTranspose(flux, y, 1.0);
   if (el2) shapes2[s]->AddMultTranspose(flux, y, -1.0);
}

void DGLaxFriedrichsFluxIntegrator::AddAssembleGrad_Fast(
   const int s, const double qw, FaceElementTransformations &T, const IntegrationPoint &ip, const Vector &x, DenseMatrix &jac)
{
   const bool el2 = (T.Elem2No >= 0);
   T.SetAllIntPoints(&ip);
   // Access the neighboring elements' integration points
   // Note: eip2 will only contain valid data if Elem2 exists
   const IntegrationPoint &eip1 = T.GetElement1IntPoint();
   // const IntegrationPoint &eip2 = T.GetElement2IntPoint();

   dim = shapes1[s]->NumRows();
   int nbasis = shapes1[s]->NumCols();
   nor.SetSize(dim);
   flux.SetSize(dim);
   u1.SetSize(dim);
   elmat_comp11.SetSize(dim);
   elmat_comp11 = 0.0;
   tmp.SetSize(dim, nbasis);

   if (el2)
   {
      u2.SetSize(dim);
      elmat_comp12.SetSize(dim);
      elmat_comp21.SetSize(dim);
      elmat_comp22.SetSize(dim);

      elmat_comp12 = 0.0;
      elmat_comp21 = 0.0;
      elmat_comp22 = 0.0;
   }

   double w = qw * T.Weight();
   if (Q) 
      w *= Q->Eval(T, ip);

   if (dim == 1)
   {
      nor(0) = 2*eip1.x - 1.0;
   }
   else
   {
      CalcOrtho(T.Jacobian(), nor);
   }

   nor *= w;

   shapes1[s]->Mult(x, u1);
   if (el2) shapes2[s]->Mult(x, u2);

   un1 = nor * u1;
   un2 = (el2) ? nor * u2 : 0.0;

   un = max( abs(un1), abs(un2) );
   double sgn = 0.0;
   bool u1_lg_u2 = (abs(un1) >= abs(un2));
   if (u1_lg_u2)
   {
      sgn = (un1 >= 0.0) ? 1.0 : -1.0;
   }
   else
   {
      sgn = (un2 >= 0.0) ? 1.0 : -1.0;
   }

   double factor = 1.0;
   if (el2)
   {
      un1 *= 0.5;
      un2 *= 0.5;
      factor = 0.5;
   }

   for (int di = 0; di < dim; di++)
   {
      elmat_comp11(di, di) += un1 + un;
      if (el2)
      {
         elmat_comp21(di, di) += -un1 - un;
         elmat_comp12(di, di) += un2 - un;
         elmat_comp22(di, di) += -un2 + un;
      }
   }

   AddMult_a_VWt(factor, u1, nor, elmat_comp11);
   if (el2)
   {
      AddMult_a_VWt(-factor, u1, nor, elmat_comp21);
      AddMult_a_VWt(factor, u2, nor, elmat_comp12);
      AddMult_a_VWt(-factor, u2, nor, elmat_comp22);
   }

   // [ u ] = u1 - u2
   if (ndofs2)
      u1.Add(-1.0, u2);

   if (u1_lg_u2)
   {
      AddMult_a_VWt(sgn, u1, nor, elmat_comp11);
      if (el2)
         AddMult_a_VWt(-sgn, u1, nor, elmat_comp21);
   }
   else
   {
      AddMult_a_VWt(sgn, u1, nor, elmat_comp12);
      AddMult_a_VWt(-sgn, u1, nor, elmat_comp22);
   }

   Mult(elmat_comp11, *shapes1[s], tmp);
   Vector jac_col;
   for (int k = 0; k < nbasis; k++)
   {
      jac.GetColumnReference(k, jac_col);
      tmp.GetColumnReference(k, tmp_vec);
      shapes1[s]->AddMultTranspose(tmp_vec, jac_col);
   }

   if (el2)
   {
      Mult(elmat_comp12, *shapes2[s], tmp);
      for (int k = 0; k < nbasis; k++)
      {
         jac.GetColumnReference(k, jac_col);
         tmp.GetColumnReference(k, tmp_vec);
         shapes1[s]->AddMultTranspose(tmp_vec, jac_col);
      }

      Mult(elmat_comp21, *shapes1[s], tmp);
      for (int k = 0; k < nbasis; k++)
      {
         jac.GetColumnReference(k, jac_col);
         tmp.GetColumnReference(k, tmp_vec);
         shapes2[s]->AddMultTranspose(tmp_vec, jac_col);
      }

      Mult(elmat_comp22, *shapes2[s], tmp);
      for (int k = 0; k < nbasis; k++)
      {
         jac.GetColumnReference(k, jac_col);
         tmp.GetColumnReference(k, tmp_vec);
         shapes2[s]->AddMultTranspose(tmp_vec, jac_col);
      }
   }
}

}
