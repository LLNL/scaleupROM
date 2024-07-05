// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "hyperreduction_integ.hpp"
#include "linalg_utils.hpp"

using namespace std;

namespace mfem
{

void EQPElement::Save(hid_t file_id, const std::string &dsetname, const IntegratorType type)
{
   std::string eldset;
   switch (type)
   {
      case IntegratorType::DOMAIN:        eldset = "elem"; break;
      case IntegratorType::INTERIORFACE:  eldset = "face"; break;
      case IntegratorType::BDRFACE:       eldset = "be"; break;
      case IntegratorType::INTERFACE:     eldset = "itf"; break;
      default:
         mfem_error("EQPElement::Save- Unknown IntegratorType!\n");
   }

   Array<int> el, qp;
   Array<double> qw;

   el.SetSize(0);
   qp.SetSize(0);
   qw.SetSize(0);

   for (int s = 0; s < samples.Size(); s++)
   {
      el.Append(samples[s]->info.el);
      qp.Append(samples[s]->info.qp);
      qw.Append(samples[s]->info.qw);
   }

   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, dsetname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::WriteDataset(grp_id, eldset, el);
   hdf5_utils::WriteDataset(grp_id, "quad-pt", qp);
   hdf5_utils::WriteDataset(grp_id, "quad-wt", qw);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void EQPElement::Load(hid_t file_id, const std::string &dsetname, const IntegratorType type)
{
   std::string eldset;
   switch (type)
   {
      case IntegratorType::DOMAIN:        eldset = "elem"; break;
      case IntegratorType::INTERIORFACE:  eldset = "face"; break;
      case IntegratorType::BDRFACE:       eldset = "be"; break;
      case IntegratorType::INTERFACE:     eldset = "itf"; break;
      default:
         mfem_error("EQPElement::Load- Unknown IntegratorType!\n");
   }

   Array<int> el, qp;
   Array<double> qw;

   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gopen2(file_id, dsetname.c_str(), H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::ReadDataset(grp_id, eldset, el);
   hdf5_utils::ReadDataset(grp_id, "quad-pt", qp);
   hdf5_utils::ReadDataset(grp_id, "quad-wt", qw);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   samples.SetSize(el.Size());
   for (int k = 0; k < el.Size(); k++)
      samples[k] = new EQPSample(SampleInfo({.el=el[k], .qp=qp[k], .qw=qw[k]}));
}

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
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, Vector &y)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleVector_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleVector_Fast(
   const int s, const EQPSample &eqp_sample, FaceElementTransformations &T, const Vector &x, Vector &y)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleVector_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleGrad_Fast(
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, DenseMatrix &jac)
{
   mfem_error ("HyperReductionIntegrator::AddAssembleGrad_Fast(...)\n"
               "is not implemented for this class,\n"
               "even though this class is set to be precomputable!\n");
}

void HyperReductionIntegrator::AddAssembleGrad_Fast(
   const int s, const EQPSample &eqp_sample, FaceElementTransformations &T, const Vector &x, DenseMatrix &jac)
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
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, Vector &y)
{
   const IntegrationPoint &ip = GetIntegrationRule()->IntPoint(eqp_sample.info.qp);
   const double qw = eqp_sample.info.qw;

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
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, DenseMatrix &jac)
{
   const IntegrationPoint &ip = GetIntegrationRule()->IntPoint(eqp_sample.info.qp);
   const double qw = eqp_sample.info.qw;
   
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
      
      AddMult_a_ABt(w, dshape, uu, ELV);
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
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, Vector &y)
{
   const IntegrationPoint &ip = GetIntegrationRule()->IntPoint(eqp_sample.info.qp);
   const double qw = eqp_sample.info.qw;

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
   const int s, const EQPSample &eqp_sample, ElementTransformation &T, const Vector &x, DenseMatrix &jac)
{
   const IntegrationPoint &ip = GetIntegrationRule()->IntPoint(eqp_sample.info.qp);
   const double qw = eqp_sample.info.qw;

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

}
