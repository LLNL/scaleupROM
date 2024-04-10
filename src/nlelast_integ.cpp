// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nlelast_integ.hpp"

using namespace std;
namespace mfem
{

double TestLinModel::EvalW(const DenseMatrix &J)
{MFEM_ABORT("TODO")};

double TestLinModel::EvalwLM(const double w, ElementTransformation &Ttr, const IntegrationPoint &ip)
{
const double wL = w * c_lambda->Eval(Ttr, ip);
const double wM = w * c_mu->Eval(Ttr, ip);
return wL + 2.0*wM;
}

void TestLinModel::EvalP(
    const FiniteElement &el, const IntegrationPoint &ip, const DenseMatrix &PMatI, FaceElementTransformations &Trans, DenseMatrix &P)
{
   const int dof = el.GetDof();
    const int dim = el.GetDim();
    double L, M;
 
    DenseMatrix dshape(dof, dim);
    double gh_data[9], grad_data[9];
    DenseMatrix gh(gh_data, dim, dim);
    DenseMatrix grad(grad_data, dim, dim);

   el.CalcDShape(ip, dshape);
   MultAtB(PMatI, dshape, gh);

   Mult(gh, Trans.InverseJacobian(), grad);

   // The part below has been changed
   // TODO: This isn't correct in general
   DenseMatrix tempadj(dim, dim);
   CalcAdjugate(Trans.Elem1->Jacobian(), tempadj);
   Mult(gh, tempadj, grad);
   M = c_mu->Eval(Trans, ip);
   L = c_lambda->Eval(Trans, ip);

   DenseMatrix DS(dof, dim);
   Mult(dshape, tempadj, DS);

   P = 0.0;
   // Calculate divergence of strain tensor
   double e_div = 0.0;
   for (size_t i = 0; i < dim; i++)
   {
      for (size_t j = 0; j < dof; j++)
      {
         e_div += PMatI(j,i) * DS(j,i);
      }
   }
   
   // Fill stress tensor
   for (size_t i = 0; i < dim; i++)
   {
      for (size_t j = 0; j < dim; j++)
      {
         double temp = 0.0;
         for (size_t k = 0; k < dof; k++)
         {
            temp += PMatI(k,j) * DS(k,i);
            temp += PMatI(k,i) * DS(k,j);
         }
         temp *= M;
         
         P(i,j) = temp;
         if (i == j)
         {
            P(i,j) += L * e_div;
         }
         
      }
   }
}

void TestLinModel::EvalP(
    const FiniteElement &el, const IntegrationPoint &ip, const DenseMatrix &PMatI, ElementTransformation &Trans, DenseMatrix &P)
{
   const int dof = el.GetDof();
    const int dim = el.GetDim();
    double L, M;
 
    DenseMatrix dshape(dof, dim);
    double gh_data[9], grad_data[9];
    DenseMatrix gh(gh_data, dim, dim);
    DenseMatrix grad(grad_data, dim, dim);

   el.CalcDShape(ip, dshape);
   MultAtB(PMatI, dshape, gh);
   
   Mult(gh, Trans.InverseJacobian(), grad);
   
   M = c_mu->Eval(Trans, ip);

   L = c_lambda->Eval(Trans, ip);

   // stress = 2*M*e(u) + L*tr(e(u))*I, where
   //   e(u) = (1/2)*(grad(u) + grad(u)^T)
   const double M2 = 2.0*M;
   if (dim == 2)
   {
      L *= (grad(0,0) + grad(1,1));
      P(0,0) = M2*grad(0,0) + L;
      P(1,1)  = M2*grad(1,1) + L;
      P(1,0)  = M*(grad(0,1) + grad(1,0));
      P(0,1)  = M*(grad(0,1) + grad(1,0));
   }
   else if (dim == 3)
   {
      L *= (grad(0,0) + grad(1,1) + grad(2,2));
      P(0,0) = M2*grad(0,0) + L;
      P(1,1) = M2*grad(1,1) + L;
      P(2,2) = M2*grad(2,2) + L;
      P(0,1) = M*(grad(0,1) + grad(1,0));
      P(1,0) = M*(grad(0,1) + grad(1,0));
      P(0,2) = M*(grad(0,2) + grad(2,0));
      P(2,0) = M*(grad(0,2) + grad(2,0));
      P(1,2) = M*(grad(1,2) + grad(2,1));
      P(2,1) = M*(grad(1,2) + grad(2,1));
   }
}

void TestLinModel::EvalDmat(const int dim, const int dof, const IntegrationPoint ip, FaceElementTransformations &Trans, 
const DenseMatrix gshape, DenseMatrix &Dmat)
{
   double M = c_mu->Eval(Trans, ip);
   double L = c_lambda->Eval(Trans, ip);
for (size_t i = 0; i < dim; i++) 
      {
         for (size_t j = 0; j < dim; j++) // Looping over each entry in residual
         {
            const int S_ij = j * dim + i;

            for (size_t m = 0; m < dof; m++) 
            for (size_t n = 0; n < dim; n++) // Looping over derivatives with respect to U
            {
               const int U_mn = n * dof + m;
               Dmat(S_ij, U_mn) = ((i == j) ? L * gshape(m,n) : 0.0);
               Dmat(S_ij, U_mn) += ((n == i) ? M * gshape(m,j) : 0.0);
               Dmat(S_ij, U_mn) += ((n == j) ? M * gshape(m,i) : 0.0);
            }
         }
      }
}

void TestLinModel::EvalDmat(const int dim, const int dof, const IntegrationPoint ip, ElementTransformation &Trans, 
const DenseMatrix gshape, DenseMatrix &Dmat)
{
   double M = c_mu->Eval(Trans, ip);
   double L = c_lambda->Eval(Trans, ip);
for (size_t i = 0; i < dim; i++) 
      {
         for (size_t j = 0; j < dim; j++) // Looping over each entry in residual
         {
            const int S_ij = j * dim + i;

            for (size_t m = 0; m < dof; m++) 
            for (size_t n = 0; n < dim; n++) // Looping over derivatives with respect to U
            {
               const int U_mn = n * dof + m;
               Dmat(S_ij, U_mn) = ((i == j) ? L * gshape(m,n) : 0.0);
               Dmat(S_ij, U_mn) += ((n == i) ? M * gshape(m,j) : 0.0);
               Dmat(S_ij, U_mn) += ((n == j) ? M * gshape(m,i) : 0.0);
            }
         }
      }
}

void TestLinModel::AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                     const double w, DenseMatrix &elmat, const FiniteElement &el, const IntegrationPoint &ip,ElementTransformation &Trans)
      {
      const int dof = el.GetDof();
      const int dim = el.GetDim();
      double L, M;
      DenseMatrix dshape(dof, dim), gshape(dof, dim), pelmat(dof);
      Vector divshape(dim*dof);

      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      Mult(dshape, Trans.InverseJacobian(), gshape);
      MultAAt(gshape, pelmat);
      gshape.GradToDiv(divshape);

      DenseMatrix Dmat(dim * dim, dim * dof);
      EvalDmat(dim, dof, ip, Trans, gshape, Dmat);
   
      //Assemble elmat
      for (size_t i = 0; i < dof; i++) 
      {
         for (size_t j = 0; j < dim; j++) // Looping over each entry in residual
         {
            const int ij = j * dof + i;

            for (size_t m = 0; m < dof; m++) 
            for (size_t n = 0; n < dim; n++) // Looping over derivatives with respect to U
            {
               const int mn = n * dof + m;
               double temp = 0.0;
               for (size_t k = 0; k < dim; k++)
               {
                  const int S_jk = k * dim + j;
                  temp += Dmat(S_jk, mn) * w * gshape(i,k);
               }
               elmat(ij, mn) += temp;
               
            }
         }
      }

       }; 
void _PrintMatrix(const DenseMatrix &mat,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < mat.NumRows(); i++)
   {
      for (int j = 0; j < mat.NumCols(); j++)
         fprintf(fp, "%.15E\t", mat(i,j));
      fprintf(fp, "\n");
   }

   fclose(fp);
   return;
}

void _PrintVector(const Vector &vec,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < vec.Size(); i++)
      fprintf(fp, "%.15E\n", vec(i));

   fclose(fp);
   return;
}

void DGHyperelasticNLFIntegrator::AssembleJmat(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset, const Vector &row_shape,
       const Vector &col_shape, const double jmatcoef,DenseMatrix &jmat){
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

 // Boundary integrator
void DGHyperelasticNLFIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Trans,
                                 const Vector &elfun, Vector &elvect)
{
   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;

   const int nvdofs = dim*(ndofs1 + ndofs2);

   // TODO: Assert ndofs1 == ndofs2

   Vector elfun_copy(elfun); // FIXME: How to avoid this?
    nor.SetSize(dim);
    Jrt.SetSize(dim);
   elvect.SetSize(nvdofs);
   elvect = 0.0;

   const bool kappa_is_nonzero = (kappa != 0.0);
    if (kappa_is_nonzero)
    {
       jmat.SetSize(nvdofs);
       jmat = 0.;
    }

   model->SetTransformation(Trans);

   shape1.SetSize(ndofs1);
    elfun1.MakeRef(elfun_copy,0,ndofs1*dim);
    elvect1.MakeRef(elvect,0,ndofs1*dim);
    PMatI1.UseExternalData(elfun1.GetData(), ndofs1, dim);
    DSh1.SetSize(ndofs1, dim);
    DS1.SetSize(ndofs1, dim);
    Jpt1.SetSize(dim);
    P1.SetSize(dim);

    if (ndofs2)
    {
   shape2.SetSize(ndofs2);
      elfun2.MakeRef(elfun_copy,ndofs1*dim,ndofs2*dim);
      elvect2.MakeRef(elvect,ndofs1*dim,ndofs2*dim);
      PMatI2.UseExternalData(elfun2.GetData(), ndofs2, dim);
      DSh2.SetSize(ndofs2, dim);
      DS2.SetSize(ndofs2, dim);
      Jpt2.SetSize(dim);
      P2.SetSize(dim);
    }
    
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el1.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), intorder);
   }

   // TODO: Add to class
   Vector tau1(dim);
   Vector tau2(dim);

   Vector big_row1(dim*ndofs1);
   Vector big_row2(dim*ndofs2);

   //for (int i = 0; i < 1; i++)
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      CalcInverse(Trans.Jacobian(), Jrt);

      double w = ip.weight;
      double wLM = 0.0;
      if (ndofs2)
      {
      w /= 2.0;

      el2.CalcShape(eip2, shape2);
      el2.CalcDShape(eip2, DSh2);
      Mult(DSh2, Jrt, DS2);
      MultAtB(PMatI2, DS2, Jpt2);

      model->EvalP(el2, eip2, PMatI2, Trans, P2);

      double w2 = w / Trans.Elem2->Weight();
      wLM = model->EvalwLM(w2, *Trans.Elem2, eip2);
      P2 *= w2;
      P2.Mult(nor, tau2);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, DSh1);
      Mult(DSh1, Jrt, DS1);
      MultAtB(PMatI1, DS1, Jpt1);

      model->EvalP(el1, eip1, PMatI1, Trans, P1);

      double w1 = w / Trans.Elem1->Weight();
      wLM += model->EvalwLM(w1, *Trans.Elem1, eip1);
      P1 *= w1;
      P1.Mult(nor, tau1);

      const double jmatcoef = kappa * (nor*nor) * wLM;

      // (1,1) block
      for (int im = 0, i = 0; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs1; ++idof, ++i)
         {
         elvect(i) += shape1(idof) * tau1(im);
         }
      } 
      if (ndofs2 != 0) {
      shape2.Neg();
      }

      if (kappa_is_nonzero)
    {
       jmat = 0.;
       AssembleJmat(
       dim, ndofs1, ndofs1, 0, 0, shape1,
       shape1, jmatcoef, jmat);
       if (ndofs2 != 0) {
       AssembleJmat(
       dim, ndofs1, ndofs2, 0, ndofs1*dim, shape1,
       shape2, jmatcoef, jmat);
       AssembleJmat(
       dim, ndofs2, ndofs1, ndofs1*dim, 0, shape2,
       shape1, jmatcoef, jmat);
       AssembleJmat(
       dim, ndofs2, ndofs2, ndofs1*dim, ndofs1*dim, shape2,
       shape2, jmatcoef, jmat);
       }

       //Flatten jmat
       for (int i = 0; i < nvdofs; ++i)
       {
          for (int j = 0; j < i; ++j)
          {
             jmat(j,i) = jmat(i,j);
          }
       } 
       // Apply jmat
       for (size_t i = 0; i < nvdofs; i++)
       {
            for (size_t j = 0; j < nvdofs; j++)
         {
            elvect(i) -= jmat(i,j) * elfun(j);
         }
       }
    }

      if (ndofs2 == 0) {continue;}
       

      // (1,2) block
      for (int im = 0, i = 0; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs1; ++idof, ++i)
         {
         elvect(i) += shape1(idof) * tau2(im);
         }
      }

      // (2,1) block
      for (int im = 0, i = ndofs1*dim; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs2; ++idof, ++i)
         {
         elvect(i) += shape2(idof) * tau1(im);
         }
      }

      // (2,2) block
      for (int im = 0, i = ndofs1*dim; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs2; ++idof, ++i)
         {
         elvect(i) += shape2(idof) * tau2(im);
         }
      }
   
      }

   elvect *= -1.0;

}

void DGHyperelasticNLFIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Tr,
                              const Vector &elfun, DenseMatrix &elmat){
const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (Tr.Elem2No >= 0) ? el2.GetDof() : 0;

   const int nvdofs = dim*(ndofs1 + ndofs2);

   Vector elfun_copy(elfun); // FIXME: How to avoid this?
    nor.SetSize(dim);
    Jrt.SetSize(dim);
   elmat.SetSize(nvdofs);
   elmat = 0.0;
   model->SetTransformation(Tr);

   const bool kappa_is_nonzero = (kappa != 0.0);
    if (kappa_is_nonzero)
    {
       jmat.SetSize(nvdofs);
       jmat = 0.;
    }

   shape1.SetSize(ndofs1);
    elfun1.MakeRef(elfun_copy,0,ndofs1*dim);
    PMatI1.UseExternalData(elfun1.GetData(), ndofs1, dim);
    DSh1.SetSize(ndofs1, dim);
    DS1.SetSize(ndofs1, dim);
    Jpt1.SetSize(dim);
    P1.SetSize(dim);

    if (ndofs2)
    {
   shape2.SetSize(ndofs2);
      elfun2.MakeRef(elfun_copy,ndofs1*dim,ndofs2*dim);
      PMatI2.UseExternalData(elfun2.GetData(), ndofs2, dim);
      DSh2.SetSize(ndofs2, dim);
      DS2.SetSize(ndofs2, dim);
      Jpt2.SetSize(dim);
      P2.SetSize(dim);
    }
    

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el1.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   }

   // TODO: Add to class
   Vector tau1(dim);
   Vector tau2(dim);

   Vector wnor1(dim);
   Vector wnor2(dim);

   Vector big_row1(dim*ndofs1);
   Vector big_row2(dim*ndofs2);

   DenseMatrix Dmat1(dim * dim,dim*ndofs1);
   DenseMatrix Dmat2(dim * dim,dim*ndofs2);

   Dmat1 = 0.0;
   Dmat2 = 0.0;

   DenseMatrix adjJ2(dim);
   
   DenseMatrix dshape1_ps(ndofs1, dim);
   DenseMatrix dshape2_ps(ndofs2, dim);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      CalcInverse(Tr.Jacobian(), Jrt);

      double w = ip.weight;
      double wLM = 0.0;
      if (ndofs2)
      {
      w /= 2.0;
      el2.CalcShape(eip2, shape2);
      el2.CalcDShape(eip2, DSh2);
      Mult(DSh2, Jrt, DS2);

      CalcAdjugate(Tr.Elem2->Jacobian(), adjJ2);
      Mult(DSh2, adjJ2, dshape2_ps);

      //model->EvalDmat(dim, ndofs1, eip2, Tr, DS2, Dmat2);
      model->EvalDmat(dim, ndofs2, eip2, Tr, dshape2_ps, Dmat2);
      double w2 = w / Tr.Elem2->Weight();
      wLM = model->EvalwLM(w2, *Tr.Elem2, eip2);
      wnor2.Set(w2,nor);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, DSh1);
      Mult(DSh1, Jrt, DS1);

      double w1 = w / Tr.Elem1->Weight();
      wLM += model->EvalwLM(w1, *Tr.Elem1, eip1);

      // Temporary stuff
      DenseMatrix adjJ1(dim);
      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ1);
      Mult(DSh1, adjJ1, dshape1_ps);

      model->EvalDmat(dim, ndofs1, eip1, Tr, dshape1_ps, Dmat1);

      const double jmatcoef = kappa * (nor*nor) * wLM;

      // (1,1) block //works
      wnor1.Set(w1,nor);
      AssembleBlock(dim, ndofs1, ndofs1, 0, 0, shape1, shape1, jmatcoef, wnor1,Dmat1, elmat,jmat);
 
      if (ndofs2 == 0) {continue;}
       shape2.Neg(); 

       // (1,2) block works
      AssembleBlock(dim, ndofs1, ndofs2, 0, dim*ndofs1, shape1, shape2,jmatcoef,wnor2, Dmat2, elmat,jmat);

       // (2,1) block
      AssembleBlock(dim, ndofs2, ndofs1, dim*ndofs1, 0, shape2, shape1,jmatcoef,wnor1, Dmat1, elmat,jmat); 

       // (2,2) block
      AssembleBlock(dim, ndofs2, ndofs2, dim*ndofs1, dim*ndofs1, shape2, shape2,jmatcoef,wnor2, Dmat2, elmat,jmat);

   }

   // elmat := -elmat + jmat
    elmat *= -1.0;
    if (kappa_is_nonzero)
    {
       for (int i = 0; i < nvdofs; ++i)
       {
          for (int j = 0; j < i; ++j)
          {
             double mij = jmat(i,j);
             elmat(i,j) += mij;
             elmat(j,i) += mij;
          }
          elmat(i,i) += jmat(i,i);
       } 
    }
   };

void DGHyperelasticNLFIntegrator::AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset, const Vector &row_shape,
       const Vector &col_shape, const double jmatcoef,
       const Vector &wnor, const DenseMatrix &Dmat, DenseMatrix &elmat, DenseMatrix &jmat){
for (int n = 0, jj = col_offset; n < dim; ++n)
    {
       for (int m = 0; m < col_ndofs; ++m, ++jj)
       {
         const int mn = n * col_ndofs + m;
          for (int j = 0, ii = row_offset; j < dim; ++j)
          {
             for (int i = 0; i < row_ndofs; ++i, ++ii)
             {
               double temp = 0.0;
               for (size_t k = 0; k < dim; k++)
               {
                  const int S_jk = k * dim + j;
                  temp += Dmat(S_jk, mn) * wnor(k);
               }
                elmat(ii, jj) += row_shape(i) * temp;
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

// Domain integrator
void HyperelasticNLFIntegratorHR::AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Ttr,
                                      const Vector &elfun,
                                      Vector &elvect){
       int dof = el.GetDof(), dim = el.GetDim();
 
    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    Jrt.SetSize(dim);
    Jpt.SetSize(dim);
    P.SetSize(dim);
    PMatI.UseExternalData(elfun.GetData(), dof, dim);
    elvect.SetSize(dof*dim);
    PMatO.UseExternalData(elvect.GetData(), dof, dim);
 
    const IntegrationRule *ir = IntRule;
    if (!ir)
    {
       ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
    }
 
    elvect = 0.0;
    model->SetTransformation(Ttr);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Ttr.SetIntPoint(&ip);
       CalcInverse(Ttr.Jacobian(), Jrt);
 
       el.CalcDShape(ip, DSh);
       Mult(DSh, Jrt, DS);
       MultAtB(PMatI, DS, Jpt);
 
       //model->EvalP(Jpt, P);
       model->EvalP(el, ip, PMatI, Ttr, P);
 
       P *= ip.weight * Ttr.Weight();
       AddMultABt(DS, P, PMatO);
    }
       }

void HyperelasticNLFIntegratorHR::AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Ttr,
                                    const Vector &elfun,
                                    DenseMatrix &elmat){
int dof = el.GetDof(), dim = el.GetDim();
 
    DSh.SetSize(dof, dim);
    DS.SetSize(dof, dim);
    Jrt.SetSize(dim);
    Jpt.SetSize(dim);
    PMatI.UseExternalData(elfun.GetData(), dof, dim);
    elmat.SetSize(dof*dim);
 
    const IntegrationRule *ir = IntRule;
    if (!ir)
    {
       ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
    }
 
    elmat = 0.0;
    model->SetTransformation(Ttr);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Ttr.SetIntPoint(&ip);
       CalcInverse(Ttr.Jacobian(), Jrt);
 
       el.CalcDShape(ip, DSh);
       Mult(DSh, Jrt, DS);
       MultAtB(PMatI, DS, Jpt);
 
       //model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
       model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat, el, ip, Ttr);
    }
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

    shape.SetSize(ndofs);
    nor.SetSize(dim);
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
 
       if (dim == 1)
       {
          nor(0) = 2*eip.x - 1.0;
       }
       else
       {
          CalcOrtho(Tr.Jacobian(), nor);
       }
 
       double jcoef;
       double wLM;

      const double w = ip.weight / Tr.Elem1->Weight();
      wLM = model->EvalwLM(w, *Tr.Elem1, eip);
      jcoef = kappa * wLM * (nor*nor);

 
       for (int im = 0, i = 0; im < dim; ++im)
       {
          const double tj = jcoef * u_dir(im);
          for (int idof = 0; idof < ndofs; ++idof, ++i)
          {
             elvect(i) += tj*shape(idof);
          }
       }
    }
};

} // namespace mfem
