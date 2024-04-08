// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nlelast_integ.hpp"

using namespace std;
namespace mfem
{

double TestLinModel::EvalW(const DenseMatrix &J)
{MFEM_ABORT("TODO")};


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
   DenseMatrix tempadj(dim, dim);
   CalcAdjugate(Trans.Elem1->Jacobian(), tempadj);
   Mult(gh, tempadj, grad);
 
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
      gshape.GradToDiv (divshape);

      M = c_mu->Eval(Trans, ip);
 
      L = c_lambda->Eval(Trans, ip);

      AddMult_a_VVt(L * w, divshape, elmat);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
            {
               elmat (dof*d+k, dof*d+l) += (M * w) * pelmat(k, l);
            }
      }
      for (int ii = 0; ii < dim; ii++)
         for (int jj = 0; jj < dim; jj++)
         {
            for (int kk = 0; kk < dof; kk++)
               for (int ll = 0; ll < dof; ll++)
               {
                  elmat(dof*ii+kk, dof*jj+ll) +=
                     (M * w) * gshape(kk, jj) * gshape(ll, ii);
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

 // Boundary integrator
void DGHyperelasticNLFIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Trans,
                                 const Vector &elfun, Vector &elvect)
{
   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   //const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;
   int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0; // TEMP: Prevents resizing of elmat

   const int nvdofs = dim*(ndofs1 + ndofs2);

   // TODO: Assert ndofs1 == ndofs2

   Vector elfun_copy(elfun); // FIXME: How to avoid this?
    nor.SetSize(dim);
    Jrt.SetSize(dim);
   elvect.SetSize(nvdofs);
   elvect = 0.0;
   model->SetTransformation(Trans);

   //ndofs2 = 0; // TEMP: Prevents resizing of elmat

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

   //for (int i = 0; i < ir->GetNPoints(); i++)
   for (int i = 0; i < 1; i++)
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
      if (ndofs2)
      {
         w /= 2.0;

      el2.CalcShape(eip2, shape2);
      el2.CalcDShape(eip2, DSh2);
      Mult(DSh2, Jrt, DS2);
      MultAtB(PMatI2, DS2, Jpt2);

      model->EvalP(el2, eip2, PMatI2, Trans, P2);

      double w2 = w / Trans.Elem2->Weight();
      P2 *= w2;
      //_PrintMatrix(P2, "Pmat.txt");
      P2.Mult(nor, tau2);
      }

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, DSh1);
      Mult(DSh1, Jrt, DS1);
      MultAtB(PMatI1, DS1, Jpt1);

      model->EvalP(el1, eip1, PMatI1, Trans, P1);

      double w1 = w / Trans.Elem1->Weight();
      P1 *= w1;
      //_PrintMatrix(P1, "Pmat.txt");
      P1.Mult(nor, tau1);
      //_PrintVector(tau, "tauprint.txt");

      // Works
      for (int im = 0, i = 0; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs1; ++idof, ++i)
         {
         elvect(i) += shape1(idof) * tau1(im);
         }
      }

      if (ndofs2 == 0) {continue;}
       shape2.Neg();

      /* for (int im = 0, i = 0; im < dim; ++im)
      {
         for (int idof = 0; idof < ndofs1; ++idof, ++i)
         {
         //elvect(i) += shape1(idof) * tau2(im);

         }
      } */

      // Works
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
/* void DGHyperelasticNLFIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Trans,
                                 const Vector &elfun, Vector &elvect){
    const int dim = el1.GetDim();
    const int ndofs1 = el1.GetDof();
    const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;
    const int nvdofs = dim*(ndofs1 + ndofs2);

    // TODO: Assert that elvect and elfun are of size nvdofs.
    // TODO: Make sure that kappa works and is correct
    kappa = 0.0;
    const bool kappa_is_nonzero = (kappa != 0.0);
    if (kappa_is_nonzero)
    {
       jmat.SetSize(nvdofs);
       jmat = 0.;
    }
   Vector elfun_copy(elfun); // FIXME: How to avoid this?
    elvect.SetSize(nvdofs*dim);
    elvect = 0.0;
    nor.SetSize(dim);
    Jrt.SetSize(dim);
    elfun1.MakeRef(elfun_copy,0,ndofs1*dim);
    elvect1.MakeRef(elvect,0,ndofs1*dim);

    PMatI1.UseExternalData(elfun1.GetData(), ndofs1, dim);
    PMatO1.UseExternalData(elvect1.GetData(), ndofs1, dim);
    NorMat1.SetSize(ndofs1, dim);
    DSh1.SetSize(ndofs1, dim);
    DS1.SetSize(ndofs1, dim);
    Jpt1.SetSize(dim);
    P1.SetSize(dim);
   
    if (ndofs2)
    {
    elfun2.MakeRef(elfun_copy,0,ndofs2*dim);
    elvect2.MakeRef(elvect,0,ndofs2*dim);

    PMatI2.UseExternalData(elfun2.GetData(), ndofs2, dim);
    PMatO2.UseExternalData(elvect2.GetData(), ndofs2, dim);
    NorMat2.SetSize(ndofs2, dim);

    DSh2.SetSize(ndofs2, dim);
    DS2.SetSize(ndofs2, dim);
    Jpt2.SetSize(dim);
    P2.SetSize(dim);
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
 
       if (dim == 1)
       {
          nor(0) = 2*eip1.x - 1.0;
       }
       else
       {
          CalcOrtho(Trans.Jacobian(), nor);
       }

       // Fill normal matrix
       for (size_t i = 0; i < ndofs1; i++)
       {
         for (size_t j = 0; j < dim; j++)
         {
            NorMat1(i,j) = nor(j);
         } 
       }
       
       if (ndofs2)
       {
         for (size_t i = 0; i < ndofs2; i++)
       {
         for (size_t j = 0; j < dim; j++)
         {
            NorMat2(i,j) = nor(j);
         } 
       }
       }

       CalcInverse(Trans.Jacobian(), Jrt);
 
       double w;
       if (ndofs2)
       {
          w = ip.weight/2;
          const double w2 = w * Trans.Elem2->Weight();
          //Trans.SetIntPoint(&eip2); //Correct?
          el1.CalcDShape(eip2, DSh2);
          Mult(DSh2, Jrt, DS2);
          MultAtB(PMatI2, DS2, Jpt2);

          //model->EvalP(Jpt2, P2);
         model->EvalP(el2, eip2, PMatI2, Trans, P2);
         P2 *= w2;
         //AddMultABt(NorMat2,P2, PMatO2);

       }
       else
       {
          w = ip.weight;
       }
 
       {
          const double w1 = w * Trans.Elem1->Weight();
          //Trans.SetIntPoint(&ip); //Correct?
          el1.CalcDShape(ip, DSh1);
          Mult(DSh1, Jrt, DS1);
          MultAtB(PMatI1, DS1, Jpt1);

          //model->EvalP(Jpt, P);
         model->EvalP(el1, ip, PMatI1, Trans, P1);
 
         //P1 *= w1;
         for (size_t i = 0; i < dim; i++)
         {
            nor(i) /= Trans.Elem1->Weight();
            //nor(i) /= 0.25;
         } 

         cout<<"nor.Norml2() is: "<<nor.Norml2()<<endl;
         
         Vector tau(dim);
         P1.Mult(nor, tau);

         tau *= w1;
         shape1.SetSize(ndofs1);
         el1.CalcShape(eip1, shape1);

         for (int i = 0; i < ndofs1; i++)
          for (int j = 0; j < dim; j++)
          {
             PMatO1(i, j) += shape1(i) * tau(j);
          }

       }

    }
}; */

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
       model->AssembleH(Jpt, DS,ip.weight * Ttr.Weight(), elmat, el, ip, Ttr);
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
