// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <random>
#include "mms_suite.hpp"
#include "nlelast_integ.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{
 class Test_DGElasticityIntegrator : public BilinearFormIntegrator
 {
 public:
    Test_DGElasticityIntegrator(double alpha_, double kappa_)
       : lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) { }
 
    Test_DGElasticityIntegrator(Coefficient &lambda_, Coefficient &mu_,
                           double alpha_, double kappa_)
       : lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }
 
    using BilinearFormIntegrator::AssembleFaceMatrix;
    virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                    const FiniteElement &el2,
                                    FaceElementTransformations &Trans,
                                    DenseMatrix &elmat);
 
 protected:
    Coefficient *lambda, *mu;
    double alpha, kappa;
 
 #ifndef MFEM_THREAD_SAFE
    // values of all scalar basis functions for one component of u (which is a
    // vector) at the integration point in the reference space
    Vector shape1, shape2;
    // values of derivatives of all scalar basis functions for one component
    // of u (which is a vector) at the integration point in the reference space
    DenseMatrix dshape1, dshape2;
    // Adjugate of the Jacobian of the transformation: adjJ = det(J) J^{-1}
    DenseMatrix adjJ;
    // gradient of shape functions in the real (physical, not reference)
    // coordinates, scaled by det(J):
    //    dshape_ps(jdof,jm) = sum_{t} adjJ(t,jm)*dshape(jdof,t)
    DenseMatrix dshape1_ps, dshape2_ps;
    Vector nor;  // nor = |weight(J_face)| n
    Vector nL1, nL2;  // nL1 = (lambda1 * ip.weight / detJ1) nor
    Vector nM1, nM2;  // nM1 = (mu1     * ip.weight / detJ1) nor
    Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
    // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
    DenseMatrix jmat;
 #endif
 
    static void AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset,
       const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
       const Vector &row_shape, const Vector &col_shape,
       const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
       DenseMatrix &elmat, DenseMatrix &jmat);
 };

 void _AssembleBlock(
    const int dim, const int row_ndofs, const int col_ndofs,
    const int row_offset, const int col_offset,
    const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
    const Vector &row_shape, const Vector &col_shape,
    const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
    DenseMatrix &elmat, DenseMatrix &jmat)
 {
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

 void Test_DGElasticityIntegrator::AssembleFaceMatrix(
    const FiniteElement &el1, const FiniteElement &el2,
    FaceElementTransformations &Trans, DenseMatrix &elmat)
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
 
    //for (int pind = 0; pind < 1; ++pind)
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
      cout<<"jmatcoef is: "<<jmatcoef<<endl;

 
       // (1,1) block
       _AssembleBlock(
          dim, ndofs1, ndofs1, 0, 0, jmatcoef, nL1, nM1,
          shape1, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);
 
       if (ndofs2 == 0) { continue; }
 
       // In both elmat and jmat, shape2 appears only with a minus sign.
       shape2.Neg();
 
       // (1,2) block
       _AssembleBlock(
          dim, ndofs1, ndofs2, 0, dim*ndofs1, jmatcoef, nL2, nM2,
          shape1, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
       // (2,1) block
       _AssembleBlock(
          dim, ndofs2, ndofs1, dim*ndofs1, 0, jmatcoef, nL1, nM1,
          shape2, shape1, dshape1_dnM, dshape1_ps, elmat, jmat);
       // (2,2) block
       _AssembleBlock(
          dim, ndofs2, ndofs2, dim*ndofs1, dim*ndofs1, jmatcoef, nL2, nM2,
          shape2, shape2, dshape2_dnM, dshape2_ps, elmat, jmat);
    }
 
    // elmat := -elmat + alpha*elmat^t + jmat
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
   PrintMatrix(elmat, "checkmat.txt");
 }
/*  PrintMatrix(elmat, "checkmat.txt");
 } */
} // namespace mfem
/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

void InitDisplacement(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0);
}


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

// Currently Domain test and Boundary test works. Todo: RHS, Boundary gradients
TEST(TempLinStiffnessMatrices, Test_NLElast)
{
   // Test that the nonlinear operators do the correct things
    Mesh mesh("meshes/test.2x1.mesh", 1, 1);

   int dim = mesh.Dimension();
   int order = 2;
   double alpha = 0.0; // IIPG
   double kappa = (order+1)*(order+1); 
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   VectorFunctionCoefficient init_x(dim, InitDisplacement);

   Vector lambda(mesh.attributes.Max());
   double _lambda = 3.14;      
   lambda = _lambda;      // Set lambda for all element attributes.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   double _mu = 2.33;      
   mu = _mu;       // Set mu = 1 for all element attributes.
   PWConstCoefficient mu_c(mu);

    Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   BilinearForm a1(&fespace);
   a1.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
   a1.AddInteriorFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa));
   a1.AddBdrFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);
   a1.Assemble();
   cout<<"a1.Height() is: "<<a1.Height()<<endl;

   LinElastMaterialModel model(_mu, _lambda);
   NonlinearForm a2(&fespace);
   a2.AddDomainIntegrator(new HyperelasticNLFIntegratorHR(&model));
   a2.AddInteriorFaceIntegrator(
      new DGHyperelasticNLFIntegrator(&model, alpha, kappa));
   a2.AddBdrFaceIntegrator(
      new DGHyperelasticNLFIntegrator(&model, alpha, kappa), dir_bdr);

      /* BilinearForm a2(&fespace);
   //a1.AddDomainIntegrator(new ElasticityIntegrator(lambda_c, mu_c));
   //a2.AddInteriorFaceIntegrator(
     // new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa)); //Needed??
   a2.AddBdrFaceIntegrator(
      new DGElasticityIntegrator(lambda_c, mu_c, alpha, kappa), dir_bdr);
   a2.Assemble(); */
    
    // Create vectors to hold the values of the forms
    Vector x, y1, y2;

    x.SetSize(fespace.GetTrueVSize());

    double lower_bound = -1;
    double upper_bound = 1;
 
    uniform_real_distribution<double> unif(lower_bound,
                                           upper_bound);
    default_random_engine re;

    for (size_t i = 0; i < x.Size(); i++)
    {
      x[i] = unif(re);
      //x[i] = 1.0;
      //x[i] = 0.0;
      //x[i] = i;
    }
    
    y1.SetSize(fespace.GetTrueVSize());
    y1 = 0.0;

    y2.SetSize(fespace.GetTrueVSize());
    y2 = 0.0;

    a1.Mult(x, y1);
    a2.Mult(x, y2);

   double norm_diff = 0.0;
   cout << "Linear residual norm: " << y1.Norml2() << endl;
    cout << "Nonlinear residual norm: " << y2.Norml2() << endl;

    y1 -= y2;
    norm_diff = y1.Norml2();
    cout << "Residual difference norm: " << norm_diff << endl;

    a1.Mult(x, y1);
    a2.Mult(x, y2);

    y1/=y1.Norml2();
    y2/=y2.Norml2();

    cout << "Scaled Linear residual norm: " << y1.Norml2() << endl;
    cout << "Scaled Nonlinear residual norm: " << y2.Norml2() << endl;

    y1 -= y2;
    norm_diff = y1.Norml2();
    cout << "Scaled Residual difference norm: " << norm_diff << endl;

    Operator *J_op = &(a2.GetGradient(x));
    SparseMatrix *J = dynamic_cast<SparseMatrix *>(J_op);

    SparseMatrix diff_matrix(*J);
   _PrintMatrix(*(diff_matrix.ToDenseMatrix()), "test_elmat1.txt");
   SparseMatrix a1view(a1.SpMat());
   a1view.Finalize();
   _PrintMatrix(*(a1view.ToDenseMatrix()), "test_elmat2.txt");

    diff_matrix.Add(-1.0, a1.SpMat());
    
    norm_diff = diff_matrix.MaxNorm();

    cout << "Nonlinear Stiffness matrix norm: " << J->MaxNorm() << endl;
    cout << "Linear Stiffness matrix norm: " << a1.SpMat().MaxNorm() << endl;
    cout << "Stiffness matrix difference norm: " << norm_diff << endl;

    LinearForm b1(&fespace);
   b1.AddBdrFaceIntegrator(
      new DGElasticityDirichletLFIntegrator(
         init_x, lambda_c, mu_c, alpha, kappa), dir_bdr);
   b1.Assemble();

   LinearForm b2(&fespace);
   b2.AddBdrFaceIntegrator(
      new DGHyperelasticDirichletLFIntegrator(
         init_x, &model, alpha, kappa), dir_bdr);
   b2.Assemble();

    cout << "Linear RHS norm: " << b1.Norml2() << endl;
    cout << "Nonlinear RHS norm: " << b2.Norml2() << endl;

    b1 -= b2;
    norm_diff = b1.Norml2();

    // Print the norm of the difference
    cout << "RHS difference norm: " << norm_diff << endl;

   return;
} 

   void ExactSolutionNeoHooke(const Vector &X, Vector &U)
   {
      int dim = 2;
      U = 0.0;
      for (size_t i = 0; i < U.Size()/dim; i++)
      {
         U(i*dim) = pow(X(i*dim), 2.0) + X(i*dim);
         U(i*dim+1) = pow(X(i*dim + 1), 2.0) + X(i*dim + 1);
      }
   }

   void ExactRHSNeoHooke(const Vector &X, Vector &U)
   {
      int dim = 2;
      double K = 3.14; 
      double mu = 2.33;
      U = 0.0;
      for (size_t i = 0; i < U.Size()/dim; i++)
      {
         const double x_1 = X(i*dim);
         const double x_2 = X(i*dim + 1);
         U(i*dim) = (128.0*K + 128.0*mu + 1024.0*K*x_1 + 1024.0*K*x_2 + 128.0*mu*x_1 + 384.0*mu*x_2 + 3072.0*K*(pow(x_1,2)) + 8192.0*K*x_1*x_2 + 3072.0*K*(pow(x_2,2)) + 128.0*mu*(pow(x_1,2)) + 384.0*mu*(pow(x_2,2)) + 4096.0*K*(pow(x_1,3)) + 24576.0*K*(pow(x_1,2))*x_2 + 24576.0*K*x_1*(pow(x_2,2)) + 4096.0*K*(pow(x_2,3)) + 2048.0*K*(pow(x_1,4)) + 32768.0*K*(pow(x_1,3))*x_2 + 73728.0*K*(pow(x_1,2))*(pow(x_2,2)) + 32768.0*K*x_1*(pow(x_2,3)) + 2048.0*K*(pow(x_2,4)) + 16384.0*K*(pow(x_1,4))*x_2 + 98304.0*K*(pow(x_1,3))*(pow(x_2,2)) + 98304.0*K*(pow(x_1,2))*(pow(x_2,3)) + 16384.0*K*x_1*(pow(x_2,4)) + 49152.0*K*(pow(x_1,4))*(pow(x_2,2)) + 131072.0*K*(pow(x_1,3))*(pow(x_2,3)) + 49152.0*K*(pow(x_1,2))*(pow(x_2,4)) + 65536.0*K*(pow(x_1,4))*(pow(x_2,3)) + 65536.0*K*(pow(x_1,3))*(pow(x_2,4)) + 32768.0*K*(pow(x_1,4))*(pow(x_2,4))) / (64.0 * (pow((1 + 2*x_1), 4))*(pow((1 + 2*x_2),2)));
         U(i*dim+1) = (128.0*K + 128.0*mu + 1024.0*K*x_1 + 1024.0*K*x_2 + 384.0*mu*x_1 + 128.0*mu*x_2 + 3072.0*K*(pow(x_1,2)) + 8192.0*K*x_1*x_2 + 3072.0*K*(pow(x_2,2)) + 384.0*mu*(pow(x_1,2)) + 128.0*mu*(pow(x_2,2)) + 4096.0*K*(pow(x_1,3)) + 24576.0*K*(pow(x_1,2))*x_2 + 24576.0*K*x_1*(pow(x_2,2)) + 4096.0*K*(pow(x_2,3)) + 2048.0*K*(pow(x_1,4)) + 32768.0*K*(pow(x_1,3))*x_2 + 73728.0*K*(pow(x_1,2))*(pow(x_2,2)) + 32768.0*K*x_1*(pow(x_2,3)) + 2048.0*K*(pow(x_2,4)) + 16384.0*K*(pow(x_1,4))*x_2 + 98304.0*K*(pow(x_1,3))*(pow(x_2,2)) + 98304.0*K*(pow(x_1,2))*(pow(x_2,3)) + 16384.0*K*x_1*(pow(x_2,4)) + 49152.0*K*(pow(x_1,4))*(pow(x_2,2)) + 131072.0*K*(pow(x_1,3))*(pow(x_2,3)) + 49152.0*K*(pow(x_1,2))*(pow(x_2,4)) + 65536.0*K*(pow(x_1,4))*(pow(x_2,3)) + 65536.0*K*(pow(x_1,3))*(pow(x_2,4)) + 32768.0*K*(pow(x_1,4))*(pow(x_2,4))) / (64.0 * (pow((1 + 2*x_1),2))*(pow((1 + 2*x_2),4)));
      }
      
      //u *= -1.0;
   }

   void SimpleExactSolutionNeoHooke(const Vector &X, Vector &U)
   {
      int dim = 2;
      int dof = U.Size()/dim;
      U = 0.0;
      for (size_t i = 0; i < U.Size()/dim; i++)
      {
         U(i) = pow(X( i), 2.0) + X( i);
         U(dof + i) = pow(X(dof + i), 2.0) + X(dof + i);
      }
   }

   void SimpleExactRHSNeoHooke(const Vector &X, Vector &U)
   {
      int dim = 2;
      int dof = U.Size()/dim;

      double K = 3.14; 
      //double mu = 2.33;
/*       K = 2.33;
      mu = 3.14; */
      U = 0.0;
         for (size_t i = 0; i < U.Size()/dim; i++)
      {
         U(i) = K;
         U(dof + i) = K/2.0;
      }

      
      //u *= -1.0;
   }

// Currently Domain test and Boundary test works. Todo: RHS, Boundary gradients
TEST(TempNeoHookeanStiffnessMatrices, Test_NLElast)
{
   // Test that the nonlinear operators do the correct things
    Mesh mesh("meshes/test.2x1.mesh", 1, 1);

   int dim = mesh.Dimension();
   int order = 1;
   double alpha = 0.0; // IIPG
   double kappa = -1.0; 
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fespace(&mesh, &fec, dim);

   VectorFunctionCoefficient init_x(dim, InitDisplacement);

   Vector lambda(mesh.attributes.Max());
   double _lambda = 3.14;      
   lambda = _lambda;      // Set lambda for all element attributes.
   PWConstCoefficient lambda_c(lambda);
   Vector mu(mesh.attributes.Max());
   double _mu = 2.33;  
   //_mu = 0.0;    
   mu = _mu;       // Set mu = 1 for all element attributes.
   PWConstCoefficient mu_c(mu);

    Array<int> dir_bdr(mesh.bdr_attributes.Max());
   dir_bdr = 0;
   dir_bdr[0] = 1; // boundary attribute 1 is Dirichlet
   dir_bdr[1] = 1; // boundary attribute 2 is Dirichlet

   NeoHookeanHypModel model1(_mu, _lambda);
   NonlinearForm a1(&fespace);
   a1.AddDomainIntegrator(new HyperelasticNLFIntegratorHR(&model1));

   NeoHookeanModel model2(_mu, _lambda);
   NonlinearForm a2(&fespace);
   a2.AddDomainIntegrator(new HyperelasticNLFIntegrator(&model2));

    // Create vectors to hold the values of the forms
    Vector x, y0, y1, y2, y3;

    GridFunction x_ref(&fespace);
    mesh.GetNodes(x_ref);
    x.SetSize(fespace.GetTrueVSize());
    x = x_ref.GetTrueVector();

    double lower_bound = -1;
    double upper_bound = 1;
 
    //uniform_real_distribution<double> unif(lower_bound,
    //                                       upper_bound);
    //default_random_engine re;
    /* cout<<"couting xi"<<endl;
    for (size_t i = 0; i < x.Size(); i++)
    {
      cout<<"x[i] is: "<<x[i]<<endl;
    } */
    
    y0.SetSize(fespace.GetTrueVSize());
    y0 = 0.0;

    y1.SetSize(fespace.GetTrueVSize());
    y1 = 0.0;

    y2.SetSize(fespace.GetTrueVSize());
    y2 = 0.0;

    y3.SetSize(fespace.GetTrueVSize());
    y3 = 0.0;

    SimpleExactSolutionNeoHooke(x, y0);
    //y0 = x;
    SimpleExactRHSNeoHooke(y0, y3);

    cout<<"couting y0 and x"<<endl;
    for (size_t i = 0; i < y0.Size(); i++)
    {
      cout<<"x["<<i<<"] is: "<<x[i]<<endl;
      cout<<"y0["<<i<<"] is: "<<y0[i]<<endl<<endl;
    } 

    a1.Mult(y0, y1); //ScaleupROM Neohookean
    a2.Mult(y0, y2); //MFEM Neohookean

    cout<<"couting y1 and y3"<<endl;
    for (size_t i = 0; i < y0.Size(); i++)
    {
      cout<<"y1["<<i<<"] is: "<<y1[i]<<endl;
      cout<<"y3["<<i<<"] is: "<<y3[i]<<endl<<endl;
    }

   double norm_diff = 0.0;
   cout << "Scaleup NeoHooke residual norm: " << y1.Norml2() << endl;
    cout << "MFEM NeoHooke residual norm: " << y2.Norml2() << endl;
    cout << "Analytic NeoHooke residual norm: " << y3.Norml2() << endl;

    y1 -= y2; // Already tested y1 ==y2
    norm_diff = y1.Norml2();
    cout << "NeoHooke residual difference norm: " << norm_diff << endl;

    /* a1.Mult(x, y1);
    a2.Mult(x, y2);

    y1/=y1.Norml2();
    y2/=y2.Norml2();

    cout << "Scaled Scaleup NeoHooke residual norm: " << y1.Norml2() << endl;
    cout << "Scaled MFEM NeoHooke residual norm: " << y2.Norml2() << endl;

    y1 -= y2;
    norm_diff = y1.Norml2();
    cout << "Scaled NeoHooke Residual difference norm: " << norm_diff << endl; */

    Operator *J_op1 = &(a1.GetGradient(x));
    SparseMatrix *J1 = dynamic_cast<SparseMatrix *>(J_op1);

    Operator *J_op2 = &(a2.GetGradient(x));
    SparseMatrix *J2 = dynamic_cast<SparseMatrix *>(J_op2);

   //_PrintMatrix(*(diff_matrix.ToDenseMatrix()), "test_elmat1.txt");
   //_PrintMatrix(*(a1view.ToDenseMatrix()), "test_elmat2.txt");

    cout << "Scaleup NeoHooke matrix norm: " << J1->MaxNorm() << endl;
    cout << "MFEM NeoHooke matrix norm: " << J2->MaxNorm() << endl;

    J1->Add(-1.0, *J2);
    cout << "Stiffness matrix difference norm: " << J1->MaxNorm() << endl;

   return;
} 
 

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}