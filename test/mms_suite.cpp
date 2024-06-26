// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mms_suite.hpp"
#include<gtest/gtest.h>
#include "dg_linear.hpp"
#include "nonlinear_integ.hpp"
#include "hyperreduction_integ.hpp"
#include "interfaceinteg.hpp"

using namespace std;
using namespace mfem;

namespace mms
{

namespace poisson
{

double ExactSolution(const Vector &x, double t)
{
   double result = constant;
   for (int d = 0; d < x.Size(); d++)
      result += amp[d] * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
   return result;
}

double ExactRHS(const Vector &x, double t)
{
   double result = 0.0;
   for (int d = 0; d < x.Size(); d++)
      result += amp[d] * (2.0 * pi / L[d]) * (2.0 * pi / L[d]) * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
   return result;
}

PoissonSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   PoissonSolver *test = new PoissonSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(ExactSolution);
   test->SetBdrType(BoundaryType::DIRICHLET);
   test->AddRHSFunction(ExactRHS);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

   return test;
}

void CheckConvergence()
{
   amp[0] = config.GetOption<double>("manufactured_solution/amp1", 0.22);
   amp[1] = config.GetOption<double>("manufactured_solution/amp2", 0.13);
   amp[2] = config.GetOption<double>("manufactured_solution/amp3", 0.37);
   L[0] = config.GetOption<double>("manufactured_solution/L1", 0.31);
   L[1] = config.GetOption<double>("manufactured_solution/L2", 0.72);
   L[2] = config.GetOption<double>("manufactured_solution/L2", 0.47);
   offset[0] = config.GetOption<double>("manufactured_solution/offset1", 0.35);
   offset[1] = config.GetOption<double>("manufactured_solution/offset2", 0.73);
   offset[2] = config.GetOption<double>("manufactured_solution/offset3", 0.59);
   constant = config.GetOption<double>("manufactured_solution/constant", -0.27);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   // Compare with exact solution
   FunctionCoefficient exact_sol(ExactSolution);

   printf("Num. Elem.\tRelative Error\tConvergence Rate\tNorm\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      PoissonSolver *test = SolveWithRefinement(r);

      int order = test->GetDiscretizationOrder();
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double norm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         norm += pow(ComputeLpNorm(2.0, exact_sol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      norm = sqrt(norm);

      double error = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetGridFunction(k);
         error += pow(uk->ComputeLpError(2, exact_sol), 2);
      }
      error = sqrt(error);
      error /= norm;
      
      if (r > base_refine)
      {
         conv_rate(r) = error1 / error;
      }
      printf("%d\t%.15E\t%.15E\t%.15E\n", numEl, error, conv_rate(r), norm);

      // reported convergence rate
      if (r > base_refine)
         EXPECT_TRUE(conv_rate(r) > pow(2.0, order+1) - 0.5);

      error1 = error;
   }

   return;
}

}  // namespace poisson

namespace stokes
{

void uFun_ex(const Vector & x, double t, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x, double t)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * nu * sin(xi)*sin(yi);
}

void fFun(const Vector & x, double t, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = 4.0 * nu * cos(xi) * sin(yi);
   f(1) = 0.0;
}

double gFun(const Vector & x, double t)
{
   assert(x.Size() == 2);

   return 0.0;
}

StokesSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   StokesSolver *test = new StokesSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(uFun_ex);
   test->SetBdrType(BoundaryType::DIRICHLET);
   test->AddRHSFunction(fFun);
   // NOTE(kevin): uFun_ex already satisfies zero divergence.
   //              no need to set complementary flux.
   // Array<bool> nz_dbcs(test->GetNumBdr());
   // nz_dbcs = true;
   // test->SetComplementaryFlux(nz_dbcs);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

   return test;
}

void CheckConvergence(const double &threshold)
{
   nu = config.GetOption<double>("stokes/nu", 1.0);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   //printf("Num. Elem.\tRel. v err.\tConv Rate\tNorm\tRel. p err.\tConv Rate\tNorm\n");
   printf("%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
          "Num. Elem.", "Rel v err", "Conv Rate", "Norm", "Rel p err", "Conv Rate", "Norm");

   Vector uconv_rate(num_refine), pconv_rate(num_refine);
   uconv_rate = 0.0;
   pconv_rate = 0.0;
   double uerror1 = 0.0, perror1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      StokesSolver *test = SolveWithRefinement(r);

      // Compare with exact solution
      int dim = test->GetDim();
      VectorFunctionCoefficient exact_usol(dim, uFun_ex);
      FunctionCoefficient exact_psol(pFun_ex);

      // For all velocity dirichlet bc, pressure does not have the absolute value.
      // specify the constant scalar for the reference value.
      double p_const = 0.0;
      int ps = 0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         GridFunction p_ex(*pk);
         p_ex.ProjectCoefficient(exact_psol);
         ps += p_ex.Size();
         p_const += p_ex.Sum();
         // If p_ex is the view vector of pk, then this will prevent false negative test result.
         p_ex += 1.0;
      }
      p_const /= static_cast<double>(ps);

      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         (*pk) += p_const;
      }

      int uorder = test->GetVelFEOrder();
      int porder = test->GetPresFEOrder();
      int order_quad = max(2, 2*uorder+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double unorm = 0.0, pnorm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         unorm += pow(ComputeLpNorm(2.0, exact_usol, *mk, irs), 2);
         pnorm += pow(ComputeLpNorm(2.0, exact_psol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      unorm = sqrt(unorm);
      pnorm = sqrt(pnorm);

      double uerror = 0.0, perror = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetVelGridFunction(k);
         GridFunction *pk = test->GetPresGridFunction(k);
         uerror += pow(uk->ComputeLpError(2, exact_usol), 2);
         perror += pow(pk->ComputeLpError(2, exact_psol), 2);
      }
      uerror = sqrt(uerror);
      perror = sqrt(perror);

      uerror /= unorm;
      perror /= pnorm;
      
      if (r > base_refine)
      {
         uconv_rate(r) = uerror1 / uerror;
         pconv_rate(r) = perror1 / perror;
      }
      printf("%10d\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\n", numEl, uerror, uconv_rate(r), unorm, perror, pconv_rate(r), pnorm);

      // reported convergence rate
      if (r > base_refine)
      {
         EXPECT_TRUE(uconv_rate(r) > pow(2.0, uorder+1) - threshold);
         EXPECT_TRUE(pconv_rate(r) > pow(2.0, porder+1) - threshold);
      }

      uerror1 = uerror;
      perror1 = perror;

      delete test;
   }

   return;
}

}  // namespace stokes

namespace steady_ns
{

void uFun_ex(const Vector & x, double t, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x, double t)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * nu * sin(xi)*sin(yi);
}

void fFun(const Vector & x, double t, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = 4.0 * nu * cos(xi) * sin(yi);
   f(1) = 0.0;

   f(0) += - zeta * sin(xi) * cos(xi);
   f(1) += - zeta * sin(yi) * cos(yi);
}

double gFun(const Vector & x, double t)
{
   assert(x.Size() == 2);

   return 0.0;
}

SteadyNSSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   SteadyNSSolver *test = new SteadyNSSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(uFun_ex);
   test->SetBdrType(BoundaryType::DIRICHLET);
   test->AddRHSFunction(fFun);
   // NOTE(kevin): uFun_ex already satisfies zero divergence.
   //              no need to set complementary flux.
   // Array<bool> nz_dbcs(test->GetNumBdr());
   // nz_dbcs = true;
   // test->SetComplementaryFlux(nz_dbcs);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

   return test;
}

void CheckConvergence(const double &threshold)
{
   nu = config.GetOption<double>("stokes/nu", 1.0);
   zeta = config.GetOption<double>("navier-stokes/zeta", 1.0);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   //printf("Num. Elem.\tRel. v err.\tConv Rate\tNorm\tRel. p err.\tConv Rate\tNorm\n");
   printf("%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
          "Num. Elem.", "Rel v err", "Conv Rate", "Norm", "Rel p err", "Conv Rate", "Norm");

   Vector uconv_rate(num_refine), pconv_rate(num_refine);
   uconv_rate = 0.0;
   pconv_rate = 0.0;
   double uerror1 = 0.0, perror1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      SteadyNSSolver *test = SolveWithRefinement(r);

      // Compare with exact solution
      int dim = test->GetDim();
      VectorFunctionCoefficient exact_usol(dim, uFun_ex);
      FunctionCoefficient exact_psol(pFun_ex);

      // For all velocity dirichlet bc, pressure does not have the absolute value.
      // specify the constant scalar for the reference value.
      double p_const = 0.0;
      int ps = 0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         GridFunction p_ex(*pk);
         p_ex.ProjectCoefficient(exact_psol);
         ps += p_ex.Size();
         p_const += p_ex.Sum();
         // If p_ex is the view vector of pk, then this will prevent false negative test result.
         p_ex += 1.0;
      }
      p_const /= static_cast<double>(ps);

      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         (*pk) += p_const;
      }

      int uorder = test->GetVelFEOrder();
      int porder = test->GetPresFEOrder();
      int order_quad = max(2, 2*uorder+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double unorm = 0.0, pnorm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         unorm += pow(ComputeLpNorm(2.0, exact_usol, *mk, irs), 2);
         pnorm += pow(ComputeLpNorm(2.0, exact_psol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      unorm = sqrt(unorm);
      pnorm = sqrt(pnorm);

      double uerror = 0.0, perror = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetVelGridFunction(k);
         GridFunction *pk = test->GetPresGridFunction(k);
         uerror += pow(uk->ComputeLpError(2, exact_usol), 2);
         perror += pow(pk->ComputeLpError(2, exact_psol), 2);
      }
      uerror = sqrt(uerror);
      perror = sqrt(perror);

      uerror /= unorm;
      perror /= pnorm;
      
      if (r > base_refine)
      {
         uconv_rate(r) = uerror1 / uerror;
         pconv_rate(r) = perror1 / perror;
      }
      printf("%10d\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\n", numEl, uerror, uconv_rate(r), unorm, perror, pconv_rate(r), pnorm);

      // reported convergence rate
      if (r > base_refine)
      {
         EXPECT_TRUE(uconv_rate(r) > pow(2.0, uorder+1) - threshold);
         EXPECT_TRUE(pconv_rate(r) > pow(2.0, porder+1) - threshold);
      }

      uerror1 = uerror;
      perror1 = perror;

      delete test;
   }

   return;
}

} // namespace steady_ns

namespace unsteady_ns
{

UnsteadyNSSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   UnsteadyNSSolver *test = new UnsteadyNSSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(mms::steady_ns::uFun_ex);
   test->SetBdrType(BoundaryType::DIRICHLET);
   test->AddRHSFunction(mms::steady_ns::fFun);
   
   BlockVector *U = test->GetSolution();
   for (int k = 0; k < U->Size(); k++)
      (*U)(k) = 1e-2 * (2. * UniformRandom() - 1.);

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();

   test->SaveVisualization();

   return test;
}

void CheckConvergence(const double &threshold)
{
   mms::steady_ns::nu = config.GetOption<double>("stokes/nu", 1.0);
   mms::steady_ns::zeta = config.GetOption<double>("navier-stokes/zeta", 1.0);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   //printf("Num. Elem.\tRel. v err.\tConv Rate\tNorm\tRel. p err.\tConv Rate\tNorm\n");
   printf("%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
          "Num. Elem.", "Rel v err", "Conv Rate", "Norm", "Rel p err", "Conv Rate", "Norm");

   Vector uconv_rate(num_refine), pconv_rate(num_refine);
   uconv_rate = 0.0;
   pconv_rate = 0.0;
   double uerror1 = 0.0, perror1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      UnsteadyNSSolver *test = SolveWithRefinement(r);

      // Compare with exact solution
      int dim = test->GetDim();
      VectorFunctionCoefficient exact_usol(dim, mms::steady_ns::uFun_ex);
      FunctionCoefficient exact_psol(mms::steady_ns::pFun_ex);

      // For all velocity dirichlet bc, pressure does not have the absolute value.
      // specify the constant scalar for the reference value.
      double p_const = 0.0;
      int ps = 0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         GridFunction p_ex(*pk);
         p_ex.ProjectCoefficient(exact_psol);
         ps += p_ex.Size();
         p_const += p_ex.Sum();
         // If p_ex is the view vector of pk, then this will prevent false negative test result.
         p_ex += 1.0;
      }
      p_const /= static_cast<double>(ps);

      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *pk = test->GetPresGridFunction(k);
         (*pk) += p_const;
      }

      int uorder = test->GetVelFEOrder();
      int porder = test->GetPresFEOrder();
      int order_quad = max(2, 2*uorder+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double unorm = 0.0, pnorm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         unorm += pow(ComputeLpNorm(2.0, exact_usol, *mk, irs), 2);
         pnorm += pow(ComputeLpNorm(2.0, exact_psol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      unorm = sqrt(unorm);
      pnorm = sqrt(pnorm);

      double uerror = 0.0, perror = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetVelGridFunction(k);
         GridFunction *pk = test->GetPresGridFunction(k);
         uerror += pow(uk->ComputeLpError(2, exact_usol), 2);
         perror += pow(pk->ComputeLpError(2, exact_psol), 2);
      }
      uerror = sqrt(uerror);
      perror = sqrt(perror);

      uerror /= unorm;
      perror /= pnorm;
      
      if (r > base_refine)
      {
         uconv_rate(r) = uerror1 / uerror;
         pconv_rate(r) = perror1 / perror;
      }
      printf("%10d\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\t%10.5E\n", numEl, uerror, uconv_rate(r), unorm, perror, pconv_rate(r), pnorm);

      // reported convergence rate
      if (r > base_refine)
      {
         EXPECT_TRUE(uconv_rate(r) > pow(2.0, uorder+1) - threshold);
         EXPECT_TRUE(pconv_rate(r) > pow(2.0, porder+1) - threshold);
      }

      uerror1 = uerror;
      perror1 = perror;

      delete test;
   }

   return;
}

}

namespace linelast
{
   void ExactSolution(const Vector &x, double t, Vector &u)
   {
      u = 0.0;
      for (size_t i = 0; i < dim; i++)
      {
         u(i) = pow(x(i), 3.0);
      }
   }

   void ExactRHS(const Vector &x, double t, Vector &u)
   {
      u = 0.0;
      for (size_t i = 0; i < dim; i++)
      {
         u(i) = 6.0 * x(i) * (lambda + 2.0 * mu);
      }
      u *= -1.0;
   }

   LinElastSolver *SolveWithRefinement(const int num_refinement)
   {
      config.dict_["mesh"]["uniform_refinement"] = num_refinement;
      LinElastSolver *test = new LinElastSolver();

      dim = test->GetDim();

      test->InitVariables();
      test->InitVisualization();

      test->AddBCFunction(ExactSolution, 1);
      test->AddBCFunction(ExactSolution, 2);
      test->AddBCFunction(ExactSolution, 3);
      test->SetBdrType(BoundaryType::DIRICHLET);
      test->AddRHSFunction(ExactRHS);

      test->BuildOperators();

      test->SetupBCOperators();

      test->Assemble();

      test->Solve();

      return test;
   }

   void CheckConvergence()
   {
      int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
      int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

      // Compare with exact solution
      int dim = 2; // only check two dimensions
      VectorFunctionCoefficient exact_sol(dim, ExactSolution);

      printf("Num. Elem.\tRelative Error\tConvergence Rate\tNorm\n");

      Vector conv_rate(num_refine);
      conv_rate = 0.0;
      double error1 = 0.0;
      for (int r = base_refine; r < num_refine; r++)
      {
         LinElastSolver *test = SolveWithRefinement(r);

         int order = test->GetDiscretizationOrder();
         int order_quad = max(2, 2 * order + 1);
         const IntegrationRule *irs[Geometry::NumGeom];
         for (int i = 0; i < Geometry::NumGeom; ++i)
         {
            irs[i] = &(IntRules.Get(i, order_quad));
         }

         int numEl = 0;
         double norm = 0.0;
         for (int k = 0; k < test->GetNumSubdomains(); k++)
         {
            Mesh *mk = test->GetMesh(k);
            norm += pow(ComputeLpNorm(2.0, exact_sol, *mk, irs), 2);
            numEl += mk->GetNE();
         }
         norm = sqrt(norm);

         double error = 0.0;
         for (int k = 0; k < test->GetNumSubdomains(); k++)
         {
            GridFunction *uk = test->GetGridFunction(k);
            error += pow(uk->ComputeLpError(2, exact_sol), 2);
         }
         error = sqrt(error);
         error /= norm;

         if (r > base_refine)
         {
            conv_rate(r) = error1 / error;
         }

         printf("%d\t%.15E\t%.15E\t%.15E\n", numEl, error, conv_rate(r), norm);

         // reported convergence rate
         if (r > base_refine)
            EXPECT_TRUE(conv_rate(r) > pow(2.0, order + 1) - 0.5);

         error1 = error;
      }

      return;
   }

} // namespace linelast

namespace advdiff
{

double ExactSolution(const Vector &x, double t)
{
   double result = constant;
   for (int d = 0; d < x.Size(); d++)
      result += amp[d] * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
   return result;
}

void ExactFlow(const Vector &x, double t, Vector &y)
{
   y.SetSize(x.Size());
   y = 0.0;
   y(0) = u0; y(1) = v0;

   double xi = 2.0 * pi * wn * (x(0) - uoffset[0]);
   double yi = 2.0 * pi * wn * (x(1) - uoffset[1]);

   // incompressible flow field
   y(0) += du * cos(xi) * sin(yi);
   y(1) += -du * sin(xi) * cos(yi);

   return;
}

double ExactRHS(const Vector &x, double t)
{
   Vector flow;
   ExactFlow(x, t, flow);

   double result = 0.0;
   for (int d = 0; d < x.Size(); d++)
   {
      result += amp[d] * (2.0 * pi / L[d]) * (2.0 * pi / L[d]) * sin(2.0 * pi / L[d] * (x(d) - offset[d]));
      result += Pe * flow(d) * amp[d] * (2.0 * pi / L[d]) * cos(2.0 * pi / L[d] * (x(d) - offset[d]));
   }
   return result;
}

AdvDiffSolver *SolveWithRefinement(const int num_refinement)
{
   config.dict_["mesh"]["uniform_refinement"] = num_refinement;
   AdvDiffSolver *test = new AdvDiffSolver();

   test->InitVariables();
   test->InitVisualization();

   test->AddBCFunction(ExactSolution);
   test->SetBdrType(BoundaryType::DIRICHLET);
   test->AddRHSFunction(ExactRHS);
   test->SetFlowAtSubdomain(ExactFlow);

   FunctionCoefficient exact_sol(ExactSolution);
   for (int k = 0; k < test->GetNumSubdomains(); k++)
   {
      GridFunction *uk = test->GetGridFunction(k);
      uk->ProjectCoefficient(exact_sol);
   }
   BlockVector *exact_U = test->GetSolutionCopy();
   Vector error;

   test->BuildOperators();

   test->SetupBCOperators();

   test->Assemble();

   test->Solve();
   // test->CompareSolution(*exact_U, error);
   test->SaveVisualization();

   delete exact_U;
   return test;
}

void CheckConvergence()
{
   amp[0] = config.GetOption<double>("manufactured_solution/amp1", 0.22);
   amp[1] = config.GetOption<double>("manufactured_solution/amp2", 0.13);
   amp[2] = config.GetOption<double>("manufactured_solution/amp3", 0.37);
   L[0] = config.GetOption<double>("manufactured_solution/L1", 0.31);
   L[1] = config.GetOption<double>("manufactured_solution/L2", 0.72);
   L[2] = config.GetOption<double>("manufactured_solution/L2", 0.47);
   offset[0] = config.GetOption<double>("manufactured_solution/offset1", 0.35);
   offset[1] = config.GetOption<double>("manufactured_solution/offset2", 0.73);
   offset[2] = config.GetOption<double>("manufactured_solution/offset3", 0.59);
   constant = config.GetOption<double>("manufactured_solution/constant", -0.27);

   Pe = config.GetOption<double>("adv-diff/peclet_number", 0.2);
   u0 = config.GetOption<double>("manufactured_solution/u0", 1.2);
   v0 = config.GetOption<double>("manufactured_solution/v0", -0.7);
   du = config.GetOption<double>("manufactured_solution/du", 0.21);
   wn = config.GetOption<double>("manufactured_solution/wn", 0.8);
   uoffset[0] = config.GetOption<double>("manufactured_solution/uoffset1", 0.51);
   uoffset[1] = config.GetOption<double>("manufactured_solution/uoffset2", 0.19);
   uoffset[2] = config.GetOption<double>("manufactured_solution/uoffset3", 0.91);

   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);
   int base_refine = config.GetOption<int>("manufactured_solution/baseline_refinement", 0);

   // Compare with exact solution
   FunctionCoefficient exact_sol(ExactSolution);

   printf("Num. Elem.\tRelative Error\tConvergence Rate\tNorm\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = base_refine; r < num_refine; r++)
   {
      AdvDiffSolver *test = SolveWithRefinement(r);

      int order = test->GetDiscretizationOrder();
      int order_quad = max(2, 2*order+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      int numEl = 0;
      double norm = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         Mesh *mk = test->GetMesh(k);
         norm += pow(ComputeLpNorm(2.0, exact_sol, *mk, irs), 2);
         numEl += mk->GetNE();
      }
      norm = sqrt(norm);

      double error = 0.0;
      for (int k = 0; k < test->GetNumSubdomains(); k++)
      {
         GridFunction *uk = test->GetGridFunction(k);
         error += pow(uk->ComputeLpError(2, exact_sol), 2);
      }
      error = sqrt(error);
      error /= norm;
      
      if (r > base_refine)
      {
         conv_rate(r) = error1 / error;
      }
      printf("%d\t%.15E\t%.15E\t%.15E\n", numEl, error, conv_rate(r), norm);

      // reported convergence rate
      if (r > base_refine)
         EXPECT_TRUE(conv_rate(r) > pow(2.0, order+1) - 0.5);

      error1 = error;

      delete test;
   }

   return;
}

}  // namespace advdiff

namespace fem
{

namespace dg_bdr_normal_lf
{

void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * sin(xi)*sin(yi);
}

double EvalWithRefinement(const int num_refinement, int &order_out)
{  
   // 1. Parse command-line options.
   std::string mesh_file = config.GetRequiredOption<std::string>("mesh/filename");
   bool use_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);
   int order = config.GetOption<int>("discretization/order", 1);
   order_out = order;

   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < num_refinement; l++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes;
   if (use_dg)
   {
      fes = new FiniteElementSpace(mesh, dg_coll);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, h1_coll);
   }

   Array<int> p_ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   p_ess_attr = 0;
   p_ess_attr[1] = 1;

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction p(fes);

   p.ProjectCoefficient(pcoeff);

   LinearForm *gform = new LinearForm(fes);
   // gform->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), p_ess_attr);
   gform->AddBoundaryIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), p_ess_attr);
   gform->Assemble();

   double product = p * (*gform);

   // 17. Free the used memory.
   delete gform;
   delete fes;
   delete dg_coll;
   delete h1_coll;
   delete mesh;

   return product;
}

void CheckConvergence()
{
   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);

   double Lx = 1.0, Ly = 1.0;
   double product_ex = sin(Lx) * cos(Lx) * (Ly - 0.5 * sin(2.0 * Ly));
   printf("(p, n dot u_d)_ex = %.5E\n", product_ex);

   printf("Num. Refine.\tRel. Error\tConv Rate\tProduct\tProduct_ex\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = 0; r < num_refine; r++)
   {
      int order = -1;
      double product = EvalWithRefinement(r, order);

      double error = abs(product - product_ex) / abs(product_ex);
      
      if (r > 0)
         conv_rate(r) = error1 / error;
      printf("%d\t%.5E\t%.5E\t%.5E\t%.5E\n", r, error, conv_rate(r), product, product_ex);

      // reported convergence rate
      if (r > 0)
         EXPECT_TRUE(conv_rate(r) > pow(2.0, order+1) - 0.1);

      error1 = error;
   }

   return;
}

}  // namespace dg_bdr_normal_lf

namespace dg_temam
{

void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi) * sin(yi);
   u(1) = - sin(xi) * cos(yi);
}

void usFun_ex(const Vector & x, Vector & us)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   us(0) = cos(xi) + sin(yi);
   us(1) = - sin(xi) + cos(yi);
}

void EvalWithRefinement(const int num_refinement, int &order_out, double &product1, double &product2)
{  
   // 1. Parse command-line options.
   std::string mesh_file = config.GetRequiredOption<std::string>("mesh/filename");
   bool use_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);
   int order = config.GetOption<int>("discretization/order", 1);
   order_out = order;

   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < num_refinement; l++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes;
   if (use_dg)
   {
      fes = new FiniteElementSpace(mesh, dg_coll, dim);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, h1_coll, dim);
   }

   Array<int> ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   ess_attr = 0;
   ess_attr[1] = 1;

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   VectorFunctionCoefficient uscoeff(dim, usFun_ex);
   ConstantCoefficient one(1.0), minus_one(-1.0), half(0.5), minus_half(-0.5);
   Vector test(dim);
   test(0) = 1.0; test(1) = 1.0;
   VectorConstantCoefficient test_coeff(test);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u(fes), us(fes);
   Vector Nu(u);

   u.ProjectCoefficient(ucoeff);
   us.ProjectCoefficient(uscoeff);

   /* this integration rule can be used only for domain integrator. */
   IntegrationRule gll_ir_nl = IntRules.Get(fes->GetFE(0)->GetGeomType(),
                                             (int)(ceil(1.5 * (2 * fes->GetMaxElementOrder() - 1))));

   auto *domain_integ1 = new VectorConvectionTrilinearFormIntegrator(one);
   domain_integ1->SetIntRule(&gll_ir_nl);

   NonlinearForm nform1(fes);
   nform1.AddDomainIntegrator(domain_integ1);

   nform1.Mult(u, Nu);
   product1 = (us * Nu);

   // auto *temam_integ2 = new IncompressibleInviscidFluxNLFIntegrator(minus_one);
   // auto *temam_bdr_integ1 = new DGBdrTemamLFIntegrator(ucoeff);
   // temam_integ2->SetIntRule(&gll_ir_nl);

   auto *temam_integ1 = new VectorConvectionTrilinearFormIntegrator(half);
   auto *temam_integ2 = new IncompressibleInviscidFluxNLFIntegrator(minus_half);
   auto *temam_integ3 = new DGTemamFluxIntegrator(minus_half);
   auto *temam_bdr_integ1 = new DGBdrTemamLFIntegrator(ucoeff, &minus_half);
   temam_integ1->SetIntRule(&gll_ir_nl);
   temam_integ2->SetIntRule(&gll_ir_nl);

   NonlinearForm nform2(fes);
   nform2.AddDomainIntegrator(temam_integ1);
   nform2.AddDomainIntegrator(temam_integ2);
   if (use_dg)
      nform2.AddInteriorFaceIntegrator(temam_integ3);

   LinearForm gform(fes);
   gform.AddBdrFaceIntegrator(temam_bdr_integ1);
   // gform.AddBoundaryIntegrator(temam_bdr_integ1);
   gform.Assemble();

   nform2.Mult(u, Nu);
   product2 = (us * Nu) - (us * gform);

   // 17. Free the used memory.
   delete fes;
   delete dg_coll;
   delete h1_coll;
   delete mesh;

   return;
}

void CheckConvergence()
{
   int num_refine = config.GetOption<int>("manufactured_solution/number_of_refinement", 3);

   double Lx = 1.0, Ly = 1.0;
   /* < us, u dot grad u >_D */
   double product_ex = -0.561514263166004384178127162067055248532791;
   /* - < grad us, u kron u >_D */
   double product_ex2 = 0.153111262927311;
   printf("(us, u_d dot grad u_d)_ex = %.15E\n", product_ex);

   printf("Num. Refine.\tRel. Error\tConv Rate\tProduct\tProduct_ex\n");

   Vector conv_rate(num_refine);
   conv_rate = 0.0;
   double error1 = 0.0;
   for (int r = 0; r < num_refine; r++)
   {
      int order = -1;
      double product1, product2;
      EvalWithRefinement(r, order, product1, product2);

      double error = abs(product2 - product_ex) / abs(product_ex);
      
      if (r > 0)
         conv_rate(r) = error1 / error;
      printf("%d\t%.5E\t%.5E\t%.5E\t%.5E\n", r, error, conv_rate(r), product2, product_ex);

      // reported convergence rate
      if (r > 0)
         EXPECT_TRUE(conv_rate(r) > pow(2.0, order+1) - 0.1);

      error1 = error;
   }

   return;
}

}  // namespace dg_temam

}  // namespace fem

}  // namespace mms
