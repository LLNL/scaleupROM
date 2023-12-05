// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "linalg_utils.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
void mlap_uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void grad_pFun_ex(const Vector & x, Vector & y);
double mlap_pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);
void fvec_natural(const Vector & x, Vector & y);
void dudx_ex(const Vector & x, Vector & y);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int refine = 0;
   bool use_dg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.AddOption(&use_dg, "-dg", "--use-dg", "-no-dg", "--no-use-dg",
                  "Use discontinuous Galerkin scheme.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   double sigma = -1.0;
   // DG terms are employed for velocity space, which is order+1. resulting kappa becomes (order+2)^2.
   double kappa = (order + 1) * (order + 1);

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
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
   Array<int> p_ess_tdof, empty;
   fes->GetEssentialTrueDofs(p_ess_attr, p_ess_tdof);

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0), minus_one(-1.0), one(1.0), zero(0.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient dudxcoeff(dim, dudx_ex);
   VectorFunctionCoefficient minus_fcoeff(dim, fFun, &minus_one);
   // VectorFunctionCoefficient grad_pcoeff(dim, grad_pFun_ex);
   VectorFunctionCoefficient mlap_ucoeff(dim, mlap_uFun_ex);
   // FunctionCoefficient mlap_pcoeff(mlap_pFun_ex);
   FunctionCoefficient fnatcoeff(f_natural);
   VectorFunctionCoefficient fvecnatcoeff(dim*dim, fvec_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);
   // GridFunction u_ex(ufes), p_ex(pfes);
   // u_ex.ProjectCoefficient(ucoeff);
   // p_ex.ProjectCoefficient(pcoeff);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction p(fes);

   p.ProjectCoefficient(pcoeff);

   LinearForm *gform = new LinearForm(fes);
   gform->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), p_ess_attr);
   gform->Assemble();

   double product = p * (*gform);
   printf("(p, n dot u_d) = %.5E\n", product);

   double Lx = 1.0, Ly = 1.0;
   double product_ex = sin(Lx) * cos(Lx) * (Ly - 0.5 * sin(2.0 * Ly));
   printf("(p, n dot u_d)_ex = %.5E\n", product_ex);

   double error = abs(product - product_ex) / abs(product_ex);
   printf("Rel. error: %.5E\n", error);

   // 17. Free the used memory.
   delete gform;
   delete fes;
   // delete qfes;
   delete dg_coll;
   delete h1_coll;
   delete mesh;

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

void mlap_uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = 2.0 * cos(xi)*sin(yi);
   u(1) = - 2.0 * sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * sin(xi)*sin(yi);
}

void grad_pFun_ex(const Vector & x, Vector & y)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   y.SetSize(2);

   y(0) = 2.0 * cos(xi)*sin(yi);
   y(1) = 2.0 * sin(xi)*cos(yi);
   return;
}

double mlap_pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * sin(xi)*sin(yi);
   // return 0.0;
}

void fFun(const Vector & x, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = 4.0 * cos(xi) * sin(yi);
   f(1) = 0.0;
}

double gFun(const Vector & x)
{
   assert(x.Size() == 2);

   return 0.0;
}

double f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}

void dudx_ex(const Vector & x, Vector & y)
{
   assert(x.Size() == 2);
   y.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   y(0) = - sin(xi)*sin(yi);
   y(1) = - cos(xi)*cos(yi);
}

void fvec_natural(const Vector & x, Vector & y)
{
   assert(x.Size() == 2);
   y.SetSize(x.Size() * x.Size());

   double xi(x(0));
   double yi(x(1));

   // Grad u = du_i/dx_j - column-major order
   y(0) = - sin(xi)*sin(yi);
   y(1) = - cos(xi)*cos(yi);
   y(2) = cos(xi)*cos(yi);
   y(3) = sin(xi)*sin(yi);

   y(0) -= pFun_ex(x);
   y(3) -= pFun_ex(x);
}