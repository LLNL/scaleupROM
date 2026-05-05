// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT
//
// Reference Poisson solver for HCDDPINN/.../2d_poisson/buffer2.py.
//
//   -div(kappa * grad u) = f   on  Omega = [0, 2] x [0, 1]
//
// kappa is piecewise constant: kappa1 = 0.1 (element attribute 1, left of
// the tilted interface) and kappa2 = 1.0 (element attribute 2, right).
// f is a sum of three Gaussians. Boundary attribute 1 is homogeneous Neumann
// (natural BC, no integrator needed). Boundary attribute 2 is homogeneous
// Dirichlet (essential BC, satisfied by the zero initial guess).

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace mfem;

double forcing(const Vector &x);

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../playground/meshes/beam-quad.mesh";
   int order = 1;
   int ref_levels = 0;
   const char *paraview_dir = "paraview_phyml";
   const char *txt_file = "phyml_sol.txt";
   bool visualization = true;
   bool dump_txt = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree.");
   args.AddOption(&ref_levels, "-r", "--refine", "Number of uniform mesh refinements.");
   args.AddOption(&paraview_dir, "-vd", "--vis-dir", "ParaView output directory.");
   args.AddOption(&txt_file, "-t", "--txt", "Output text file with (x, y, u) rows.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable ParaView output.");
   args.AddOption(&dump_txt, "-txt", "--dump-txt",
                  "-no-txt", "--no-dump-txt",
                  "Enable or disable (x, y, u) text dump.");
   args.ParseCheck();

   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   MFEM_VERIFY(mesh.attributes.Max() >= 2,
               "Expected mesh with at least 2 element attributes (got "
               << mesh.attributes.Max() << ").");
   MFEM_VERIFY(mesh.bdr_attributes.Max() >= 2,
               "Expected mesh with at least 2 boundary attributes (got "
               << mesh.bdr_attributes.Max() << ").");

   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

   // Boundary attribute 2 is Dirichlet (homogeneous). Attribute 1 is Neumann.
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[1] = 1; // attribute 2 -> index 1
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Zero initial guess satisfies homogeneous Dirichlet BC.
   GridFunction x(&fespace);
   x = 0.0;

   // kappa(attr=1) = 0.1, kappa(attr=2) = 1.0.
   Vector kappa_vec(mesh.attributes.Max());
   kappa_vec = 1.0;
   kappa_vec(0) = 0.1;
   kappa_vec(1) = 1.0;
   PWConstCoefficient kappa(kappa_vec);

   FunctionCoefficient f_coeff(forcing);

   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   b.Assemble();

   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(kappa));
   a.Assemble();

   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   cout << "System size: " << A.Height() << endl;

   GSSmoother M(A);
   PCG(A, M, B, X, 1, 2000, 1e-12, 0.0);

   a.RecoverFEMSolution(X, b, x);

   x.Save("phyml_sol");
   mesh.Save("phyml_mesh");

   if (visualization)
   {
      ParaViewDataCollection pv(paraview_dir, &mesh);
      pv.SetLevelsOfDetail(order);
      pv.SetHighOrderOutput(true);
      pv.SetPrecision(8);
      pv.RegisterField("solution", &x);
      pv.Save();
   }

   if (dump_txt)
   {
      ofstream out(txt_file);
      out << scientific << setprecision(15);
      Array<int> dofs;
      Vector phys(dim);
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         const FiniteElement *fe = fespace.GetFE(e);
         ElementTransformation *Tr = fespace.GetElementTransformation(e);
         fespace.GetElementDofs(e, dofs);
         const IntegrationRule &nodes = fe->GetNodes();
         for (int j = 0; j < nodes.GetNPoints(); j++)
         {
            Tr->Transform(nodes.IntPoint(j), phys);
            int dof = dofs[j];
            double sign = 1.0;
            if (dof < 0) { dof = -1 - dof; sign = -1.0; }
            const double val = sign * x(dof);
            out << phys(0) << " " << phys(1) << " " << val << "\n";
         }
      }
      cout << "Wrote " << txt_file << endl;
   }

   return 0;
}

double forcing(const Vector &x)
{
   const double ctrs[3][2] = {{0.3, 0.6}, {1.0, 0.2}, {1.6, 0.7}};
   const double rs[3]      = {0.08, 0.2, 0.1};
   const double amps[3]    = {1e1, 2e1, 1.5e1};

   double f = 0.0;
   for (int i = 0; i < 3; i++)
   {
      const double dx = x(0) - ctrs[i][0];
      const double dy = x(1) - ctrs[i][1];
      const double r0 = rs[i];
      f += amps[i] * exp(-(dx * dx + dy * dy) / (r0 * r0));
   }
   return f;
}
