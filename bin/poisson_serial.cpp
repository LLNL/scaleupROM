//                       MFEM Example 0 - Parallel Version
//
// Compile with: make ex0p
//
// Sample runs:  mpirun -np 4 ex0p
//               mpirun -np 4 ex0p -m ../data/fichera.mesh
//               mpirun -np 4 ex0p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with zero Dirichlet boundary
//              conditions. General 2D/3D serial mesh files and finite element
//              polynomial degrees can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dbc1(const Vector &);
double dbc3(const Vector &);

int main(int argc, char *argv[])
{
   // // 1. Initialize MPI and HYPRE.
   // Mpi::Init(argc, argv);
   // Hypre::Init();

   // 2. Parse command line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh mesh(mesh_file);

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   FiniteElementSpace fespace(&mesh, &fec);
   int total_num_dofs = fespace.GetTrueVSize();
   cout << "Number of unknowns: " << total_num_dofs << endl;

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);
   printf("Boundary dofs\n");
   for (int i = 0; i < boundary_dofs.Size(); i++) {
     printf("%d\t", boundary_dofs[i]);
   }
   printf("\n");

   // Specify where Dirichlet BC is applied as essential boundary condition.
   Array<int> ess_attr(mesh.bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   ess_attr = 1;
   ess_attr[2] = 0;
   ess_attr[3] = 0;
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(ess_attr, ess_tdof_list);
   printf("Essential dofs\n");
   for (int i = 0; i < ess_tdof_list.Size(); i++) {
     printf("%d\t", ess_tdof_list[i]);
   }
   printf("\n");

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // add dirichlet boundary condition.
   Coefficient *bdrCoeffs[mesh.bdr_attributes.Max()];
   Array<int> bdrAttr(mesh.bdr_attributes.Max());
   bdrCoeffs[0] = new ConstantCoefficient(0.0);
   // bdrCoeffs[1] = new FunctionCoefficient(dbc1);
   bdrCoeffs[1] = new ConstantCoefficient(2.0);
   bdrCoeffs[2] = NULL;
   // bdrCoeffs[3] = new FunctionCoefficient(dbc3);
   bdrCoeffs[3] = NULL;
   for (int b = 0; b < mesh.bdr_attributes.Max(); b++) {
     // Determine which boundary attribute will use the b-th boundary coefficient.
     // Since all boundary attributes use different BCs, only one index is 'turned on'.
     bdrAttr = 0;
     bdrAttr[b] = 1;
     // Project the b-th boundary coefficient.
     x.ProjectBdrCoefficient(*bdrCoeffs[b], bdrAttr);
   }

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   SparseMatrix A;
   DenseMatrix Ad;
   Vector B, X;
   // a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   {
     DenseMatrix mat, mat_e;
     printf("a.mat\n");
     a.SpMat().ToDenseMatrix(mat);
     for (int i = 0; i < a.Size(); i++) {
       for (int j = 0; j < a.Size(); j++) {
         printf("%2.3f\t", mat(i, j));
       }
       printf("\n");
     }

     printf("a.mat_e\n");
     a.SpMatElim().ToDenseMatrix(mat_e);
     for (int i = 0; i < a.Size(); i++) {
       for (int j = 0; j < a.Size(); j++) {
         printf("%2.3f\t", mat_e(i, j));
       }
       printf("\n");
     }
   }
   A.ToDenseMatrix(Ad);
   printf("A (%d x %d)\n", Ad.Width(), Ad.Height());

   for (int h = 0; h < Ad.Height(); h++) {
     for (int w = 0; w < Ad.Width(); w++) {
       printf("%2.3f\t", Ad(h, w));
     }
     printf(" | %2.3f \n", B(h));
   }

   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol");
   mesh.Save("mesh");

   // Save visualization in paraview output.
   ParaViewDataCollection *paraviewColl = NULL;
   paraviewColl = new ParaViewDataCollection("paraview_output", &mesh);
   paraviewColl->SetLevelsOfDetail(order);
   paraviewColl->SetHighOrderOutput(true);
   paraviewColl->SetPrecision(8);

   paraviewColl->RegisterField("solution", &x);
   paraviewColl->SetOwnData(true);
   paraviewColl->Save();

   return 0;
}

double dbc1(const Vector &x)
{
  return 0.1 - 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

double dbc3(const Vector &x)
{
  return -0.1 + 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}
