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
   // ess_attr[1] = 0;
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
   bdrCoeffs[0] = new ConstantCoefficient(2.0);
   // bdrCoeffs[1] = new FunctionCoefficient(dbc1);
   bdrCoeffs[1] = NULL;;
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

   // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
   GSSmoother M(A);
   PCG(A, M, B, X, 1, 200, 1e-12, 0.0);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   x.Save("sol");
   mesh.Save("mesh");

   {
     const int totalDofs = fespace.GetNDofs();
     double *dataX = x.GetData();
     for (int i = 0; i < totalDofs; i++) {
       printf("x[%d] = %2.3f\n", i, dataX[i]);
     }
   }

   // Save visualization in paraview output.
   ParaViewDataCollection *paraviewColl = NULL;
   paraviewColl = new ParaViewDataCollection("paraview_output", &mesh);
   paraviewColl->SetLevelsOfDetail(order);
   paraviewColl->SetHighOrderOutput(true);
   paraviewColl->SetPrecision(8);

   paraviewColl->RegisterField("solution", &x);
   paraviewColl->SetOwnData(true);
   paraviewColl->Save();

   // // Domain decomposition from DD-LSPG.
   // printf("Domain-decomposition with two subdomains.");
   // {
   //   // TODO: figure out how to extract these index information out of subdomain attributes.
   // }

   //Now solve with FETI method for two subdomains.
   printf("Solving with FETI method.\n");
   {
     std::string submesh_file(mesh_file);
     submesh_file = submesh_file.substr(0, submesh_file.find_last_of('.')) + ".submesh";
     Mesh mesh(submesh_file.c_str());

     H1_FECollection fec(order, mesh.Dimension());
     FiniteElementSpace fespace(&mesh, &fec);
     int total_num_dofs = fespace.GetTrueVSize();
     cout << "Number of unknowns: " << total_num_dofs << endl;

     // Specify where Dirichlet BC is applied as essential boundary condition.
     Array<int> ess_attr1(mesh.bdr_attributes.Max()), ess_attr2(mesh.bdr_attributes.Max());
     ess_attr1 = 1;
     ess_attr2 = 1;
     // Interface
     ess_attr1[1] = 0;
     ess_attr2[0] = 0;
     Array<int> ess_tdof_list1, ess_tdof_list2;
     fespace.GetEssentialTrueDofs(ess_attr1, ess_tdof_list1);
     fespace.GetEssentialTrueDofs(ess_attr2, ess_tdof_list2);

     GridFunction x1(&fespace), x2(&fespace);
     x1 = 0.0;
     x2 = 0.0;

     // add dirichlet boundary condition.
     Coefficient *bdrCoeffs[mesh.bdr_attributes.Max()];
     Array<int> bdrAttr(mesh.bdr_attributes.Max());
     bdrCoeffs[0] = new ConstantCoefficient(2.0);
     // bdrCoeffs[1] = new FunctionCoefficient(dbc1);
     bdrCoeffs[1] = NULL;;
     for (int b = 0; b < mesh.bdr_attributes.Max(); b++) {
       // Determine which boundary attribute will use the b-th boundary coefficient.
       // Since all boundary attributes use different BCs, only one index is 'turned on'.
       bdrAttr = 0;
       bdrAttr[b] = 1;
       // Project the b-th boundary coefficient.
       x1.ProjectBdrCoefficient(*bdrCoeffs[b], bdrAttr);
     }

     ConstantCoefficient one(1.0);
     LinearForm b1(&fespace), b2(&fespace);
     b1.AddDomainIntegrator(new DomainLFIntegrator(one));
     b1.Assemble();
     b2.AddDomainIntegrator(new DomainLFIntegrator(one));
     b2.Assemble();

     Array<int> block_offsets(4); // number of variables + 1
     block_offsets[0] = 0;
     block_offsets[1] = fespace.GetVSize();
     block_offsets[2] = fespace.GetVSize();
     block_offsets[3] = 1;
     block_offsets.PartialSum();

     // std::cout << "***********************************************************\n";
     // std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
     // std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
     // std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
     // std::cout << "***********************************************************\n";
     BlockOperator kktOp(block_offsets);

     BilinearForm a1(&fespace), a2(&fespace);
     a1.AddDomainIntegrator(new DiffusionIntegrator);
     a1.Assemble();
     a2.AddDomainIntegrator(new DiffusionIntegrator);
     a2.Assemble();

     SparseMatrix A1, A2;
     Vector R1, R2, X1, X2;
     // a1.FormSystemMatrix(ess_tdof_list1, A1);
     a1.FormLinearSystem(ess_tdof_list1, x1, b1, A1, X, R1);
     // for (int i = 0; i < R1.Size(); i++) {
     //   printf("R1[%d] = %2.3f\n", i, R1[i]);
     // }
     // a2.FormSystemMatrix(ess_tdof_list2, A2);
     a2.FormLinearSystem(ess_tdof_list2, x2, b2, A2, X, R2);
     // for (int i = 0; i < R2.Size(); i++) {
     //   printf("R2[%d] = %2.3f\n", i, R2[i]);
     // }
     kktOp.SetBlock(0,0, &A1);
     kktOp.SetBlock(1,1, &A2);

     SparseMatrix B1(fespace.GetVSize(), 1), B2(fespace.GetVSize(), 1);
     Array<int> boundary_dofs;
     fespace.GetBoundaryTrueDofs(boundary_dofs);
     B1.Set(boundary_dofs[1], 0, 1.0);
     B2.Set(boundary_dofs[0], 0, -1.0);
     B1.Finalize();
     B2.Finalize();

     TransposeOperator B1t(B1), B2t(B2);

     kktOp.SetBlock(0,2, &B1);
     kktOp.SetBlock(1,2, &B2);
     kktOp.SetBlock(2,0, &B1t);
     kktOp.SetBlock(2,1, &B2t);

     BlockVector XB(block_offsets), rhs(block_offsets);
     Vector R(R1.Size() + R2.Size() + 1);
     R = 0.0;
     for (int i = 0; i < block_offsets[1]; i++) R(i) = R1(i);
     for (int i = block_offsets[1]; i < block_offsets[2]; i++) R(i) = R2(i - block_offsets[1]);
     rhs.Update(R, block_offsets);
     // for (int i = 0; i < block_offsets[3]; i++) {
     //   printf("rhs[%d] = %2.3f\n", i, rhs[i]);
     // }

     int maxIter(1000);
     double rtol(1.e-6);
     double atol(1.e-10);

     BlockDiagonalPreconditioner kktPrec(block_offsets);
     MINRESSolver solver;
     solver.SetAbsTol(atol);
     solver.SetRelTol(rtol);
     solver.SetMaxIter(maxIter);
     solver.SetOperator(kktOp);
     solver.SetPreconditioner(kktPrec);
     solver.SetPrintLevel(1);
     XB = 0.0;
     solver.Mult(rhs, XB);

     for (int i = 0; i < block_offsets[3]; i++) {
       printf("xb[%d] = %2.3f\n", i, XB[i]);
     }

   }


   return 0;
}
