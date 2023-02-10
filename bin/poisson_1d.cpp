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
   int globalNE = mesh.GetNE();

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
     int subNE = mesh.GetNE();
     if (globalNE % subNE != 0) {
       printf("Number of elements in global domain cannot be divided by that of subdomain.");
       exit(1);
     }
     int numSub = globalNE / subNE;

     H1_FECollection fec(order, mesh.Dimension());
     FiniteElementSpace fespace(&mesh, &fec);
     int total_num_dofs = fespace.GetTrueVSize();
     cout << "Number of unknowns: " << total_num_dofs << endl;

     // Specify where Dirichlet BC is applied as essential boundary condition.
     std::vector<Array<int>> ess_attrs(numSub), ess_tdof_lists(numSub);
     for (int i = 0; i < numSub; i++) {
       ess_attrs[i].SetSize(mesh.bdr_attributes.Max());
       ess_attrs[i] = 0;
     }
     ess_attrs[0][0] = 1;
     ess_attrs[numSub - 1][1] = 1;

     for (int i = 0; i < numSub; i++)
       fespace.GetEssentialTrueDofs(ess_attrs[i], ess_tdof_lists[i]);

     std::vector<GridFunction *> xs(numSub);
     for (int i = 0; i < numSub; i++) {
       xs[i] = new GridFunction(&fespace);
       *(xs[i]) = 0.0;
     }

     // add dirichlet boundary condition.
     Coefficient *bdrCoeffs[mesh.bdr_attributes.Max()];
     Array<int> bdrAttr(mesh.bdr_attributes.Max());
     bdrCoeffs[0] = new ConstantCoefficient(2.0);
     bdrCoeffs[1] = NULL;;
     for (int b = 0; b < mesh.bdr_attributes.Max(); b++) {
       // Determine which boundary attribute will use the b-th boundary coefficient.
       // Since all boundary attributes use different BCs, only one index is 'turned on'.
       bdrAttr = 0;
       bdrAttr[b] = 1;
       // Project the b-th boundary coefficient.
       xs[0]->ProjectBdrCoefficient(*bdrCoeffs[b], bdrAttr);
     }

     ConstantCoefficient one(1.0);
     std::vector<LinearForm *> bs(numSub);
     for (int i = 0; i < numSub; i++) {
       bs[i] = new LinearForm(&fespace);
       bs[i]->AddDomainIntegrator(new DomainLFIntegrator(one));
       bs[i]->Assemble();
     }

     Array<int> block_offsets(numSub + 1 + 1); // number of subdomain + lagrangian (1) + 1
     block_offsets[0] = 0;
     for (int i = 0; i < numSub; i++) {
       block_offsets[i + 1] = fespace.GetVSize();
     }
     block_offsets[numSub + 1] = numSub - 1; // number of interfaces
     block_offsets.PartialSum();

     std::cout << "***********************************************************\n";
     // // std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
     // // std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
     std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
     std::cout << "***********************************************************\n";
     BlockOperator kktOp(block_offsets);

     std::vector<BilinearForm *> as(numSub);
     for (int i = 0; i < numSub; i++) {
       as[i] = new BilinearForm(&fespace);
       as[i]->AddDomainIntegrator(new DiffusionIntegrator);
       as[i]->Assemble();
     }

     std::vector<SparseMatrix> As(numSub);
     std::vector<Vector> Bs(numSub), Xs(numSub);
     for (int i = 0; i < numSub; i++) {
       as[i]->FormLinearSystem(ess_tdof_lists[i], *(xs[i]), *(bs[i]), As[i], Xs[i], Bs[i]);
       kktOp.SetBlock(i, i, &As[i]);

       DenseMatrix Ad;
       As[i].ToDenseMatrix(Ad);
       printf("As[%d] (%d x %d)\n", i, Ad.Width(), Ad.Height());
       for (int h = 0; h < Ad.Height(); h++) {
         for (int w = 0; w < Ad.Width(); w++) {
           printf("%2.3f\t", Ad(h, w));
         }
         printf(" | %2.3f \n", Bs[i](h));
       }
     }

     std::vector<SparseMatrix *> BBs(numSub);
     for (int i = 0; i < numSub; i++) {
       BBs[i] = new SparseMatrix(fespace.GetVSize(), numSub - 1);
     }
     Array<int> boundary_dofs;
     fespace.GetBoundaryTrueDofs(boundary_dofs);
     for (int i = 0; i < numSub - 1; i++) {
       BBs[i]->Set(boundary_dofs[1], 0, 1.0);
       BBs[i+1]->Set(boundary_dofs[0], 0, -1.0);
     }

     for (int i = 0; i < numSub; i++) {
       BBs[i]->Finalize();
     }

     std::vector<TransposeOperator *> BBst(numSub);
     for (int i = 0; i < numSub; i++)
       BBst[i] = new TransposeOperator(BBs[i]);

     for (int i = 0; i < numSub; i++) {
       kktOp.SetBlock(i, numSub, BBs[i]);
       kktOp.SetBlock(numSub, i, BBst[i]);
     }

     BlockVector XB(block_offsets), RHS(block_offsets);
     Vector R(block_offsets.Last());
     R = 0.0;
     for (int i = 0; i < numSub; i++) {
       for (int n = block_offsets[i]; n < block_offsets[i + 1]; n++) {
         R(n) = Bs[i](n - block_offsets[i]);
       }
     }
     RHS.Update(R, block_offsets);
     for (int i = 0; i < block_offsets.Last(); i++) {
       printf("RHS[%d] = %2.3f\n", i, RHS[i]);
     }

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
     solver.Mult(RHS, XB);

     for (int i = 0; i < block_offsets.Last(); i++) {
       printf("xb[%d] = %2.3f\n", i, XB[i]);
     }

   }


   return 0;
}
