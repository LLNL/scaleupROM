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
#include "interfaceinteg.hpp"
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
   const char *dode_type = "feti";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dode_type, "-d", "--domain-decomposition", "Domain decomposition type: feti, ddlspg, ip.");
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

   std::string dode_type_str(dode_type);
   if (dode_type_str == "ddlspg") {

     // Use the same global mesh.
     int numElem = mesh.GetNE();
     if (numElem != 6) {
       printf("This example is demonstrated only for 6 elements!");
       exit(1);
     }

     int total_num_dofs = fespace.GetTrueVSize();
     Array<bool> isInterfaceDof(total_num_dofs), isRes1Dof(total_num_dofs), isRes2Dof(total_num_dofs);
     Array<bool> isInterior1Dof(total_num_dofs), isInterior2Dof(total_num_dofs);
     isInterfaceDof = false;
     isRes1Dof = false;
     isRes2Dof = false;
     isInterior1Dof = false;
     isInterior2Dof = false;

     Array<bool> isInterfaceElem(numElem), isDomain1Elem(numElem);
     isInterfaceElem = false;
     isInterfaceElem[2] = true;
     isInterfaceElem[3] = true;

     isDomain1Elem = false;
     for (int i = 0; i < 3; i++) isDomain1Elem[i] = true;

     printf("number of elements: %d\n", numElem);
     for (int el = 0; el < numElem; el++) {
       int attr = mesh.GetAttribute(el);
       Vector center(2);
       mesh.GetElementCenter(el, center);
       printf("Element %d: %d - (%f, %f)\n", el, attr, center(0), center(1));

       const FiniteElement *elem = fespace.GetFE(el);
       const int eldDof = elem->GetDof();
       Array<int> vdofs;
       fespace.GetElementVDofs(el, vdofs);
       // printf("%d =?= %d\n", eldDof, vdofs.Size());
       printf("Element %d vdofs: ", el);
       for (int k = 0; k < eldDof; k++) {
         printf("%d ", vdofs[k]);
         if (isInterfaceElem[el]) {
           isInterfaceDof[vdofs[k]] = true;
         }
       }
       printf("\n");
     }

     for (int el = 0; el < numElem; el++) {
       const FiniteElement *elem = fespace.GetFE(el);
       const int eldDof = elem->GetDof();
       Array<int> vdofs;
       fespace.GetElementVDofs(el, vdofs);

       for (int k = 0; k < eldDof; k++) {
         printf("vdof %d: ", vdofs[k]);
         if (isDomain1Elem[el]) {
           printf("residual 1, ");
           isRes1Dof[vdofs[k]] = true;
           if (!isInterfaceDof[vdofs[k]]) {
             printf("interior, ");
             isInterior1Dof[vdofs[k]] = true;
           }
         } else {
           printf("residual 2, ");
           isRes2Dof[vdofs[k]] = true;
           if (!isInterfaceDof[vdofs[k]]) {
             printf("interior, ");
             isInterior2Dof[vdofs[k]] = true;
           }
         }
         printf("\n");
       }
     }

     Array<int> interfaceDofs(0), interior1Dofs(0), interior2Dofs(0), res1Dofs(0), res2Dofs(0);
     for (int i = 0; i < total_num_dofs; i++) {
       if (isInterfaceDof[i]) interfaceDofs.Append(i);
       if (isInterior1Dof[i]) interior1Dofs.Append(i);
       if (isInterior2Dof[i]) interior2Dofs.Append(i);
       if (isRes1Dof[i]) res1Dofs.Append(i);
       if (isRes2Dof[i]) res2Dofs.Append(i);
     }

     Array<int> domain1Dofs = interior1Dofs;
     domain1Dofs.Append(interfaceDofs);
     Array<int> domain2Dofs = interfaceDofs;
     domain2Dofs.Append(interior2Dofs);

     printf("Interior 1 dofs\n");
     for (int i = 0; i < interior1Dofs.Size(); i++) printf("%d ", interior1Dofs[i]);
     printf("\n");

     printf("Interior 2 dofs\n");
     for (int i = 0; i < interior2Dofs.Size(); i++) printf("%d ", interior2Dofs[i]);
     printf("\n");

     printf("Interface dofs\n");
     for (int i = 0; i < interfaceDofs.Size(); i++) printf("%d ", interfaceDofs[i]);
     printf("\n");

     printf("Residual 1 dofs\n");
     for (int i = 0; i < res1Dofs.Size(); i++) printf("%d ", res1Dofs[i]);
     printf("\n");

     printf("Residual 2 dofs\n");
     for (int i = 0; i < res2Dofs.Size(); i++) printf("%d ", res2Dofs[i]);
     printf("\n");

     printf("Domain 1 dofs\n");
     for (int i = 0; i < domain1Dofs.Size(); i++) printf("%d ", domain1Dofs[i]);
     printf("\n");

     printf("Domain 2 dofs\n");
     for (int i = 0; i < domain2Dofs.Size(); i++) printf("%d ", domain2Dofs[i]);
     printf("\n");

     DenseMatrix res1(res1Dofs.Size(), domain1Dofs.Size()), res2(res2Dofs.Size(), domain2Dofs.Size());
     A.GetSubMatrix(res1Dofs, domain1Dofs, res1);
     for (int i = 0; i < res1Dofs.Size(); i++) {
       for (int j = 0; j < domain1Dofs.Size(); j++) {
         printf("%2.3f\t", res1(i, j));
       }
       printf("\n");
     }

     A.GetSubMatrix(res2Dofs, domain2Dofs, res2);
     for (int i = 0; i < res2Dofs.Size(); i++) {
       for (int j = 0; j < domain2Dofs.Size(); j++) {
         printf("%2.3f\t", res2(i, j));
       }
       printf("\n");
     }

     // TODO: these are manual set up of compatibility constraints. Need to automate it.
     SparseMatrix com1(interfaceDofs.Size(), domain1Dofs.Size()), com2(interfaceDofs.Size(), domain2Dofs.Size());
     for (int i = 0; i < interfaceDofs.Size(); i++) {
       com1.Set(i, interior1Dofs.Size() + i, 1.0);
     }
     com1.Finalize();
     for (int i = 0; i < interfaceDofs.Size(); i++) {
       com2.Set(i, i, -1.0);
     }
     com2.Finalize();

     Array<int> row_offsets(4), col_offsets(3);
     row_offsets[0] = 0;
     row_offsets[1] = res1Dofs.Size();
     row_offsets[2] = res2Dofs.Size();
     row_offsets[3] = interfaceDofs.Size();
     row_offsets.PartialSum();

     col_offsets[0] = 0;
     col_offsets[1] = domain1Dofs.Size();
     col_offsets[2] = domain2Dofs.Size();
     col_offsets.PartialSum();

     BlockOperator Ap(row_offsets, col_offsets);
     Ap.SetBlock(0, 0, &res1);
     Ap.SetBlock(1, 1, &res2);
     Ap.SetBlock(2, 0, &com1);
     Ap.SetBlock(2, 1, &com2);

     Vector tmp(row_offsets.Last());
     for (int i = row_offsets[0]; i < row_offsets[1]; i++) tmp(i) = B(res1Dofs[i - row_offsets[0]]);
     for (int i = row_offsets[1]; i < row_offsets[2]; i++) tmp(i) = B(res2Dofs[i - row_offsets[1]]);

     BlockVector XB(col_offsets), RHS(row_offsets);
     RHS.Update(tmp, row_offsets);

     int maxIter(1000);
     double rtol(1.e-6);
     double atol(1.e-10);

     // BlockDiagonalPreconditioner kktPrec(block_offsets);
     GMRESSolver solver;
     solver.SetAbsTol(atol);
     solver.SetRelTol(rtol);
     solver.SetMaxIter(maxIter);
     solver.SetOperator(Ap);
     // solver.SetPreconditioner(kktPrec);
     solver.SetPrintLevel(1);
     XB = 0.0;
     solver.Mult(RHS, XB);

     for (int i = 0; i < col_offsets.Last(); i++) {
       printf("xb[%d] = %2.3f\n", i, XB[i]);
     }

   } else if (dode_type_str == "ip") {

     printf("Solving with Interior Penalty method.\n");

     std::string submesh_file(mesh_file);
     submesh_file = submesh_file.substr(0, submesh_file.find_last_of('.')) + ".submesh";
     Mesh submesh(submesh_file.c_str());
     int subNE = submesh.GetNE();
     if (globalNE % subNE != 0) {
       printf("Number of elements in global domain cannot be divided by that of subdomain.");
       exit(1);
     }
     int numSub = globalNE / subNE;

     std::vector<Mesh *> meshes(numSub);
     for (int m = 0; m < meshes.size(); m++) {
       meshes[m] = new Mesh(submesh);
     }

     H1_FECollection fec(order, submesh.Dimension());
     std::vector<FiniteElementSpace *> fespaces(numSub);
     for (int m = 0; m < numSub; m++) {
       fespaces[m] = new FiniteElementSpace(meshes[m], &fec);
     }
     int total_num_dofs = fespaces[0]->GetTrueVSize();
     cout << "Number of unknowns in each subdomain: " << total_num_dofs << endl;

     // Dirichlet BC on the first and the last meshes.
     Array<Array<int> *> ess_attrs(numSub);
     Array<Array<int> *> ess_tdof_lists(numSub);
     for (int m = 0; m < numSub; m++) {
       ess_attrs[m] = new Array<int>(2);
       (*ess_attrs[m]) = 0;
       if (m == 0) (*ess_attrs[m])[0] = 1;
       if (m == numSub - 1) (*ess_attrs[m])[1] = 1;
       ess_tdof_lists[m] = new Array<int>;

       fespaces[m]->GetEssentialTrueDofs((*ess_attrs[m]), (*ess_tdof_lists[m]));
     }

     // 7. Define the solution x as a finite element grid function in fespace. Set
     //    the initial guess to zero, which also sets the boundary conditions.
     Array<GridFunction *> xs(numSub);
     for (int m = 0; m < numSub; m++) {
       xs[m] = new GridFunction(fespaces[m]);
       (*xs[m]) = 0.0;
     }
     bdrAttr = 0;
     bdrAttr[0] = 1;
     xs[0]->ProjectBdrCoefficient(*bdrCoeffs[0], bdrAttr);
     bdrAttr = 0;
     bdrAttr[1] = 1;
     xs[numSub-1]->ProjectBdrCoefficient(*bdrCoeffs[1], bdrAttr);

     // 8. Set up the linear form b(.) corresponding to the right-hand side.
     ConstantCoefficient one(1.0);
     Array<LinearForm *> bs(numSub);
     Array<BilinearForm *> as(numSub);
     for (int m = 0; m < numSub; m++) {
       bs[m] = new LinearForm(fespaces[m]);
       bs[m]->AddDomainIntegrator(new DomainLFIntegrator(one));
       bs[m]->Assemble();

       as[m] = new BilinearForm(fespaces[m]);
       as[m]->AddDomainIntegrator(new DiffusionIntegrator);
       as[m]->Assemble();
     }

     // Set up interior penalty integrator.
     double sigma = -1.0;
     double kappa = 2.0;
     InterfaceDGDiffusionIntegrator interface_integ(one, sigma, kappa);
     Array<SparseMatrix *> blockMats(numSub * numSub);
     for (int i = 0; i < numSub; i++) {
       for (int j = 0; j < numSub; j++) {
         // row major order.
         int bidx = i * numSub + j;
         if (i == j) {
           blockMats[bidx] = &(as[i]->SpMat());
         } else {
           blockMats[bidx] = new SparseMatrix(fespaces[i]->GetTrueVSize(), fespaces[j]->GetTrueVSize());
         }
       }
     }

     // TODO: will need to loop over all interface boundary element pairs.
     int skip_zeros = 1;
     for (int m = 0; m < numSub - 1; m++) {
       DenseMatrix elemmat;
       FaceElementTransformations *tr1, *tr2;
       const FiniteElement *fe1, *fe2;
       Array<Array<int> *> vdofs(2);
       vdofs[0] = new Array<int>;
       vdofs[1] = new Array<int>;
       tr1 = meshes[m]->GetBdrFaceTransformations(1);
       tr2 = meshes[m+1]->GetBdrFaceTransformations(0);
       if ((tr1 != NULL) && (tr2 != NULL))
       {
          fespaces[m]->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
          fespaces[m+1]->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
          // Both domains will have the adjacent element as Elem1.
          fe1 = fespaces[m]->GetFE(tr1->Elem1No);
          fe2 = fespaces[m+1]->GetFE(tr2->Elem1No);

          interface_integ.AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmat);

          DenseMatrix subelemmat;
          Array<int> block_offsets(3);
          block_offsets[0] = 0;
          block_offsets[1] = fe1->GetDof();
          block_offsets[2] = fe2->GetDof();
          block_offsets.PartialSum();
          for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
              elemmat.GetSubMatrix(block_offsets[i], block_offsets[i+1],
                                   block_offsets[j], block_offsets[j+1], subelemmat);
              blockMats[(m+i) * numSub + (m+j)]->AddSubMatrix(*vdofs[i], *vdofs[j], subelemmat, skip_zeros);
            }
          }
       }
     }

     // Array<SparseMatrix *> As(numSub);
     // DenseMatrix Ad;
     // Vector B, X;
     // // a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
     // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

     Array<SparseMatrix *> As(numSub);
     Array<Vector *> Bs(numSub), Xs(numSub);
     // TODO: higher dimension will need off-diagonal block matrix handling as well.
     for (int i = 0; i < numSub; i++) {
       As[i] = new SparseMatrix;
       Bs[i] = new Vector;
       Xs[i] = new Vector;
       as[i]->FormLinearSystem(*ess_tdof_lists[i], *xs[i], *bs[i], *As[i], *Xs[i], *Bs[i]);
     }

     for (int i = 0; i < numSub; i++) {
       for (int j = 0; j < numSub; j++) {
         printf("Block matrix (%d, %d)", i, j);
         // row major order.
         int bidx = i * numSub + j;
         DenseMatrix blockmat;
         blockMats[bidx]->Finalize();
         blockMats[bidx]->ToDenseMatrix(blockmat);
         printf(": (%d x %d)\n", blockmat.Width(), blockmat.Height());

         for (int h = 0; h < blockmat.Height(); h++) {
           for (int w = 0; w < blockmat.Width(); w++) {
             printf("%2.3f\t", blockmat(h, w));
           }
           printf("\n");
         }
       }
     }

     Array<int> block_offsets(numSub + 1); // number of subdomain + 1
     block_offsets[0] = 0;
     for (int i = 0; i < numSub; i++) {
       block_offsets[i + 1] = fespaces[i]->GetTrueVSize();
     }
     block_offsets.PartialSum();

     BlockOperator globalA(block_offsets);
     for (int i = 0; i < numSub; i++) {
       for (int j = 0; j < numSub; j++) {
         // row major order.
         int bidx = i * numSub + j;
         globalA.SetBlock(i, j, blockMats[bidx]);
       }
     }

     BlockVector globalX(block_offsets), globalRHS(block_offsets);
     Vector R(block_offsets.Last());
     R = 0.0;
     for (int i = 0; i < numSub; i++) {
       for (int n = block_offsets[i]; n < block_offsets[i + 1]; n++) {
         R(n) = (*Bs[i])(n - block_offsets[i]);
       }
     }
     globalRHS.Update(R, block_offsets);
     for (int i = 0; i < block_offsets.Last(); i++) {
       printf("RHS[%d] = %2.3f\n", i, globalRHS[i]);
     }

     int maxIter(1000);
     double rtol(1.e-6);
     double atol(1.e-10);

     BlockDiagonalPreconditioner globalPrec(block_offsets);
     GMRESSolver solver;
     solver.SetAbsTol(atol);
     solver.SetRelTol(rtol);
     solver.SetMaxIter(maxIter);
     solver.SetOperator(globalA);
     solver.SetPreconditioner(globalPrec);
     solver.SetPrintLevel(1);
     globalX = 0.0;
     solver.Mult(globalRHS, globalX);

     for (int i = 0; i < block_offsets.Last(); i++) {
       printf("xb[%d] = %2.3f\n", i, globalX[i]);
     }

   } else if (dode_type_str == "feti") {

     //Now solve with FETI method for two subdomains.
     printf("Solving with FETI method.\n");

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
     GMRESSolver solver;
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

   } else {
     printf("Unknown domain decomposition type: %s!\n", dode_type);
     exit(1);
   }


   return 0;
}
