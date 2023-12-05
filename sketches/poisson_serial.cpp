// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dbc1(const Vector &);
double dbc3(const Vector &);
void shiftMesh(const Vector &, Vector &);

int main(int argc, char *argv[])
{
   // // 1. Initialize MPI and HYPRE.
   // Mpi::Init(argc, argv);
   // Hypre::Init();

   // 2. Parse command line options.
   const char *mesh_file = "../data/star.mesh";
   const char *dode_type = "no-dd";
   int order = 1;
   bool printFOMMat = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&dode_type, "-d", "--domain-decomposition", "Domain decomposition type: feti, ddlspg, ip.");
   // args.AddOption(&printFOMMat, "-p", "--print-fom", "-np", "--no-print-fom", "Print FOM matrix.");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh mesh(mesh_file);
   int globalNE = mesh.GetNE();

   // {// Building FaceInfo, FaceElementTransformations
   //   for (int i = 0; i < mesh.GetNBE(); i++) {
   //     // FaceInfo face_info;
   //     // int Elem1No, Elem2No, Elem1Inf, Elem2Inf;
   //     // int NCFace; // -1 if this is a regular conforming/boundary face;
   //     // face_info.NCFace = -1;
   //
   //     int elem_id, face_info;
   //     mesh.GetBdrElementAdjacentElement(i, elem_id, face_info);
   //     printf("Boundary element %d - Adjacent element %d\n", i, elem_id);
   //     printf("Face index: %d, face orientation : %d\n", face_info / 64, face_info % 64);
   //
   //     int fn = mesh.GetBdrFace(i);
   //     int face_inf[2];
   //     mesh.GetFaceInfos(fn, &face_inf[0], &face_inf[1]);
   //     printf("From faces_info - Face index: %d, face orientation : %d\n", face_inf[0] / 64, face_inf[0] % 64);
   //   }
   //
   //   int nfaces = mesh.GetNumFaces();
   //   for (int i = 0; i < nfaces; i++)
   //   {
   //      int face_inf[2];
   //      mesh.GetFaceInfos(i, &face_inf[0], &face_inf[1]);
   //
   //      Mesh::FaceInformation face_info = mesh.GetFaceInformation(i);
   //      for (int j = 0; j < 2; j++) {
   //        printf("Face %d Element %d information\n", i, j);
   //        printf("Index: %d\n", face_info.element[j].index);
   //        printf("Local Face ID: %d =? %d\n", face_info.element[j].local_face_id, face_inf[j] / 64);
   //        printf("Orientation: %d =? %d\n", face_info.element[j].orientation, face_inf[j] % 64);
   //      }
   //   }
   // }

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
   ess_attr = 0;
  //  ess_attr[0] = 0;
   ess_attr[2] = 0;
   // ess_attr[3] = 0;
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
   bdrCoeffs[1] = new FunctionCoefficient(dbc1);
   // bdrCoeffs[1] = new ConstantCoefficient(2.0);
   bdrCoeffs[2] = NULL;
   bdrCoeffs[3] = new FunctionCoefficient(dbc3);
   // bdrCoeffs[3] = NULL;
   for (int b = 0; b < mesh.bdr_attributes.Max(); b++) {
     // Determine which boundary attribute will use the b-th boundary coefficient.
     // Since all boundary attributes use different BCs, only one index is 'turned on'.
     bdrAttr = 0;
     bdrAttr[b] = 1;
     // Project the b-th boundary coefficient.
     x.ProjectBdrCoefficient(*bdrCoeffs[b], bdrAttr);
   }

   // boundary markers for each boundary attribute.
   Array<Array<int> *> bdr_markers(mesh.bdr_attributes.Max());
   for (int b = 0; b < bdr_markers.Size(); b++) {
     bdr_markers[b] = new Array<int>(mesh.bdr_attributes.Max());
     (*bdr_markers[b]) = 0;
     (*bdr_markers[b])[b] = 1;
   }

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   double sigma = -1.0;
   double kappa = (order + 1) * (order + 1);
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[0], one, sigma, kappa), *bdr_markers[0]);
   b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[1], one, sigma, kappa), *bdr_markers[1]);
   b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[3], one, sigma, kappa), *bdr_markers[3]);
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[0]);
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[1]);
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[3]);
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   SparseMatrix A;
   DenseMatrix Ad;
   Vector B, X;
   // a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   if (printFOMMat) {
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

     A.ToDenseMatrix(Ad);
     printf("A (%d x %d)\n", Ad.Width(), Ad.Height());

     for (int h = 0; h < Ad.Height(); h++) {
       for (int w = 0; w < Ad.Width(); w++) {
         printf("%2.3f\t", Ad(h, w));
       }
       printf(" | %2.3f \n", B(h));
     }
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

   std::string dode_type_str(dode_type);
   if (dode_type_str == "feti") {
     printf("FETI method not implemented yet!\n");
     exit(1);
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

     Array<Mesh *> meshes(numSub);
     meshes[0] = new Mesh(submesh);
     for (int m = 1; m < meshes.Size(); m++) {
       meshes[m] = new Mesh(*meshes[m-1]);
       meshes[m]->Transform(&shiftMesh);
     }

     H1_FECollection fec(order, submesh.Dimension());
     std::vector<FiniteElementSpace *> fespaces(numSub);
     for (int m = 0; m < numSub; m++) {
       fespaces[m] = new FiniteElementSpace(meshes[m], &fec);
     }
     int total_num_dofs = fespaces[0]->GetTrueVSize();
     cout << "Number of unknowns in each subdomain: " << total_num_dofs << endl;

     for (int i = 0; i < meshes[0]->GetNBE(); i++) {
       int fn = meshes[0]->GetBdrFace(i);
       int bdrAttr = meshes[0]->GetBdrAttribute(i);

       int elem_id, face_info;
       meshes[0]->GetBdrElementAdjacentElement(i, elem_id, face_info);
       Vector adjElCenter;
       meshes[0]->GetElementCenter(elem_id, adjElCenter);
       printf("Boundary element %d - boundary attribute %d, face index %d: ", i, bdrAttr, fn);
       for (int d = 0; d < adjElCenter.Size(); d++) {
         printf("%f, ", adjElCenter(d));
       }
       printf("\n");
     }

     // TODO: Manually pair the interface elements now.
     Array<Array<int> *> interface_pairs(0);
     // Boundary 2 - 4 pair
     interface_pairs.Append(new Array<int>({12, 8}));
     interface_pairs.Append(new Array<int>({13, 9}));
     interface_pairs.Append(new Array<int>({14, 10}));
     interface_pairs.Append(new Array<int>({15, 11}));

     // Boundary conditions are weakly constrained.
     Array<Array<int> *> ess_attrs(numSub);
     Array<Array<int> *> ess_tdof_lists(numSub);
     for (int m = 0; m < numSub; m++) {
       ess_attrs[m] = new Array<int>(meshes[m]->bdr_attributes.Max());
       (*ess_attrs[m]) = 0;
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

     // 8. Set up the linear form b(.) corresponding to the right-hand side.
     Array<LinearForm *> bs(numSub);
     Array<BilinearForm *> as(numSub);
     for (int m = 0; m < numSub; m++) {
       bs[m] = new LinearForm(fespaces[m]);
       bs[m]->AddDomainIntegrator(new DomainLFIntegrator(one));

       as[m] = new BilinearForm(fespaces[m]);
       as[m]->AddDomainIntegrator(new DiffusionIntegrator);

       // Bottom boundary condition for all subdomains.
       bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[0], one, sigma, kappa), *bdr_markers[0]);
       as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[0]);

       // Left, right boundary for leftmost/rightmost subdomain.
       if (m == 0) {
         bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[3], one, sigma, kappa), *bdr_markers[3]);
         as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[3]);
       } else if (m == numSub - 1) {
         bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[1], one, sigma, kappa), *bdr_markers[1]);
         as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[1]);
       }

       bs[m]->Assemble();
       as[m]->Assemble();
     }
     
     // Set up interior penalty integrator.
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

     int skip_zeros = 1;
     for (int m = 0; m < numSub - 1; m++) {
       for (int bn = 0; bn < interface_pairs.Size(); bn++) {
         Array2D<DenseMatrix*> elemmats;
         FaceElementTransformations *tr1, *tr2;
         const FiniteElement *fe1, *fe2;
         Array<Array<int> *> vdofs(2);
         vdofs[0] = new Array<int>;
         vdofs[1] = new Array<int>;
         tr1 = meshes[m]->GetBdrFaceTransformations((*interface_pairs[bn])[0]);
         tr2 = meshes[m+1]->GetBdrFaceTransformations((*interface_pairs[bn])[1]);
         if ((tr1 != NULL) && (tr2 != NULL))
         {
            fespaces[m]->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
            fespaces[m+1]->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
            // Both domains will have the adjacent element as Elem1.
            fe1 = fespaces[m]->GetFE(tr1->Elem1No);
            fe2 = fespaces[m+1]->GetFE(tr2->Elem1No);

            interface_integ.AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);

            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 2; j++) {
                blockMats[(m+i) * numSub + (m+j)]->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
              }
            }
         }  // if ((tr1 != NULL) && (tr2 != NULL))
       }  // for (int bn = 0; bn < interface_pairs.Size(); bn++)
     }  // for (int m = 0; m < numSub - 1; m++)

     Array<SparseMatrix *> As(numSub);
     Array<Vector *> Bs(numSub), Xs(numSub);
     // Array<SparseMatrix *> blockMats_e(numSub * numSub);
     for (int i = 0; i < numSub; i++) {
       As[i] = new SparseMatrix;
       Bs[i] = new Vector;
       Xs[i] = new Vector;
       as[i]->FormLinearSystem(*ess_tdof_lists[i], *xs[i], *bs[i], *As[i], *Xs[i], *Bs[i]);

       // NOTE: no need of essential dof handlings due to weakly-constrained bc.
       // // Off-diagonal blocks
       // for (int j = 0; j < numSub; j++) {
       //   if (i == j) continue;
       //
       //   // row major order.
       //   int bidx = i * numSub + j;
       //   blockMats_e[bidx] = new SparseMatrix(fespaces[i]->GetTrueVSize(), fespaces[j]->GetTrueVSize());
       //
       //   // From BilinearForm::EliminateVDofs
       //   ess_tdof_lists[j]->HostRead();
       //   for (int k = 0; k < ess_tdof_lists[j]->Size(); k++)
       //   {
       //      int vdof = (*ess_tdof_lists[j])[k];
       //      if ( vdof >= 0 )
       //      {
       //         blockMats[bidx]->EliminateRowCol(vdof, *blockMats_e[bidx], Matrix::DIAG_KEEP);
       //      }
       //      else
       //      {
       //         blockMats[bidx]->EliminateRowCol(-1-vdof, *blockMats_e[bidx], Matrix::DIAG_KEEP);
       //      }
       //   }
       //
       //   blockMats_e[bidx]->AddMult(*xs[j], *bs[i], -1.);
       // }  // for (int j = 0; j < numSub; j++)
     }  // for (int i = 0; i < numSub; i++)

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

     // TODO: grid functions can be initialized with this global X.
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
     CGSolver solver;
     solver.SetAbsTol(atol);
     solver.SetRelTol(rtol);
     solver.SetMaxIter(maxIter);
     solver.SetOperator(globalA);
     solver.SetPreconditioner(globalPrec);
     solver.SetPrintLevel(1);
     globalX = 0.0;
     solver.Mult(globalRHS, globalX);

     for (int m = 0; m < numSub; m++) {
       *xs[m] = globalX.GetBlock(m);
     }

     // Save visualization in paraview output.
     for (int m = 0; m < numSub; m++) {
       ostringstream oss;
       oss << "paraview_output_" << std::to_string(m);

       ParaViewDataCollection *paraviewColl = NULL;
       paraviewColl = new ParaViewDataCollection(oss.str().c_str(), meshes[m]);
       paraviewColl->SetLevelsOfDetail(order);
       paraviewColl->SetHighOrderOutput(true);
       paraviewColl->SetPrecision(8);

       paraviewColl->RegisterField("solution", xs[m]);
       paraviewColl->SetOwnData(true);
       paraviewColl->Save();
     }

   } else if (dode_type_str == "no-dd") {
     printf("No domain decomposition.\n");
   } else {
     printf("Unknown domain decomposition type: %s!\n", dode_type);
     exit(1);
   }

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

void shiftMesh(const Vector &x, Vector &y)
{
  y.SetSize(x.Size());
  y = x;
  y(0) += 1.0;
}
