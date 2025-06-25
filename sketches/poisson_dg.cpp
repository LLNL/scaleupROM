// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include "interfaceinteg.hpp"
// #include "multiblock_solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dbc1(const Vector &);
double dbc3(const Vector &);
void shiftMesh(const Vector &, Vector &);

Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm);
void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map=NULL);
void UpdateBdrAttributes(Mesh& m);

struct InterfaceInfo {
   int Attr;
   int Mesh1, Mesh2;
   int BE1, BE2;

   // Inf = 64 * LocalFaceIndex + FaceOrientation
   int Inf1, Inf2;
};

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
  args.AddOption(&printFOMMat, "-p", "--print-fom", "-np", "--no-print-fom", "Print FOM matrix.");
  args.ParseCheck();

  // 3. Read the serial mesh from the given mesh file.
  Mesh mesh(mesh_file);
  int globalNE = mesh.GetNE();

  // 5. Define a finite element space on the mesh. Here we use H1 continuous
  //    high-order Lagrange finite elements of the given order.
  FiniteElementCollection *fec = new DG_FECollection(order, mesh.Dimension());
  FiniteElementSpace fespace(&mesh, fec);
  int total_num_dofs = fespace.GetTrueVSize();
  cout << "Number of unknowns: " << total_num_dofs << endl;

  // 7. Define the solution x as a finite element grid function in fespace. Set
  //    the initial guess to zero, which also sets the boundary conditions.
  GridFunction x(&fespace);
  x = 0.0;

  // add dirichlet boundary condition.
  // Coefficient *bdrCoeffs[mesh.bdr_attributes.Max()];
  Array<Coefficient *> bdrCoeffs(mesh.bdr_attributes.Max());
  bdrCoeffs[0] = new ConstantCoefficient(0.0);
  bdrCoeffs[1] = new FunctionCoefficient(dbc1);
  // bdrCoeffs[1] = new ConstantCoefficient(2.0);
  bdrCoeffs[2] = NULL;
  bdrCoeffs[3] = new FunctionCoefficient(dbc3);
  // bdrCoeffs[3] = NULL;

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
  a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
  a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[0]);
  a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[1]);
  a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[3]);
  a.Assemble();
  a.Finalize();

  // 10. Form the linear system A X = B. This includes eliminating boundary
  //     conditions, applying AMR constraints, parallel assembly, etc.
  //  SparseMatrix A;
  DenseMatrix Ad;
  //  Vector B, X;
  // a.FormLinearSystem(boundary_dofs, x, b, A, X, B);
  //  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  if (printFOMMat) {
    DenseMatrix mat, submat;
    printf("a.mat\n");
    a.SpMat().ToDenseMatrix(mat);

    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    block_offsets[1] = a.Size() / 2;
    block_offsets[2] = a.Size();
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
      printf("Block (%d, %d)\n", i, j);
      mat.GetSubMatrix(block_offsets[i], block_offsets[i+1],
                        block_offsets[j], block_offsets[j+1], submat);
        for (int k = 0; k < submat.Height(); k++) {
          for (int l = 0; l < submat.Width(); l++) {
            printf("%2.3f\t", submat(k, l));
          }
          printf("\n");
        }
      }
    }

  //  printf("a.mat_e\n");
  //  a.SpMatElim().ToDenseMatrix(mat_e);
  //  for (int i = 0; i < a.Size(); i++) {
  //    for (int j = 0; j < a.Size(); j++) {
  //      printf("%2.3f\t", mat_e(i, j));
  //    }
  //    printf("\n");
  //  }

  //  A.ToDenseMatrix(Ad);
  //  printf("A (%d x %d)\n", Ad.Width(), Ad.Height());

  //  for (int h = 0; h < Ad.Height(); h++) {
  //    for (int w = 0; w < Ad.Width(); w++) {
  //      printf("%2.3f\t", Ad(h, w));
  //    }
  //    printf(" | %2.3f \n", B(h));
  //  }
    printf("global b.\n");
    for (int k = 0; k < b.Size(); k++) {
      printf("%.3f\n", b[k]);
    }
  }

  // 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
  int maxIter(1000);
  double rtol(1.e-6);
  double atol(1.e-10);

  //  GSSmoother M(A);
  //  PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
  //  CG(A, B, X, 1, 200, 1e-12, 0.0);
  CGSolver solver;
  solver.SetAbsTol(atol);
  solver.SetRelTol(rtol);
  solver.SetMaxIter(maxIter);
  solver.SetOperator(a);
  //  solver.SetPreconditioner(globalPrec);
  solver.SetPrintLevel(1);
  //  X = 0.0;
  solver.Mult(b, x);

  // 12. Recover the solution x as a grid function and save to file. The output
  //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
  //  a.RecoverFEMSolution(X, b, x);
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

    // MultiBlockSolver test(argc, argv);
    // {
    //   test.SetupBoundaryConditions(bdrCoeffs);

    //   test.InitVariables();
    //   test.InitVisualization();

    //   test.BuildOperators();

    //   test.SetupBCOperators();

    //   test.Assemble();
    // }

    DG_FECollection fec(order, mesh.Dimension());

    // Create the sub-domains and accompanying Finite Element spaces from
    // corresponding attributes.
    int numSub = mesh.attributes.Max();
    std::vector<std::shared_ptr<SubMesh>> meshes(numSub);
    for (int k = 0; k < numSub; k++) {
      Array<int> domain_attributes(1);
      domain_attributes[0] = k+1;

      meshes[k] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(mesh, domain_attributes));
    }
    
    // NOTE: currently submesh does not generate face map for 2d mesh..
    Array<Array<int> *> parent_face_map_2d(numSub);
    for (int k = 0; k < numSub; k++) {
        parent_face_map_2d[k] = new Array<int>(BuildFaceMap2D(mesh, *meshes[k]));
        BuildSubMeshBoundary2D(mesh, *meshes[k], parent_face_map_2d[k]);
    }

    Array<Array<int> *> parent_el_map(numSub);
    for (int k = 0; k < numSub; k++) {
      parent_el_map[k] = new Array<int>(meshes[k]->GetParentElementIDMap());
    }

    for (int i = 0; i < numSub; i++) {
        printf("Submesh %d\n", i);
        for (int k = 0; k < meshes[i]->GetNBE(); k++) {
          printf("bdr element %d attribute: %d\n", k, meshes[i]->GetBdrAttribute(k));
        }

        // Setting a new boundary attribute does not append bdr_attributes.
        printf("submesh nbe: %d\n", meshes[i]->GetNBE());
        for (int k = 0; k < meshes[i]->bdr_attributes.Size(); k++) {
          printf("bdr attribute %d: %d\n", k, meshes[i]->bdr_attributes[k]);
        }

        int nfaces = meshes[i]->GetNumFaces();
        printf("submesh nfaces: %d\n", nfaces);
    }

    Array2D<int> interface_attributes(numSub, numSub);
    interface_attributes = -1;

    // NOTE(kevin): MFEM v4.6 SubMesh uses this for generated boundary attributes.
    const int generated_battr = mesh.bdr_attributes.Max() + 1;

    // interface attribute starts after the parent mesh boundary attributes.
    Array<InterfaceInfo> interface_infos(0);
    int if_attr = mesh.bdr_attributes.Max() + 1;
    for (int i = 0; i < numSub; i++) {
        // printf("Submesh %d\n", i);
        for (int ib = 0; ib < meshes[i]->GetNBE(); ib++) {
          if (meshes[i]->GetBdrAttribute(ib) != generated_battr) continue;

          int parent_face_i = (*parent_face_map_2d[i])[meshes[i]->GetBdrElementFaceIndex(ib)];
          for (int j = i+1; j < numSub; j++) {
              for (int jb = 0; jb < meshes[j]->GetNBE(); jb++) {
                int parent_face_j = (*parent_face_map_2d[j])[meshes[j]->GetBdrElementFaceIndex(jb)];
                if (parent_face_i == parent_face_j) {
                    MFEM_ASSERT(meshes[j]->GetBdrAttribute(jb) == generated_battr,
                                "This interface element has been already set!");
                    if (interface_attributes[i][j] <= 0) {
                      interface_attributes[i][j] = if_attr;
                      if_attr += 1;
                    }

                    Array<int> Infs(2);
                    {
                      Mesh::FaceInformation face_info = mesh.GetFaceInformation(parent_face_i);
                      
                      int face_inf[2];
                      mesh.GetFaceInfos(parent_face_i, &face_inf[0], &face_inf[1]);
                      int eli, eli_info;
                      meshes[i]->GetBdrElementAdjacentElement(ib, eli, eli_info);
                      eli = (*parent_el_map[i])[eli];
                      int elj, elj_info;
                      meshes[j]->GetBdrElementAdjacentElement(jb, elj, elj_info);
                      elj = (*parent_el_map[j])[elj];

                      if (eli == face_info.element[0].index) {
                          Infs[0] = face_inf[0];
                          Infs[1] = face_inf[1];
                      } else {
                          Infs[0] = face_inf[1];
                          Infs[1] = face_inf[0];
                      }
                    }

                    meshes[i]->SetBdrAttribute(ib, interface_attributes[i][j]);
                    meshes[j]->SetBdrAttribute(jb, interface_attributes[i][j]);

                    // submesh usually can inherit multiple attributes from parent.
                    // we limit to single-attribute case where attribute = index + 1;
                    interface_infos.Append(InterfaceInfo({.Attr = interface_attributes[i][j],
                                                        .Mesh1 = i, .Mesh2 = j,
                                                        .BE1 = ib, .BE2 = jb,
                                                        .Inf1 = Infs[0], .Inf2 = Infs[1]}));
                }
              }
          }
        }
    }

    for (int i = 0; i < numSub; i++) UpdateBdrAttributes(*meshes[i]);

    Array<int> interface_parent(0);
    for (int i = 0; i < numSub; i++) {
        printf("Submesh %d\n", i);
        for (int ib = 0; ib < meshes[i]->GetNBE(); ib++) {
          int interface_attr = meshes[i]->GetBdrAttribute(ib);
          if (interface_attr <= mesh.bdr_attributes.Max()) continue;

          int parent_face_i = (*parent_face_map_2d[i])[meshes[i]->GetBdrElementFaceIndex(ib)];
          
          for (int j = 0; j < numSub; j++) {
              if (i == j) continue;
              for (int jb = 0; jb < meshes[j]->GetNBE(); jb++) {
                int parent_face_j = (*parent_face_map_2d[j])[meshes[j]->GetBdrElementFaceIndex(jb)];
                if (parent_face_i == parent_face_j) {
                  interface_parent.Append(parent_face_i);
                  printf("(BE %d, face %d) - parent face %d, attr %d - Submesh %d (BE %d, face %d)\n",
                        ib, meshes[i]->GetBdrElementFaceIndex(ib), parent_face_i, interface_attr, j, jb, meshes[j]->GetBdrElementFaceIndex(jb));
                }
              }
          }
        }
    }

    for (int k = 0; k < interface_infos.Size(); k++) {
        printf("(Mesh %d, BE %d) - Attr %d - (Mesh %d, BE %d)\n",
              interface_infos[k].Mesh1, interface_infos[k].BE1, interface_infos[k].Attr,
              interface_infos[k].Mesh2, interface_infos[k].BE2);
    }

    Array<FiniteElementSpace *> fespaces(numSub);
    for (int m = 0; m < numSub; m++) {
      fespaces[m] = new FiniteElementSpace(&(*meshes[m]), &fec);
    }
    int total_num_dofs = fespaces[0]->GetTrueVSize();
    cout << "Number of unknowns in each subdomain: " << total_num_dofs << endl;

    for (int i = 0; i < meshes[0]->GetNBE(); i++) {
      int fn = meshes[0]->GetBdrElementFaceIndex(i);
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

  //  // TODO: Manually pair the interface elements now.
  //  Array<Array<int> *> interface_pairs(0);
  //  // Boundary 2 - 4 pair
  //  interface_pairs.Append(new Array<int>({12, 8}));
  //  interface_pairs.Append(new Array<int>({13, 9}));
  //  interface_pairs.Append(new Array<int>({14, 10}));
  //  interface_pairs.Append(new Array<int>({15, 11}));

    // Boundary conditions are weakly constrained.
    Array<Array<int> *> ess_attrs(numSub);
    Array<Array<int> *> ess_tdof_lists(numSub);
    for (int m = 0; m < numSub; m++) {
      ess_attrs[m] = new Array<int>(meshes[m]->bdr_attributes.Max());
      (*ess_attrs[m]) = 0;
      ess_tdof_lists[m] = new Array<int>;
      fespaces[m]->GetEssentialTrueDofs((*ess_attrs[m]), (*ess_tdof_lists[m]));
    }

    Array<int> block_offsets(numSub + 1); // number of subdomain + 1
    block_offsets[0] = 0;
    for (int i = 0; i < numSub; i++) {
      block_offsets[i + 1] = fespaces[i]->GetTrueVSize();
    }
    block_offsets.PartialSum();

    BlockVector globalX(block_offsets), globalRHS(block_offsets);

    // 7. Define the solution x as a finite element grid function in fespace. Set
    //    the initial guess to zero, which also sets the boundary conditions.
    Array<GridFunction *> xs(numSub), rhs(numSub);
    for (int m = 0; m < numSub; m++) {
      xs[m] = new GridFunction(fespaces[m]);
      xs[m]->MakeTRef(fespaces[m], globalX.GetBlock(m), 0);
      (*xs[m]) = 0.0;

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      xs[m]->SetTrueVector();

      rhs[m] = new GridFunction(fespaces[m]);
      rhs[m]->MakeTRef(fespaces[m], globalRHS.GetBlock(m), 0);
      (*rhs[m]) = 0.0;
    }

    int max_bdr_attr = -1;
    for (int m = 0; m < numSub; m++) {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
    }
    Array<Array<int> *> bdr_markers(max_bdr_attr);
    for (int k = 0; k < max_bdr_attr; k++) {
    bdr_markers[k] = new Array<int>(max_bdr_attr);
    (*bdr_markers[k]) = 0;
    (*bdr_markers[k])[k] = 1;
    }

    // 8. Set up the linear form b(.) corresponding to the right-hand side.
    Array<LinearForm *> bs(numSub);
    Array<BilinearForm *> as(numSub);
    // NOTE: each subdomain needs bdr_marker array for each boundary index, and these cannot be modified afterward.
    // Since they are never used after adding integrators, simply appending them in one array.
  //  Array<Array<int> *> bdr_markers(0);
    for (int m = 0; m < numSub; m++) {
  printf("Mesh %d\n", m);
      bs[m] = new LinearForm(fespaces[m]);
      bs[m]->AddDomainIntegrator(new DomainLFIntegrator(one));

      as[m] = new BilinearForm(fespaces[m]);
      as[m]->AddDomainIntegrator(new DiffusionIntegrator);
      // as[m]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
      as[m]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));

  printf("submesh boundaries: ");
  for (int k = 0; k < meshes[m]->bdr_attributes.Size(); k++) printf("%d\t", meshes[m]->bdr_attributes[k]);
  printf("\n");

      for (int b = 0; b < mesh.bdr_attributes.Max(); b++) 
      {
      int idx = meshes[m]->bdr_attributes.Find(mesh.bdr_attributes[b]);
      if (idx < 0) continue;
      if (bdrCoeffs[b] == NULL) continue;

  // printf("boundary %d\n", mesh.bdr_attributes[b]);
  // for (int k = 0; k < bdr_markers.Last()->Size(); k++) printf("bdr marker[%d] = %d\n", k, (*(bdr_markers.Last()))[k]);

      // bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[b], one, sigma, kappa), *bdr_markers[b]);
      // as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[b]);
      bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[b], sigma, kappa), *bdr_markers[b]);
      as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), *bdr_markers[b]);
      }

    //  // Bottom boundary condition for all subdomains.
    //  bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[0], one, sigma, kappa), *bdr_markers[0]);
    //  as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[0]);

    //  // Left, right boundary for leftmost/rightmost subdomain.
    //  if (m == 0) {
    //    bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[3], one, sigma, kappa), *bdr_markers[3]);
    //    as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[3]);
    //  } else if (m == numSub - 1) {
    //    bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdrCoeffs[1], one, sigma, kappa), *bdr_markers[1]);
    //    as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), *bdr_markers[1]);
    //  }

      bs[m]->Assemble();
      as[m]->Assemble();
    }

    // Set up interior penalty integrator.
    // InterfaceDGDiffusionIntegrator interface_integ(one, sigma, kappa);
    InterfaceDGDiffusionIntegrator interface_integ(sigma, kappa);
    Array2D<SparseMatrix *> blockMats(numSub, numSub);
    for (int i = 0; i < numSub; i++) {
      for (int j = 0; j < numSub; j++) {
      //  // row major order.
      //  int bidx = i * numSub + j;
        if (i == j) {
          blockMats(i, i) = &(as[i]->SpMat());
        } else {
          blockMats(i, j) = new SparseMatrix(fespaces[i]->GetTrueVSize(), fespaces[j]->GetTrueVSize());
        }
      }
    }

    int skip_zeros = 1;
    for (int bn = 0; bn < interface_infos.Size(); bn++)
    {
      if (printFOMMat)
      { // compare with global interior face integrator.
        printf("\n\ninterior face integrator from global mesh.\n\n");
        DGDiffusionIntegrator interior_face_integ(one, sigma, kappa);
        FaceElementTransformations *tr;
        Array<int> vdofs, vdofs2;
        DenseMatrix elemmat;

        {
          int face_inf[2];
          mesh.GetFaceInfos(interface_parent[bn], &face_inf[0], &face_inf[1]);

          Mesh::FaceInformation face_info = mesh.GetFaceInformation(interface_parent[bn]);
          for (int j = 0; j < 2; j++) {
            printf("Face %d Element %d information\n", interface_parent[bn], j);
            printf("Index: %d\n", face_info.element[j].index);
            printf("Local Face ID: %d =? %d\n", face_info.element[j].local_face_id, face_inf[j] / 64);
            printf("Orientation: %d =? %d\n", face_info.element[j].orientation, face_inf[j] % 64);
          }
        }

        tr = mesh.GetInteriorFaceTransformations(interface_parent[bn]);
        if (tr != NULL)
        {
          fespace.GetElementVDofs(tr->Elem1No, vdofs);
          fespace.GetElementVDofs(tr->Elem2No, vdofs2);
          vdofs.Append(vdofs2);
          interior_face_integ.AssembleFaceMatrix(*fespace.GetFE(tr->Elem1No),
                                                *fespace.GetFE(tr->Elem2No),
                                                *tr, elemmat);

          if (printFOMMat)
          {
            printf("vdofs\n");
            for (int i = 0; i < vdofs.Size(); i++) {
              printf("%d\t", vdofs[i]);
            }
            printf("\n");
            printf("elemmat\n");
            for (int i = 0; i < elemmat.Height(); i++) {
              for (int j = 0; j < elemmat.Width(); j++) {
                printf("%.3f\t", elemmat(i,j));
              }
              printf("\n");
            }
          }
        }

      }

      printf("\n\ninterface integrator.\n\n");

      Mesh *mesh1, *mesh2;
      FiniteElementSpace *fes1, *fes2;
      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *fe1, *fe2;
      Array<Array<int> *> vdofs(2);
      vdofs[0] = new Array<int>;
      vdofs[1] = new Array<int>;

      Array<int> midx(2);
      midx[0] = interface_infos[bn].Mesh1;
      midx[1] = interface_infos[bn].Mesh2;

      mesh1 = &(*meshes[midx[0]]);
      mesh2 = &(*meshes[midx[1]]);
      fes1 = fespaces[midx[0]];
      fes2 = fespaces[midx[1]];

      // Mesh sets face element transformation based on the face_info.
      // For boundary face, the adjacent element is always on element 1, and its orientation is "by convention" always zero.
      // This is a problem for the interface between two meshes, where both element orientations are zero.
      // At least one element should reflect a relative orientation with respect to the other.
      // Currently this is done by hijacking global mesh face information in the beginning.
      // If we would want to do more flexible global mesh building, e.g. rotating component submeshes,
      // then we will need to figure out how to actually determine relative orientation.

      // We cannot write a function that replaces this, since only Mesh can access to FaceElemTr.SetConfigurationMask.
      tr1 = mesh1->GetBdrFaceTransformations(interface_infos[bn].BE1);
      tr2 = mesh2->GetBdrFaceTransformations(interface_infos[bn].BE2);

      // Correcting the local face transformation if orientation needs correction.
      {
        int faceInf1, faceInf2;
        int face1 = mesh1->GetBdrElementFaceIndex(interface_infos[bn].BE1);
        mesh1->GetFaceInfos(face1, &faceInf1, &faceInf2);
        if (faceInf1 != interface_infos[bn].Inf1)
        {
          if ((faceInf1 / 64) != (interface_infos[bn].Inf1 / 64))
          {
            MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
          }

          int face_type = mesh1->GetFaceElementType(face1);
          int elem_type = mesh1->GetElementType((*tr1).Elem1No);

          mesh1->GetLocalFaceTransformation(face_type, elem_type,
                                            (*tr1).Loc1.Transf, interface_infos[bn].Inf1);
        }

        int face2 = mesh2->GetBdrElementFaceIndex(interface_infos[bn].BE2);
        mesh2->GetFaceInfos(face2, &faceInf2, &faceInf1);
        if (faceInf2 != interface_infos[bn].Inf2)
        {
          if ((faceInf2 / 64) != (interface_infos[bn].Inf2 / 64))
          {
            MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
          }

          int face_type = mesh2->GetFaceElementType(face2);
          int elem_type = mesh2->GetElementType((*tr2).Elem1No);

          mesh2->GetLocalFaceTransformation(face_type, elem_type,
                                            (*tr2).Loc1.Transf, interface_infos[bn].Inf2);
        }
      }

      if (printFOMMat)
      {
        printf("\telem1\telem2\n");
        printf("tr1\t%d\t%d\n", tr1->Elem1No, tr1->Elem2No);
        printf("tr2\t%d\t%d\n", tr2->Elem1No, tr2->Elem2No);
        printf("\n");

        int face1 = mesh1->GetBdrElementFaceIndex(interface_infos[bn].BE1);
        int face2 = mesh2->GetBdrElementFaceIndex(interface_infos[bn].BE2);

        printf("mesh1 face info\n");

        int face_inf[2];
        mesh1->GetFaceInfos(face1, &face_inf[0], &face_inf[1]);

        Mesh::FaceInformation face_info = mesh1->GetFaceInformation(face1);
        for (int j = 0; j < 2; j++) {
          printf("Face %d Element %d information\n", face1, j);
          printf("Index: %d\n", face_info.element[j].index);
          printf("Local Face ID: %d =? %d | %d\n", face_info.element[j].local_face_id, face_inf[j] / 64,
                                                    interface_infos[bn].Inf1 / 64);
          printf("Orientation: %d =? %d | %d\n", face_info.element[j].orientation, face_inf[j] % 64,
                                                  interface_infos[bn].Inf1 % 64);
        }

        printf("mesh2 face info\n");

        mesh2->GetFaceInfos(face2, &face_inf[0], &face_inf[1]);

        face_info = mesh2->GetFaceInformation(face2);
        for (int j = 0; j < 2; j++) {
          printf("Face %d Element %d information\n", face2, j);
          printf("Index: %d\n", face_info.element[j].index);
          printf("Local Face ID: %d =? %d | %d\n", face_info.element[j].local_face_id, face_inf[j] / 64,
                                                    interface_infos[bn].Inf2 / 64);
          printf("Orientation: %d =? %d | %d\n", face_info.element[j].orientation, face_inf[j] % 64,
                                                  interface_infos[bn].Inf2 % 64);
        }
      }

      if ((tr1 != NULL) && (tr2 != NULL))
      {
        fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
        fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
        // Both domains will have the adjacent element as Elem1.
        fe1 = fes1->GetFE(tr1->Elem1No);
        fe2 = fes2->GetFE(tr2->Elem1No);

        interface_integ.AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);
        if (printFOMMat)
        {
          printf("vdof1\n");
          for (int i = 0; i < vdofs[0]->Size(); i++) {
            printf("%d\t", (*vdofs[0])[i]);
          }
          printf("\n");
          printf("vdof2\n");
          for (int i = 0; i < vdofs[1]->Size(); i++) {
            printf("%d\t", (*vdofs[1])[i]);
          }
          printf("\n");
          for (int I = 0; I < 2; I++)
          {
            for (int J = 0; J < 2; J++)
            {
              printf("elemmat(%d,%d)\n", I, J);
              DenseMatrix *elemmat = elemmats(I,J);
              for (int i = 0; i < elemmat->NumRows(); i++) {
                for (int j = 0; j < elemmat->NumCols(); j++) {
                  printf("%.3f\t", (*elemmat)(i,j));
                }
                printf("\n");
              }
            } // for (int J = 0; J < 2; J++)
          } // for (int I = 0; I < 2; I++)
        }

        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            // TODO: can change to Array2D.
            blockMats(midx[i], midx[j])->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
          }
        }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
    }  // for (int bn = 0; bn < interface_infos.Size(); bn++)

    Array<SparseMatrix *> As(numSub);
    Array<Vector *> Bs(numSub), Xs(numSub);
    // Array<SparseMatrix *> blockMats_e(numSub * numSub);
    for (int i = 0; i < numSub; i++) {
      As[i] = new SparseMatrix;
      Bs[i] = new Vector;
      Xs[i] = new Vector;
      as[i]->FormLinearSystem(*ess_tdof_lists[i], *xs[i], *bs[i], *As[i], *Xs[i], *Bs[i]);

    }  // for (int i = 0; i < numSub; i++)

  if (printFOMMat) {
    for (int i = 0; i < numSub; i++) {
      for (int j = 0; j < numSub; j++) {
        blockMats(i, j)->Finalize();
        DenseMatrix tmp;
        blockMats(i, j)->ToDenseMatrix(tmp);
        printf("Block[%d, %d] = (%d x %d) \n", i, j, tmp.Height(), tmp.Width());
        for (int k = 0; k < tmp.Height(); k++) {
          for (int l = 0; l < tmp.Width(); l++) {
            printf("%.3f\t", tmp(k,l));
          }
          printf("\n");
        }
      }
    }

    for (int m = 0; m < numSub; m++) {
      printf("b[%d].\n", m);
      for (int k = 0; k < Bs[m]->Size(); k++) {
        printf("%.3f\n", (*Bs[m])[k]);
      }
    }

    // printf("From MultiBlock Solver.\n");
    // for (int i = 0; i < numSub; i++) {
    //   for (int j = 0; j < numSub; j++) {
    //     DenseMatrix tmp;
    //     test.mats(i, j)->ToDenseMatrix(tmp);
    //     printf("Block[%d, %d] = (%d x %d) \n", i, j, tmp.Height(), tmp.Width());
    //     for (int k = 0; k < tmp.Height(); k++) {
    //       for (int l = 0; l < tmp.Width(); l++) {
    //         printf("%.3f\t", tmp(k,l));
    //       }
    //       printf("\n");
    //     }
    //   }
    // }

    // printf("b.\n");
    // for (int k = 0; k < test.RHS->Size(); k++) {
    //   printf("%.3f\n", (*(test.RHS))[k]);
    // }
  }

    BlockOperator globalA(block_offsets);
    for (int i = 0; i < numSub; i++) {
      for (int j = 0; j < numSub; j++) {
      //  // row major order.
      //  int bidx = i * numSub + j;
        globalA.SetBlock(i, j, blockMats(i, j));
      }
    }

    // TODO: grid functions can be initialized with this global X.
    Vector R(block_offsets.Last());
    R = 0.0;
    for (int i = 0; i < numSub; i++) {
      for (int n = block_offsets[i]; n < block_offsets[i + 1]; n++) {
        R(n) = (*Bs[i])(n - block_offsets[i]);
      }
    }
    globalRHS.Update(R, block_offsets);
  //  for (int i = 0; i < block_offsets.Last(); i++) {
  //    printf("RHS[%d] = %2.3f\n", i, globalRHS[i]);
  //  }

    int maxIter(1000);
    double rtol(1.e-6);
    double atol(1.e-10);

    BlockDiagonalPreconditioner globalPrec(block_offsets);
    CGSolver solver;
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(globalA);
  //  solver.SetPreconditioner(globalPrec);
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
      paraviewColl = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
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

Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm)
{
  // TODO: Check if parent is really a parent of mesh
  MFEM_ASSERT(pm.Dimension() == 2, "Support only 2-dimension meshes!");
  MFEM_ASSERT(sm.Dimension() == 2, "Support only 2-dimension meshes!");

  Array<int> parent_element_ids = sm.GetParentElementIDMap();

  Array<int> pfids(sm.GetNumFaces());
  pfids = -1;
  for (int i = 0; i < sm.GetNE(); i++)
  {
    int peid = parent_element_ids[i];
    Array<int> sel_faces, pel_faces, o;
    sm.GetElementEdges(i, sel_faces, o);
    pm.GetElementEdges(peid, pel_faces, o);

    MFEM_ASSERT(sel_faces.Size() == pel_faces.Size(), "internal error");
    for (int j = 0; j < sel_faces.Size(); j++)
    {
        if (pfids[sel_faces[j]] != -1)
        {
          MFEM_ASSERT(pfids[sel_faces[j]] == pel_faces[j], "internal error");
        }
        pfids[sel_faces[j]] = pel_faces[j];
    }
  }
  return pfids;
}

void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map)
{
  MFEM_ASSERT(pm.Dimension() == 2, "Support only 2-dimension meshes!");
  MFEM_ASSERT(sm.Dimension() == 2, "Support only 2-dimension meshes!");

  // Array<int> parent_face_map = submesh.GetParentFaceIDMap();
  if (parent_face_map == NULL)
    parent_face_map = new Array<int>(BuildFaceMap2D(pm, sm));

  // NOTE(kevin): MFEM v4.6 SubMesh uses this for generated boundary attributes.
  const int generated_battr = pm.bdr_attributes.Max() + 1;

  // Setting boundary element attribute of submesh for 2D.
  // This does not support 2D.
  // Array<int> parent_face_to_be = mesh.GetFaceToBdrElMap();
  Array<int> parent_face_to_be(pm.GetNumFaces());
  parent_face_to_be = -1;
  for (int i = 0; i < pm.GetNBE(); i++)
  {
    parent_face_to_be[pm.GetBdrElementFaceIndex(i)] = i;
  }
  for (int k = 0; k < sm.GetNBE(); k++) {
    int pbeid = parent_face_to_be[(*parent_face_map)[sm.GetBdrElementFaceIndex(k)]];
    if (pbeid != -1)
    {
      int attr = pm.GetBdrElement(pbeid)->GetAttribute();
      sm.GetBdrElement(k)->SetAttribute(attr);
    }
    else
    {
      // This case happens when a domain is extracted, but the root parent
      // mesh didn't have a boundary element on the surface that defined
      // it's boundary. It still creates a valid mesh, so we allow it.
      sm.GetBdrElement(k)->SetAttribute(generated_battr);
    }
  }

  UpdateBdrAttributes(sm);
}

void UpdateBdrAttributes(Mesh& m)
{
  m.bdr_attributes.DeleteAll();
  for (int k = 0; k < m.GetNBE(); k++) {
    int attr = m.GetBdrAttribute(k);
    int inBdrAttr = m.bdr_attributes.Find(attr);
    if (inBdrAttr < 0) m.bdr_attributes.Append(attr);
  }
}