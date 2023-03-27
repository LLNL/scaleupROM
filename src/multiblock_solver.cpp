// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "multiblock_solver.hpp"
#include "linalg_utils.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;
using namespace mfem;

MultiBlockSolver::MultiBlockSolver()
{
   ParseInputs();

   TopologyData topol_data;
   topol_handler = new SubMeshTopologyHandler(meshes, interface_infos, topol_data);
   numSub = topol_data.numSub;
   dim = topol_data.dim;
   global_bdr_attributes = topol_data.global_bdr_attributes;

   // // Initiate parent mesh.
   // // TODO: initiate without parent mesh.
   // std::string mesh_file = config.GetRequiredOption<std::string>("mesh/filename");
   // pmesh = new Mesh(mesh_file.c_str());
   // dim = pmesh->Dimension();

   // // Uniform refinement if specified.
   // int num_refinement = config.GetOption<int>("mesh/uniform_refinement", 0);
   // for (int k = 0; k < num_refinement; k++)
   //    pmesh->UniformRefinement();

   // // Initiate SubMeshes based on attributes.
   // // TODO: a sanity check?
   // switch (dd_mode)
   // {
   //    case DecompositionMode::NODD:
   //    {
   //       numSub = 1;
   //       meshes.resize(numSub);
   //       Array<int> domain_attributes(pmesh->attributes.Max());
   //       for (int k = 0; k < pmesh->attributes.Max(); k++) {
   //          domain_attributes[k] = k+1;
   //       }
   //       meshes[0] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(*pmesh, domain_attributes));
   //       break;
   //    }
   //    default:
   //    {
   //       numSub = pmesh->attributes.Max();
   //       meshes.resize(numSub);
   //       for (int k = 0; k < numSub; k++) {
   //          Array<int> domain_attributes(1);
   //          domain_attributes[0] = k+1;

   //          meshes[k] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(*pmesh, domain_attributes));
   //       }
   //       break;
   //    }
   // }

   // // Set up element mapping between submeshes and parent mesh.
   // parent_elem_map.SetSize(numSub);
   // for (int k = 0; k < numSub; k++) {
   //    parent_elem_map[k] = new Array<int>(meshes[k]->GetParentElementIDMap());
   // }

   // // Set up face mapping between submeshes and parent mesh.
   // parent_face_map.SetSize(numSub);
   // for (int k = 0; k < numSub; k++) {
   //    if (dim == 2)
   //    {
   //       parent_face_map[k] = new Array<int>(BuildFaceMap2D(*pmesh, *meshes[k]));
   //       BuildSubMeshBoundary2D(*pmesh, *meshes[k], parent_face_map[k]);
   //    }
   //    else
   //    {
   //       parent_face_map[k] = new Array<int>(meshes[k]->GetParentFaceIDMap());
   //    }
   // }

   // BuildInterfaceInfos();
   
   // Set up FE collection/spaces.
   if (full_dg)
   {
      fec = new DG_FECollection(order, dim);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }
   
   // solution dimension is determined by initialization.
   udim = 1;
   fes.SetSize(numSub);
   for (int m = 0; m < numSub; m++) {
      fes[m] = new FiniteElementSpace(&(*meshes[m]), fec, udim);
   }

}

MultiBlockSolver::~MultiBlockSolver()
{
   delete U;
   delete RHS;
   delete interface_integ;

   if (save_visual)
      for (int k = 0; k < paraviewColls.Size(); k++) delete paraviewColls[k];

   for (int k = 0; k < bs.Size(); k++) delete bs[k];
   for (int k = 0; k < as.Size(); k++) delete as[k];
   for (int k = 0; k < us.Size(); k++) delete us[k];
   // for (int k = 0; k < parent_elem_map.Size(); k++) delete parent_elem_map[k];
   // for (int k = 0; k < parent_face_map.Size(); k++) delete parent_face_map[k];
   for (int k = 0; k < fes.Size(); k++) delete fes[k];
   for (int k = 0; k < ess_attrs.Size(); k++) delete ess_attrs[k];
   for (int k = 0; k < ess_tdof_lists.Size(); k++) delete ess_tdof_lists[k];

   delete fec;
   // delete pmesh;

   for (int k = 0; k < bdr_markers.Size(); k++)
      delete bdr_markers[k];


   for (int k = 0; k < bdr_coeffs.Size(); k++)
      delete bdr_coeffs[k];
      
   for (int k = 0; k < rhs_coeffs.Size(); k++)
      delete rhs_coeffs[k];

   delete rom_handler;
   delete topol_handler;
}

void MultiBlockSolver::ParseInputs()
{
   order = config.GetOption<int>("discretization/order", 1);
   full_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);
   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));

   // solver option;
   use_monolithic = config.GetOption<bool>("solver/use_monolithic_operator", false);

   // std::string dd_mode_str = config.GetOption<std::string>("domain-decomposition/type", "interior_penalty");
   // if (dd_mode_str == "interior_penalty")
   // {
   //    dd_mode = DecompositionMode::IP;
   // }
   // else if (dd_mode_str == "feti")
   // {
   //    mfem_error("FETI not implemented!\n");
   // }
   // else if (dd_mode_str == "none")
   // {
   //    dd_mode = DecompositionMode::NODD;
   // }
   // else
   // {
   //    mfem_error("Unknown domain decomposition mode!\n");
   // }

   save_visual = config.GetOption<bool>("visualization/enabled", false);
   if (save_visual)
   {
      // Default file path if no input file name is provided.
      visual_dir = config.GetOption<std::string>("visualization/file_path/directory", ".");
      visual_prefix = config.GetOption<std::string>("visualization/file_path/prefix", "paraview_output");
   }

   // rom inputs.
   use_rom = config.GetOption<bool>("main/use_rom", false);
}

// Array<int> MultiBlockSolver::BuildFaceMap2D(const Mesh& pm, const SubMesh& sm)
// {
//   // TODO: Check if parent is really a parent of mesh
//   MFEM_ASSERT(pm.Dimension() == 2, "Support only 2-dimension meshes!");
//   MFEM_ASSERT(sm.Dimension() == 2, "Support only 2-dimension meshes!");

//   Array<int> parent_element_ids = sm.GetParentElementIDMap();

//   Array<int> pfids(sm.GetNumFaces());
//   pfids = -1;
//   for (int i = 0; i < sm.GetNE(); i++)
//   {
//     int peid = parent_element_ids[i];
//     Array<int> sel_faces, pel_faces, o;
//     sm.GetElementEdges(i, sel_faces, o);
//     pm.GetElementEdges(peid, pel_faces, o);

//     MFEM_ASSERT(sel_faces.Size() == pel_faces.Size(), "internal error");
//     for (int j = 0; j < sel_faces.Size(); j++)
//     {
//         if (pfids[sel_faces[j]] != -1)
//         {
//           MFEM_ASSERT(pfids[sel_faces[j]] == pel_faces[j], "internal error");
//         }
//         pfids[sel_faces[j]] = pel_faces[j];
//     }
//   }
//   return pfids;
// }

// void MultiBlockSolver::BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int>* parent_face_map)
// {
//    MFEM_ASSERT(pm.Dimension() == 2, "Support only 2-dimension meshes!");
//    MFEM_ASSERT(sm.Dimension() == 2, "Support only 2-dimension meshes!");

//    // Array<int> parent_face_map = submesh.GetParentFaceIDMap();
//    if (parent_face_map == NULL)
//       parent_face_map = new Array<int>(BuildFaceMap2D(pm, sm));

//    // Setting boundary element attribute of submesh for 2D.
//    // This does not support 2D.
//    // Array<int> parent_face_to_be = mesh.GetFaceToBdrElMap();
//    Array<int> parent_face_to_be(pm.GetNumFaces());
//    parent_face_to_be = -1;
//    for (int i = 0; i < pm.GetNBE(); i++)
//    {
//       parent_face_to_be[pm.GetBdrElementEdgeIndex(i)] = i;
//    }
//    for (int k = 0; k < sm.GetNBE(); k++)
//    {
//       int pbeid = parent_face_to_be[(*parent_face_map)[sm.GetBdrFace(k)]];
//       if (pbeid != -1)
//       {
//          int attr = pm.GetBdrElement(pbeid)->GetAttribute();
//          sm.GetBdrElement(k)->SetAttribute(attr);
//       }
//       else
//       {
//          // This case happens when a domain is extracted, but the root parent
//          // mesh didn't have a boundary element on the surface that defined
//          // it's boundary. It still creates a valid mesh, so we allow it.
//          sm.GetBdrElement(k)->SetAttribute(SubMesh::GENERATED_ATTRIBUTE);
//       }
//    }

//    UpdateBdrAttributes(sm);
// }

// void MultiBlockSolver::UpdateBdrAttributes(Mesh& m)
// {
//    m.bdr_attributes.DeleteAll();
//    for (int k = 0; k < m.GetNBE(); k++)
//    {
//       int attr = m.GetBdrAttribute(k);
//       int inBdrAttr = m.bdr_attributes.Find(attr);
//       if (inBdrAttr < 0) m.bdr_attributes.Append(attr);
//    }
// }

// void MultiBlockSolver::BuildInterfaceInfos()
// {
//    Array2D<int> interface_attributes(numSub, numSub);
//    interface_attributes = -1;
//    interface_infos.SetSize(0);
//    // interface_parent.SetSize(0);

//    // interface attribute starts after the parent mesh boundary attributes.
//    int if_attr = pmesh->bdr_attributes.Max() + 1;

//    for (int i = 0; i < numSub; i++)
//    {
//       for (int ib = 0; ib < meshes[i]->GetNBE(); ib++)
//       {
//          if (meshes[i]->GetBdrAttribute(ib) != SubMesh::GENERATED_ATTRIBUTE) continue;
//          int parent_face_i = (*parent_face_map[i])[meshes[i]->GetBdrFace(ib)];

//          // Loop over each subdomain, each boundary element, to find the match.
//          for (int j = i+1; j < numSub; j++)
//          {
//             for (int jb = 0; jb < meshes[j]->GetNBE(); jb++)
//             {
//                int parent_face_j = (*parent_face_map[j])[meshes[j]->GetBdrFace(jb)];
//                if (parent_face_i != parent_face_j) continue;

//                MFEM_ASSERT(meshes[j]->GetBdrAttribute(jb) == SubMesh::GENERATED_ATTRIBUTE,
//                            "This interface element has been already set!");
//                if (interface_attributes[i][j] <= 0) {
//                   interface_attributes[i][j] = if_attr;
//                   if_attr += 1;
//                }

//                Array<int> Infs = FindParentInterfaceInfo(parent_face_i, i, ib, j, jb);

//                meshes[i]->SetBdrAttribute(ib, interface_attributes[i][j]);
//                meshes[j]->SetBdrAttribute(jb, interface_attributes[i][j]);

//                // submesh usually can inherit multiple attributes from parent.
//                // we limit to single-attribute case where attribute = index + 1;
//                interface_infos.Append(InterfaceInfo({.Attr = interface_attributes[i][j],
//                                                    .Mesh1 = i, .Mesh2 = j,
//                                                    .BE1 = ib, .BE2 = jb,
//                                                    .Inf1 = Infs[0], .Inf2 = Infs[1]}));
//                // interface_parent.Append(parent_face_i);
//             }
//          }
//       }
//    }

//    for (int i = 0; i < numSub; i++) UpdateBdrAttributes(*meshes[i]);
// }

// Array<int> MultiBlockSolver::FindParentInterfaceInfo(const int pface,
//                                                      const int imesh, const int ibe,
//                                                      const int jmesh, const int jbe)
// {
//    Array<int> Infs(2);
//    Mesh::FaceInformation face_info = pmesh->GetFaceInformation(pface);
                  
//    int face_inf[2];
//    pmesh->GetFaceInfos(pface, &face_inf[0], &face_inf[1]);
//    int eli, eli_info;
//    meshes[imesh]->GetBdrElementAdjacentElement(ibe, eli, eli_info);
//    eli = (*parent_elem_map[imesh])[eli];
//    int elj, elj_info;
//    meshes[jmesh]->GetBdrElementAdjacentElement(jbe, elj, elj_info);
//    elj = (*parent_elem_map[jmesh])[elj];

//    if (eli == face_info.element[0].index) {
//       Infs[0] = face_inf[0];
//       Infs[1] = face_inf[1];
//    } else {
//       Infs[0] = face_inf[1];
//       Infs[1] = face_inf[0];
//    }

//    return Infs;
// }

void MultiBlockSolver::SetupBCVariables()
{
   int numBdr = global_bdr_attributes.Size();
   MFEM_ASSERT(numBdr == bdr_coeffs_in.Size(), "MultiBlockSolver::SetupBoundaryConditions\n");

   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;

   // Boundary conditions are weakly constrained.
   ess_attrs.SetSize(numSub);
   ess_tdof_lists.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      ess_attrs[m] = new Array<int>(meshes[m]->bdr_attributes.Max());
      if (strong_bc) (*ess_attrs[m]) = 0;

      ess_tdof_lists[m] = new Array<int>;
      fes[m]->GetEssentialTrueDofs((*ess_attrs[m]), (*ess_tdof_lists[m]));
   }

   // Set up boundary markers.
   int max_bdr_attr = -1;
   for (int m = 0; m < numSub; m++)
   {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
   }

   bdr_markers.SetSize(max_bdr_attr);
   for (int k = 0; k < max_bdr_attr; k++) {
      bdr_markers[k] = new Array<int>(max_bdr_attr);
      (*bdr_markers[k]) = 0;
      (*bdr_markers[k])[k] = 1;
   }
}

void MultiBlockSolver::AddBCFunction(std::function<double(const Vector &)> F, const int battr)
{
   MFEM_ASSERT(bdr_coeffs.Size() > 0, "MultiBlockSolver::AddBCFunction\n");

   int idx = (battr > 0) ? battr - 1 : 0;
   bdr_coeffs[idx] = new FunctionCoefficient(F);

   if (battr < 0)
      for (int k = 1; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new FunctionCoefficient(F);
}

void MultiBlockSolver::AddBCFunction(const double &F, const int battr)
{
   MFEM_ASSERT(bdr_coeffs.Size() > 0, "MultiBlockSolver::AddBCFunction\n");

   int idx = (battr > 0) ? battr - 1 : 0;
   bdr_coeffs[idx] = new ConstantCoefficient(F);

   if (battr < 0)
      for (int k = 1; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new ConstantCoefficient(F);
}

void MultiBlockSolver::InitVariables()
{
   // number of blocks = solution dimension * number of subdomain;
   block_offsets.SetSize(udim * numSub + 1);
   domain_offsets.SetSize(numSub + 1);
   num_vdofs.SetSize(numSub);
   block_offsets[0] = 0;
   domain_offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
   {
      domain_offsets[i + 1] = fes[i]->GetTrueVSize();
      num_vdofs[i] = fes[i]->GetTrueVSize();
      for (int d = 0; d < udim; d++)
      {
         block_offsets[d + i * udim + 1] = fes[i]->GetNDofs();
      }
   }
   block_offsets.PartialSum();
   domain_offsets.PartialSum();

   SetupBCVariables();

   // Set up solution/rhs variables/
   U = new BlockVector(domain_offsets);
   RHS = new BlockVector(domain_offsets);
   /* 
      Note: for compatibility with ROM, it's better to split with domain_offsets.
      For vector-component operations, can set up a view BlockVector like below:

         BlockVector *U_blocks = new BlockVector(U->GetData(), block_offsets);

      U_blocks does not own the data.
      These are system-specific, therefore not defining it now.
   */

   us.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      us[m] = new GridFunction(fes[m], U->GetBlock(m), 0);
      (*us[m]) = 0.0;

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      us[m]->SetTrueVector();
   }

   rhs_coeffs.SetSize(0);

   if (use_rom) InitROMHandler();
}

void MultiBlockSolver::BuildOperators()
{
   SanityCheckOnCoeffs();

   bs.SetSize(numSub);
   as.SetSize(numSub);

   double sigma = -1.0;
   double kappa = (order + 1.0) * (order + 1.0);

   // These are heavily system-dependent.
   // Based on scalar/vector system, different integrators/coefficients will be used.
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m).GetData());
      for (int r = 0; r < rhs_coeffs.Size(); r++)
         bs[m]->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeffs[r]));

      as[m] = new BilinearForm(fes[m]);
      as[m]->AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         as[m]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   }

   interface_integ = new InterfaceDGDiffusionIntegrator(sigma, kappa);
}

void MultiBlockSolver::SetupBCOperators()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m] && bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (bdr_coeffs[b] == NULL) continue;

         bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdr_coeffs[b], sigma, kappa), *bdr_markers[b]);
         as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), *bdr_markers[b]);
      }
   }
}

void MultiBlockSolver::Assemble()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m] && bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");

      bs[m]->Assemble();
      as[m]->Assemble();
   }

   mats.SetSize(numSub, numSub);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i == j) {
            mats(i, i) = &(as[i]->SpMat());
         } else {
            mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());
         }
      }
   }
   AssembleInterfaceMatrix();

   for (int m = 0; m < numSub; m++)
   {
      // Do we really need SyncAliasMemory?
      bs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
      as[m]->Finalize();
   }

   // globalMat = new BlockOperator(block_offsets);
   // NOTE: currently, domain-decomposed system will have a significantly different sparsity pattern.
   // This is especially true for vector solution, where ordering of component is changed.
   // This is quite inevitable, but is it desirable?
   globalMat = new BlockMatrix(domain_offsets);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i != j) mats(i, j)->Finalize();

         globalMat->SetBlock(i, j, mats(i, j));
      }
   }

   if (use_monolithic)
      globalMat_mono = globalMat->CreateMonolithic();
}

void MultiBlockSolver::AssembleInterfaceMatrix()
{
   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);
      Mesh *mesh1, *mesh2;
      FiniteElementSpace *fes1, *fes2;
      DenseMatrix elemmat;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *fe1, *fe2;
      Array<Array<int> *> vdofs(2);
      vdofs[0] = new Array<int>;
      vdofs[1] = new Array<int>;

      Array<int> midx(2);
      midx[0] = if_info->Mesh1;
      midx[1] = if_info->Mesh2;

      mesh1 = &(*meshes[midx[0]]);
      mesh2 = &(*meshes[midx[1]]);
      fes1 = fes[midx[0]];
      fes2 = fes[midx[1]];

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
         fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         interface_integ->AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmat);

         DenseMatrix subelemmat;
         int ndof1 = fe1->GetDof();
         int ndof2 = fe2->GetDof();

         // TODO: we do not need to take these additional steps to split elemmat.
         // Need to assemble them directly from AssembleInterfaceMatrix.
         Array<int> block_offsets(3);
         block_offsets[0] = 0;
         block_offsets[1] = fe1->GetDof();
         block_offsets[2] = fe2->GetDof();
         block_offsets.PartialSum();
         for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
               elemmat.GetSubMatrix(block_offsets[i], block_offsets[i+1],
                                    block_offsets[j], block_offsets[j+1], subelemmat);
               mats(midx[i], midx[j])->AddSubMatrix(*vdofs[i], *vdofs[j], subelemmat, skip_zeros);
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
    }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}

// void MultiBlockSolver::GetInterfaceTransformations(Mesh *m1, Mesh *m2, const InterfaceInfo *if_info,
//                                                    FaceElementTransformations* &tr1, FaceElementTransformations* &tr2)
// {
//    // We cannot write a function that replaces this, since only Mesh can access to FaceElemTr.SetConfigurationMask.
//    tr1 = m1->GetBdrFaceTransformations(if_info->BE1);
//    tr2 = m2->GetBdrFaceTransformations(if_info->BE2);

//    // Correcting the local face1 transformation if orientation needs correction.
//    int faceInf1, faceInf2;
//    int face1 = m1->GetBdrFace(if_info->BE1);
//    m1->GetFaceInfos(face1, &faceInf1, &faceInf2);
//    if (faceInf1 != if_info->Inf1)
//    {
//       if ((faceInf1 / 64) != (if_info->Inf1 / 64))
//       {
//          MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
//       }

//       int face_type = m1->GetFaceElementType(face1);
//       int elem_type = m1->GetElementType(tr1->Elem1No);

//       m1->GetLocalFaceTransformation(face_type, elem_type,
//                                     tr1->Loc1.Transf, if_info->Inf1);
//    }

//    // Correcting the local face1 transformation if orientation needs correction.
//    int face2 = m2->GetBdrFace(if_info->BE2);
//    m2->GetFaceInfos(face2, &faceInf2, &faceInf1);
//    if (faceInf2 != if_info->Inf2)
//    {
//       if ((faceInf2 / 64) != (if_info->Inf2 / 64))
//       {
//          MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
//       }

//       int face_type = m2->GetFaceElementType(face2);
//       int elem_type = m2->GetElementType(tr2->Elem1No);

//       m2->GetLocalFaceTransformation(face_type, elem_type,
//                                     tr2->Loc1.Transf, if_info->Inf2);
//    }
// }

void MultiBlockSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 1000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-6);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-10);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   // Until now, did not find a meaningful speed difference between block matrix and sparse matrix.
   if (use_monolithic)
   {
      assert(globalMat_mono != NULL);
      solver.SetOperator(*globalMat_mono);
   }
   else
   {
      solver.SetOperator(*globalMat);
   }
   solver.SetPrintLevel(print_level);

   BlockDiagonalPreconditioner *globalPrec;
   if (config.GetOption<bool>("solver/block_diagonal_preconditioner", true))
   {
      globalPrec = new BlockDiagonalPreconditioner(domain_offsets);
      solver.SetPreconditioner(*globalPrec);
   }

   *U = 0.0;
   // The time for the setup above is much smaller than this Mult().
   // StopWatch test;
   // test.Start();
   solver.Mult(*RHS, *U);
   // test.Stop();
   // printf("test: %f seconds.\n", test.RealTime());
}

void MultiBlockSolver::InitVisualization(const std::string& output_path)
{
   if (!save_visual) return;

   std::string file_prefix;
   if (output_path != "")
      file_prefix = output_path;
   else
   {
      assert(visual_prefix != "");
      assert(visual_dir != "");
      file_prefix = visual_dir + "/" + visual_prefix;
   }

   unified_paraview = config.GetOption<bool>("visualization/unified_paraview", true);

   if (unified_paraview)
      InitUnifiedParaview(file_prefix);
   else
      InitIndividualParaview(file_prefix);
}

void MultiBlockSolver::InitIndividualParaview(const std::string& file_prefix)
{
   paraviewColls.SetSize(numSub);

   for (int m = 0; m < numSub; m++) {
      ostringstream oss;
      // Each subdomain needs to be save separately.
      oss << file_prefix << "_" << std::to_string(m);

      paraviewColls[m] = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
      paraviewColls[m]->SetLevelsOfDetail(order);
      paraviewColls[m]->SetHighOrderOutput(true);
      paraviewColls[m]->SetPrecision(8);

      paraviewColls[m]->RegisterField("solution", us[m]);
      paraviewColls[m]->SetOwnData(false);
   }
}

void MultiBlockSolver::InitUnifiedParaview(const std::string& file_prefix)
{
   // TODO: For truly bottom-up case, when the parent mesh does not exist?
   mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
   paraviewColls.SetSize(1);

   // grid function initialization for visual.
   // TODO: for vector solution.
   Mesh *pmesh = topol_handler->GetGlobalMesh();
   global_fes = new FiniteElementSpace(pmesh, fec, udim);
   global_us_visual = new GridFunction(global_fes);

   paraviewColls[0] = new ParaViewDataCollection(file_prefix.c_str(), pmesh);
   paraviewColls[0]->SetLevelsOfDetail(order);
   paraviewColls[0]->SetHighOrderOutput(true);
   paraviewColls[0]->SetPrecision(8);

   paraviewColls[0]->RegisterField("solution", global_us_visual);
   paraviewColls[0]->SetOwnData(false);
}

void MultiBlockSolver::SaveVisualization()
{
   if (!save_visual) return;

   if (unified_paraview)
   {
      mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
      topol_handler->TransferToGlobal(us, global_us_visual);
      // // TODO: For truly bottom-up case, when submesh is not used?
      // for (int m = 0; m < numSub; m++)
      //    meshes[m]->Transfer(*us[m], *global_us_visual);
   }

   for (int m = 0; m < paraviewColls.Size(); m++)
      paraviewColls[m]->Save();
};

void MultiBlockSolver::InitROMHandler()
{
   std::string rom_handler_str = config.GetOption<std::string>("model_reduction/rom_handler_type", "base");

   if (rom_handler_str == "base")
   {
      rom_handler = new ROMHandler(numSub, udim, num_vdofs);
   }
   else if (rom_handler_str == "mfem")
   {
      rom_handler = new MFEMROMHandler(numSub, udim, num_vdofs);
   }
   else
   {
      mfem_error("Unknown ROM handler type!\n");
   }
}

double MultiBlockSolver::CompareSolution()
{
   // Copy the rom solution.
   BlockVector romU(domain_offsets);
   romU = *U;
   Array<GridFunction *> rom_us;
   Array<VectorGridFunctionCoefficient *> rom_u_coeffs;
   ConstantCoefficient zero(0.0);
   rom_us.SetSize(numSub);
   rom_u_coeffs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      rom_us[m] = new GridFunction(fes[m], romU.GetBlock(m), 0);

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      rom_us[m]->SetTrueVector();

      rom_u_coeffs[m] = new VectorGridFunctionCoefficient(rom_us[m]);
   }

   // TODO: right now we solve the full-order system to compare the solution.
   // Need to implement loading the fom solution file?
   StopWatch solveTimer;
   solveTimer.Start();
   Solve();
   solveTimer.Stop();
   printf("FOM-solve time: %f seconds.\n", solveTimer.RealTime());

   // Compare the solution.
   double norm = 0.0;
   double error = 0.0;
   for (int m = 0; m < numSub; m++)
   {
      norm += us[m]->ComputeLpError(2, zero);
      error += us[m]->ComputeLpError(2, *rom_u_coeffs[m]);
   }
   error /= norm;
   printf("Relative error: %.5E\n", error);

   for (int m = 0; m < numSub; m++)
   {
      delete rom_us[m];
      delete rom_u_coeffs[m];
   }

   return error;
}

void MultiBlockSolver::SanityCheckOnCoeffs()
{
   if (rhs_coeffs.Size() == 0)
      MFEM_WARNING("There is no right-hand side coeffcient assigned! Make sure to set rhs coefficients before BuildOperator.\n");

   if (bdr_coeffs.Size() == 0)
      MFEM_WARNING("There is no bc coeffcient assigned! Make sure to set bc coefficients before SetupBCOperator.\n");

   bool all_null = true;
   for (int i = 0; i < rhs_coeffs.Size(); i++)
      if (rhs_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All rhs coefficents are NULL! Make sure to set rhs coefficients before BuildOperator.\n");

   all_null = false;
   for (int i = 0; i < bdr_coeffs.Size(); i++)
      if (bdr_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All bc coefficients are NULL, meaning there is no Dirichlet BC. Make sure to set bc coefficients before SetupBCOperator.\n");
}
