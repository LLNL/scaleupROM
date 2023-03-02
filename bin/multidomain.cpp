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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
// #include "multiblock_nonlinearform.hpp"

using namespace mfem;

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
   const char *mesh_file = "../data/star.mesh";

   int order = 1;
   bool verbose = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&verbose, "-v", "--verbose", "-nv", "--no-verbose", "Print out details.");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh mesh(mesh_file);

   printf("mesh nbe: %d\n", mesh.GetNBE());
   for (int k = 0; k < mesh.bdr_attributes.Size(); k++) {
     printf("bdr attribute %d: %d\n", k, mesh.bdr_attributes[k]);
   }
   int nfaces = mesh.GetNumFaces();
   printf("mesh nfaces: %d\n", nfaces);

   // parent_mesh.UniformRefinement();

   H1_FECollection fec(order, mesh.Dimension());

   // Create the sub-domains and accompanying Finite Element spaces from
   // corresponding attributes.
   int numSub = mesh.attributes.Max();
   std::vector<std::shared_ptr<SubMesh>> submeshes(numSub);
   for (int k = 0; k < numSub; k++) {
      Array<int> domain_attributes(1);
      domain_attributes[0] = k+1;

      submeshes[k] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(mesh, domain_attributes));
   }

   // NOTE: currently submesh does not generate face map for 2d mesh..
   Array<Array<int> *> parent_face_map_2d(numSub);
   for (int k = 0; k < numSub; k++) {
      parent_face_map_2d[k] = new Array<int>(BuildFaceMap2D(mesh, *submeshes[k]));
      BuildSubMeshBoundary2D(mesh, *submeshes[k], parent_face_map_2d[k]);
   }

   Array<Array<int> *> parent_el_map(numSub);
   for (int k = 0; k < numSub; k++) {
      parent_el_map[k] = new Array<int>(submeshes[k]->GetParentElementIDMap());
   }

   for (int i = 0; i < numSub; i++) {
      printf("Submesh %d\n", i);
      for (int k = 0; k < submeshes[i]->GetNBE(); k++) {
         printf("bdr element %d attribute: %d\n", k, submeshes[i]->GetBdrAttribute(k));
      }

      // Setting a new boundary attribute does not append bdr_attributes.
      printf("submesh nbe: %d\n", submeshes[i]->GetNBE());
      for (int k = 0; k < submeshes[i]->bdr_attributes.Size(); k++) {
         printf("bdr attribute %d: %d\n", k, submeshes[i]->bdr_attributes[k]);
      }

      int nfaces = submeshes[i]->GetNumFaces();
      printf("submesh nfaces: %d\n", nfaces);
   }

   Array2D<int> interface_attributes(numSub, numSub);
   interface_attributes = -1;
   // interface attribute starts after the parent mesh boundary attributes.
   int if_attr = mesh.bdr_attributes.Max() + 1;
   Array<InterfaceInfo> interface_infos(0);
   
   for (int i = 0; i < numSub; i++) {
      // printf("Submesh %d\n", i);
      for (int ib = 0; ib < submeshes[i]->GetNBE(); ib++) {
         if (submeshes[i]->GetBdrAttribute(ib) != SubMesh::GENERATED_ATTRIBUTE) continue;

         int parent_face_i = (*parent_face_map_2d[i])[submeshes[i]->GetBdrFace(ib)];
         for (int j = i+1; j < numSub; j++) {
            for (int jb = 0; jb < submeshes[j]->GetNBE(); jb++) {
               int parent_face_j = (*parent_face_map_2d[j])[submeshes[j]->GetBdrFace(jb)];
               if (parent_face_i == parent_face_j) {
                  MFEM_ASSERT(submeshes[j]->GetBdrAttribute(jb) == SubMesh::GENERATED_ATTRIBUTE,
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
                     submeshes[i]->GetBdrElementAdjacentElement(ib, eli, eli_info);
                     eli = (*parent_el_map[i])[eli];
                     int elj, elj_info;
                     submeshes[j]->GetBdrElementAdjacentElement(jb, elj, elj_info);
                     elj = (*parent_el_map[j])[elj];

                     if (eli == face_info.element[0].index) {
                        Infs[0] = face_inf[0];
                        Infs[1] = face_inf[1];
                     } else {
                        Infs[0] = face_inf[1];
                        Infs[1] = face_inf[0];
                     }

                     printf("pf %d\tel1\tel2\n", parent_face_i);
                     printf("pm\t%d\t%d\n", face_info.element[0].index, face_info.element[1].index);
                     printf("sm\t%d\t%d\n", eli, elj);
                     printf("\n");
                     printf("el1\tid\tori\n");
                     printf("pm\t%d\t%d\n", face_info.element[0].local_face_id, face_info.element[0].orientation);
                     printf("sm\t%d\t%d\n", eli_info / 64, eli_info % 64);
                     printf("\n");
                     printf("el2\tid\tori\n");
                     printf("pm\t%d\t%d\n", face_info.element[1].local_face_id, face_info.element[1].orientation);
                     printf("sm\t%d\t%d\n", elj_info / 64, elj_info % 64);
                     printf("\n");
                  }

                  submeshes[i]->SetBdrAttribute(ib, interface_attributes[i][j]);
                  submeshes[j]->SetBdrAttribute(jb, interface_attributes[i][j]);

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

   for (int i = 0; i < numSub; i++) UpdateBdrAttributes(*submeshes[i]);

   for (int i = 0; i < numSub; i++) {
      printf("Submesh %d\n", i);
      for (int ib = 0; ib < submeshes[i]->GetNBE(); ib++) {
         int interface_attr = submeshes[i]->GetBdrAttribute(ib);
         if (interface_attr <= mesh.bdr_attributes.Max()) continue;

         int parent_face_i = (*parent_face_map_2d[i])[submeshes[i]->GetBdrFace(ib)];
         
         for (int j = 0; j < numSub; j++) {
            if (i == j) continue;
            for (int jb = 0; jb < submeshes[j]->GetNBE(); jb++) {
               int parent_face_j = (*parent_face_map_2d[j])[submeshes[j]->GetBdrFace(jb)];
               if (parent_face_i == parent_face_j) {
                  printf("(BE %d, face %d) - parent face %d, attr %d - Submesh %d (BE %d, face %d)\n",
                         ib, submeshes[i]->GetBdrFace(ib), parent_face_i, interface_attr, j, jb, submeshes[j]->GetBdrFace(jb));
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
   
   // for (int i = 0; i < nfaces; i++)
   // {
      // int face_inf[2], face_idx[2];
      // submesh.GetFaceInfos(i, &face_inf[0], &face_inf[1]);
      // submesh.GetFaceElements(i, &face_idx[0], &face_idx[1]);
   
      // Mesh::FaceInformation face_info = submesh.GetFaceInformation(i);
      // for (int j = 0; j < 2; j++) {
      //    printf("Face %d Element %d information\n", i, j);
      //    printf("Index: %d =? %d\n", face_info.element[j].index, face_idx[j]);
      //    printf("Local Face ID: %d =? %d\n", face_info.element[j].local_face_id, face_inf[j] / 64);
      //    printf("Orientation: %d =? %d\n", face_info.element[j].orientation, face_inf[j] % 64);
      // }
   // }

   return 0;
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

   // Setting boundary element attribute of submesh for 2D.
   // This does not support 2D.
   // Array<int> parent_face_to_be = mesh.GetFaceToBdrElMap();
   Array<int> parent_face_to_be(pm.GetNumFaces());
   parent_face_to_be = -1;
   for (int i = 0; i < pm.GetNBE(); i++)
   {
      parent_face_to_be[pm.GetBdrElementEdgeIndex(i)] = i;
   }
   for (int k = 0; k < sm.GetNBE(); k++) {
      int pbeid = parent_face_to_be[(*parent_face_map)[sm.GetBdrFace(k)]];
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
         sm.GetBdrElement(k)->SetAttribute(SubMesh::GENERATED_ATTRIBUTE);
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