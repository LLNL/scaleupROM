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

#include "topology_handler.hpp"

using namespace std;
using namespace mfem;

/*
   TopologyHandler Base class
*/

TopologyHandler::TopologyHandler(const TopologyHandlerMode &input_type)
   : type(input_type)
{
   std::string dd_mode_str = config.GetOption<std::string>("domain-decomposition/type", "interior_penalty");
   if (dd_mode_str == "interior_penalty")
   {
      dd_mode = DecompositionMode::IP;
   }
   else if (dd_mode_str == "feti")
   {
      mfem_error("FETI not implemented!\n");
   }
   else if (dd_mode_str == "none")
   {
      dd_mode = DecompositionMode::NODD;
   }
   else
   {
      mfem_error("Unknown domain decomposition mode!\n");
   }
}

void TopologyHandler::GetInterfaceTransformations(Mesh *m1, Mesh *m2, const InterfaceInfo *if_info,
                                                   FaceElementTransformations* &tr1,
                                                   FaceElementTransformations* &tr2)
{
   // We cannot write a function that replaces this, since only Mesh can access to FaceElemTr.SetConfigurationMask.
   tr1 = m1->GetBdrFaceTransformations(if_info->BE1);
   tr2 = m2->GetBdrFaceTransformations(if_info->BE2);

   // Correcting the local face1 transformation if orientation needs correction.
   int faceInf1, faceInf2;
   int face1 = m1->GetBdrFace(if_info->BE1);
   m1->GetFaceInfos(face1, &faceInf1, &faceInf2);
   if (faceInf1 != if_info->Inf1)
   {
      if ((faceInf1 / 64) != (if_info->Inf1 / 64))
      {
         MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
      }

      int face_type = m1->GetFaceElementType(face1);
      int elem_type = m1->GetElementType(tr1->Elem1No);

      m1->GetLocalFaceTransformation(face_type, elem_type,
                                    tr1->Loc1.Transf, if_info->Inf1);
   }

   // Correcting the local face1 transformation if orientation needs correction.
   int face2 = m2->GetBdrFace(if_info->BE2);
   m2->GetFaceInfos(face2, &faceInf2, &faceInf1);
   if (faceInf2 != if_info->Inf2)
   {
      if ((faceInf2 / 64) != (if_info->Inf2 / 64))
      {
         MFEM_WARNING("Local face id from submesh and global mesh are different. This may cause inaccurate solutions.");
      }

      int face_type = m2->GetFaceElementType(face2);
      int elem_type = m2->GetElementType(tr2->Elem1No);

      m2->GetLocalFaceTransformation(face_type, elem_type,
                                    tr2->Loc1.Transf, if_info->Inf2);
   }
}

void TopologyHandler::UpdateAttributes(Mesh& m)
{
   m.attributes.DeleteAll();
   for (int k = 0; k < m.GetNE(); k++)
   {
      int attr = m.GetAttribute(k);
      int inBdrAttr = m.attributes.Find(attr);
      if (inBdrAttr < 0) m.attributes.Append(attr);
   }
}

void TopologyHandler::UpdateBdrAttributes(Mesh& m)
{
   m.bdr_attributes.DeleteAll();
   for (int k = 0; k < m.GetNBE(); k++)
   {
      int attr = m.GetBdrAttribute(k);
      int inBdrAttr = m.bdr_attributes.Find(attr);
      if (inBdrAttr < 0) m.bdr_attributes.Append(attr);
   }
}

void TopologyHandler::PrintPortInfo(const int k)
{
   int start_idx = k, end_idx = k+1;
   if (k < 0)
   {
      start_idx = 0;
      end_idx = port_infos.Size();
   }

   printf("Port\tAttr\tMesh1\tMesh2\tAttr1\tAttr2\n");
   for (int i = start_idx; i < end_idx; i++)
   {
      PortInfo *port = &(port_infos[i]);
      printf("%d\t%d\t%d\t%d\t%d\t%d\n", i, port->PortAttr,
            port->Mesh1, port->Mesh2, port->Attr1, port->Attr2);
   }
}

void TopologyHandler::PrintInterfaceInfo(const int k)
{
   int start_idx = k, end_idx = k+1;
   if (k < 0)
   {
      start_idx = 0;
      end_idx = port_infos.Size();
   }

   for (int i = start_idx; i < end_idx; i++)
   {
      Array<InterfaceInfo> *info = interface_infos[i];

      printf("Port %d interface info.\n", i);
      printf("Idx\tBE1\tBE2\tLF1\tLF2\tOri1\tOri2\n");
      for (int j = 0; j < info->Size(); j++)
      {
         InterfaceInfo *info_j = &((*info)[j]);
         printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", j,
               info_j->BE1, info_j->BE2,
               info_j->Inf1 / 64, info_j->Inf2 / 64,
               info_j->Inf1 % 64, info_j->Inf2 % 64);
      }
   }
}

/*
   SubMeshTopologyHandler
*/

SubMeshTopologyHandler::SubMeshTopologyHandler(Mesh* pmesh_)
   : TopologyHandler(SUBMESH),
     pmesh(pmesh_),
     own_pmesh(false)
{
   // Input meshes may not have up-to-date attributes array.
   UpdateAttributes(*pmesh);
   dim = pmesh->Dimension();

   // Uniform refinement if specified.
   int num_refinement = config.GetOption<int>("mesh/uniform_refinement", 0);
   for (int k = 0; k < num_refinement; k++)
      pmesh->UniformRefinement();

   // Initiate SubMeshes based on attributes.
   // TODO: a sanity check?
   switch (dd_mode)
   {
      case DecompositionMode::NODD:
      {
         numSub = 1;
         meshes.resize(numSub);
         Array<int> domain_attributes(pmesh->attributes.Max());
         for (int k = 0; k < pmesh->attributes.Max(); k++) {
            domain_attributes[k] = k+1;
         }
         meshes[0] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(*pmesh, domain_attributes));
         break;
      }
      default:
      {
         numSub = pmesh->attributes.Max();
         meshes.resize(numSub);
         for (int k = 0; k < numSub; k++) {
            Array<int> domain_attributes(1);
            domain_attributes[0] = k+1;

            meshes[k] = std::make_shared<SubMesh>(SubMesh::CreateFromDomain(*pmesh, domain_attributes));
         }
         break;
      }
   }

   // for SubMeshTopologyHandler, only single-component is allowed.
   num_comp = 1;
   mesh_types.SetSize(numSub);
   mesh_types = 0;
   sub_composition.SetSize(num_comp);
   sub_composition = numSub;
   mesh_comp_idx.SetSize(numSub);
   for (int m = 0; m < numSub; m++) mesh_comp_idx[m] = m;

   // Set up element mapping between submeshes and parent mesh.
   parent_elem_map.SetSize(numSub);
   for (int k = 0; k < numSub; k++) {
      parent_elem_map[k] = new Array<int>(meshes[k]->GetParentElementIDMap());
   }

   // Set up face mapping between submeshes and parent mesh.
   parent_face_map.SetSize(numSub);
   for (int k = 0; k < numSub; k++) {
      if (dim == 2)
      {
         parent_face_map[k] = new Array<int>(BuildFaceMap2D(*pmesh, *meshes[k]));
         BuildSubMeshBoundary2D(*pmesh, *meshes[k], parent_face_map[k]);
      }
      else
      {
         parent_face_map[k] = new Array<int>(meshes[k]->GetParentFaceIDMap());
      }
   }

   BuildInterfaceInfos();
}

SubMeshTopologyHandler::SubMeshTopologyHandler()
   : SubMeshTopologyHandler(new Mesh(config.GetRequiredOption<std::string>("mesh/filename").c_str()))
{
   // Do not use *this = SubMeshTopologyHandler(...), unless you define operator=!
   own_pmesh = true;
}

void SubMeshTopologyHandler::ExportInfo(Array<Mesh*> &mesh_ptrs,
                                       // Array<InterfaceInfo>* &if_infos,
                                       TopologyData &topol_data)
{
   mesh_ptrs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
      mesh_ptrs[m] = &(*meshes[m]);

   topol_data.dim = dim;
   topol_data.numSub = numSub;
   topol_data.global_bdr_attributes = &pmesh->bdr_attributes;
}

SubMeshTopologyHandler::~SubMeshTopologyHandler()
{
   for (int k = 0; k < parent_elem_map.Size(); k++) delete parent_elem_map[k];
   for (int k = 0; k < parent_face_map.Size(); k++) delete parent_face_map[k];
   if (own_pmesh)
      delete pmesh;
}

Array<int> SubMeshTopologyHandler::BuildFaceMap2D(const Mesh& pm, const SubMesh& sm)
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

void SubMeshTopologyHandler::BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int>* parent_face_map)
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
   for (int k = 0; k < sm.GetNBE(); k++)
   {
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

void SubMeshTopologyHandler::BuildInterfaceInfos()
{
   Array2D<int> interface_attributes(numSub, numSub);
   interface_attributes = -1;
   Array2D<int> interface_map(numSub, numSub);
   interface_attributes = -1;

   interface_infos.SetSize(0);
   port_infos.SetSize(0);

   // interface attribute starts after the parent mesh boundary attributes.
   int if_attr = pmesh->bdr_attributes.Max() + 1;

   for (int i = 0; i < numSub; i++)
   {
      for (int ib = 0; ib < meshes[i]->GetNBE(); ib++)
      {
         if (meshes[i]->GetBdrAttribute(ib) != SubMesh::GENERATED_ATTRIBUTE) continue;
         int parent_face_i = (*parent_face_map[i])[meshes[i]->GetBdrFace(ib)];

         // Loop over each subdomain, each boundary element, to find the match.
         for (int j = i+1; j < numSub; j++)
         {
            for (int jb = 0; jb < meshes[j]->GetNBE(); jb++)
            {
               int parent_face_j = (*parent_face_map[j])[meshes[j]->GetBdrFace(jb)];
               if (parent_face_i != parent_face_j) continue;

               assert(meshes[j]->GetBdrAttribute(jb) == SubMesh::GENERATED_ATTRIBUTE);

               if (interface_attributes[i][j] <= 0) {
                  interface_attributes[i][j] = if_attr;
                  interface_map[i][j] = port_infos.Size();
                  // NOTE: for SubMehs, component boundary attribute is simply SubMesh::GENERATED_ATTRIBUTE,
                  // which is not informative at all.
                  port_infos.Append(PortInfo({.PortAttr = if_attr,
                                              .Mesh1 = i, .Mesh2 = j,
                                              .Attr1 = if_attr, .Attr2 = if_attr}));
                  interface_infos.Append(new Array<InterfaceInfo>(0));

                  if_attr += 1;
               }
               assert(interface_map[i][j] >= 0);

               Array<int> Infs = FindParentInterfaceInfo(parent_face_i, i, ib, j, jb);

               meshes[i]->SetBdrAttribute(ib, interface_attributes[i][j]);
               meshes[j]->SetBdrAttribute(jb, interface_attributes[i][j]);

               // submesh usually can inherit multiple attributes from parent.
               // we limit to single-attribute case where attribute = index + 1;
               interface_infos[interface_map[i][j]]->Append(
                  InterfaceInfo({.BE1 = ib, .BE2 = jb, .Inf1 = Infs[0], .Inf2 = Infs[1]}));
               // interface_parent.Append(parent_face_i);
            }  // for (int jb = 0; jb < meshes[j]->GetNBE(); jb++)
         }  // for (int j = i+1; j < numSub; j++)
      }  // for (int ib = 0; ib < meshes[i]->GetNBE(); ib++)
   }  // for (int i = 0; i < numSub; i++)

   num_ports = port_infos.Size();
   for (int i = 0; i < numSub; i++) UpdateBdrAttributes(*meshes[i]);
}

Array<int> SubMeshTopologyHandler::FindParentInterfaceInfo(const int pface,
                                                            const int imesh, const int ibe,
                                                            const int jmesh, const int jbe)
{
   Array<int> Infs(2);
   Mesh::FaceInformation face_info = pmesh->GetFaceInformation(pface);
                  
   int face_inf[2];
   pmesh->GetFaceInfos(pface, &face_inf[0], &face_inf[1]);
   int eli, eli_info;
   meshes[imesh]->GetBdrElementAdjacentElement(ibe, eli, eli_info);
   eli = (*parent_elem_map[imesh])[eli];
   int elj, elj_info;
   meshes[jmesh]->GetBdrElementAdjacentElement(jbe, elj, elj_info);
   elj = (*parent_elem_map[jmesh])[elj];

   if (eli == face_info.element[0].index) {
      Infs[0] = face_inf[0];
      Infs[1] = face_inf[1];
   } else {
      Infs[0] = face_inf[1];
      Infs[1] = face_inf[0];
   }

   return Infs;
}

void SubMeshTopologyHandler::TransferToGlobal(Array<GridFunction*> &us, GridFunction* &global_u)
{
   for (int m = 0; m < numSub; m++)
      meshes[m]->Transfer(*us[m], *global_u);
}
