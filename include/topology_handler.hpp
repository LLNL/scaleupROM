// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef TOPOLOGY_HANDLER_HPP
#define TOPOLOGY_HANDLER_HPP

#include "input_parser.hpp"
#include "mfem.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

enum DecompositionMode
{
   NODD,       // no decomposition
   IP,         // interior penalty
   FETI,       // finite-element tearing and interconnecting
   NUM_DDMODE
};

struct InterfaceInfo {
   int Attr;
   int Mesh1, Mesh2;
   int BE1, BE2;

   // Inf = 64 * LocalFaceIndex + FaceOrientation
   // From the parent mesh.
   // Boundary face only have Elem1, and its orientation is always 0 by convention.
   // This causes a problem for interface between two meshes.
   // Thus stores orientation information from the parent mesh.
   int Inf1, Inf2;
};

struct TopologyData {
   int numSub = -1;
   int dim = -1;
   Array<int> global_bdr_attributes;
};

class TopologyHandler
{
protected:
   int numSub;   // number of subdomains.

   // Spatial dimension.
   int dim;

   DecompositionMode dd_mode;

   Array<InterfaceInfo> interface_infos;

public:
   TopologyHandler() {};
   TopologyHandler(Array<Mesh*> &mesh_ptrs, Array<InterfaceInfo>* &if_infos, TopologyData &topol_data)
   { mfem_error("Abstract base constructor TopologyHandler is called!\n"); }

   virtual ~TopologyHandler() {};

   // access
   const int GetNumSubdomains() { return numSub; }
   virtual Mesh* GetMesh(const int k) = 0;
   virtual Mesh* GetGlobalMesh() = 0;

   // Mesh sets face element transformation based on the face_info.
   // For boundary face, the adjacent element is always on element 1, and its orientation is "by convention" always zero.
   // This is a problem for the interface between two meshes, where both element orientations are zero.
   // At least one element should reflect a relative orientation with respect to the other.
   // Currently this is done by hijacking global mesh face information in the beginning.
   // If we would want to do more flexible global mesh building, e.g. rotating component submeshes,
   // then we will need to figure out how to actually determine relative orientation.
   virtual void GetInterfaceTransformations(Mesh *m1, Mesh *m2, const InterfaceInfo *if_info,
                                             FaceElementTransformations* &tr1,
                                             FaceElementTransformations* &tr2);

   virtual void TransferToGlobal(Array<GridFunction*> &us, GridFunction* &global_u) = 0;
};

class SubMeshTopologyHandler : public TopologyHandler
{
protected:
   // Global parent mesh that will be decomposed.
   Mesh *pmesh;

   // SubMesh does not allow creating Array of its pointers. Use std::shared_ptr.
   std::vector<std::shared_ptr<SubMesh>> meshes;

   // face/element map from each subdomain to parent mesh.
   Array<Array<int> *> parent_face_map;
   Array<Array<int> *> parent_elem_map;

public:
   SubMeshTopologyHandler();

   // Export mesh pointers and interface info.
   SubMeshTopologyHandler(Array<Mesh*> &mesh_ptrs, Array<InterfaceInfo>* &if_infos, TopologyData &topol_data);

   virtual ~SubMeshTopologyHandler();

   // access
   virtual Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   virtual Mesh* GetGlobalMesh() { return pmesh; }

   virtual void TransferToGlobal(Array<GridFunction*> &us, GridFunction* &global_u);

protected:
   // SubMesh does not support face mapping for 2d meshes.
   Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm);
   void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map=NULL);
   void UpdateBdrAttributes(Mesh& m);

   void BuildInterfaceInfos();
   Array<int> FindParentInterfaceInfo(const int pface,
                                       const int imesh, const int ibe,
                                       const int jmesh, const int jbe);
};

#endif
