// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

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

enum TopologyHandlerMode
{
   SUBMESH,
   COMPONENT,
   NUM_TOPOL_MODE
};

// Port information for global configuration.
struct PortInfo {
   int PortAttr;        // Global boundary attribute.
   int Mesh1, Mesh2;    // Mesh indexes in global configuration
   int Attr1, Attr2;    // boundary attribute of each mesh sharing the port.
};

// Interface information between two boundary elements within a port.
struct InterfaceInfo {
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
   Array<int> *global_bdr_attributes = NULL;
};

const TopologyHandlerMode SetTopologyHandlerMode();

class TopologyHandler
{
protected:
   int numSub = -1;   // number of subdomains.
   int num_comp = -1;  // number of components. Submesh - only one component / Component - multiple compoenents allowed, not yet implemented.
   Array<int> sub_composition;  // number of subdomains per each component index.
   std::vector<std::string> comp_names;

   // Spatial dimension.
   int dim = -1;

   DecompositionMode dd_mode = NUM_DDMODE;
   TopologyHandlerMode type = NUM_TOPOL_MODE;

   // Component index for each block.
   // Submesh: allows only one component (0).
   // Component: multiple components allowed, but not yet implemented.
   Array<int> mesh_types;
   Array<int> mesh_comp_idx;  // mesh[m] is (mesh_comp_idx[m])-th mesh of component mesh_types[m].

   int num_ports = -1;        // number of ports.
   Array<PortInfo> port_infos;
   Array<Array<InterfaceInfo>*> interface_infos;

public:
   TopologyHandler(const TopologyHandlerMode &input_type);

   // ownership of interface_infos changes depending on derived classes.
   // not deleting here.
   virtual ~TopologyHandler() {}

   // access
   const TopologyHandlerMode GetType() { return type; }
   const int GetNumSubdomains() { return numSub; }
   const int GetNumSubdomains(const int &c) { return sub_composition[c]; }
   const int GetNumComponents() { return num_comp; }
   const int GetNumPorts() { return num_ports; }
   const int GetMeshType(const int &m) const { return mesh_types[m]; }
   const int GetComponentIndexOfMesh(const int &m) { return mesh_comp_idx[m]; }
   const std::string GetComponentName(const int &c) const { return comp_names[c]; }
   const PortInfo* GetPortInfo(const int &k) { return &(port_infos[k]); }
   Array<InterfaceInfo>* const GetInterfaceInfos(const int &k) { return interface_infos[k]; }
   virtual Mesh* GetMesh(const int k) = 0;
   virtual Mesh* GetGlobalMesh() = 0;

   /*
      Methods only for ComponentTopologyHandler 
   */
   virtual const int GetPortType(const int &port_idx)
   { mfem_error("TopologyHandler::GetPortType is abstract method!\n"); return -1; }
   virtual const int GetNumRefPorts()
   { mfem_error("TopologyHandler::GetNumRefPorts is abstract method!\n"); return -1; }
   virtual Mesh* GetComponentMesh(const int &c)
   { mfem_error("TopologyHandler::GetComponentMesh is abstract method!\n"); return NULL; }
   // return component indexes for a reference port
   virtual void GetComponentPair(const int &ref_port_idx, int &comp1, int &comp2)
   { mfem_error("TopologyHandler::GetComponentPair is abstract method!\n"); return; }
   virtual Array<InterfaceInfo>* const GetRefInterfaceInfos(const int &k)
   { mfem_error("TopologyHandler::GetRefInterfaceInfos is abstract method!\n"); return NULL; }
   virtual Array<int>* GetBdrAttrComponentToGlobalMap(const int &m)
   { mfem_error("TopologyHandler::GetBdrAttrComponentToGlobalMap is abstract method!\n"); return NULL; }

   /******/

   // Export mesh pointers and interface info.
   virtual void ExportInfo(Array<Mesh*> &mesh_ptrs, TopologyData &topol_data) = 0;

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

   virtual void TransferToGlobal(Array<GridFunction*> &us, Array<GridFunction*> &global_u, const int &num_var) = 0;

   virtual void PrintPortInfo(const int k = -1);
   virtual void PrintInterfaceInfo(const int k = -1);

protected:
   virtual void UpdateAttributes(Mesh& m);
   virtual void UpdateBdrAttributes(Mesh& m);
};

class SubMeshTopologyHandler : public TopologyHandler
{
protected:
   // Global parent mesh that will be decomposed.
   Mesh *pmesh;
   bool own_pmesh = false;

   // SubMesh does not allow creating Array of its pointers. Use std::shared_ptr.
   std::vector<std::shared_ptr<SubMesh>> meshes;

   // face/element map from each subdomain to parent mesh.
   Array<Array<int> *> parent_face_map;
   Array<Array<int> *> parent_elem_map;

public:
   SubMeshTopologyHandler(Mesh* pmesh_);
   // Read mesh file from input.
   SubMeshTopologyHandler();

   virtual ~SubMeshTopologyHandler();

   // access
   virtual Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   virtual Mesh* GetGlobalMesh() { return pmesh; }

   // SubMeshTopologyHandler assumes only one component.
   virtual Mesh* GetComponentMesh(const int &c) { return GetMesh(0); }

   // Export mesh pointers and interface info.
   virtual void ExportInfo(Array<Mesh*> &mesh_ptrs, TopologyData &topol_data);

   virtual void TransferToGlobal(Array<GridFunction*> &us, Array<GridFunction*> &global_u, const int &num_var);

protected:
   // SubMesh does not support face mapping for 2d meshes.
   Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm);
   void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map=NULL);

   void BuildInterfaceInfos();
   Array<int> FindParentInterfaceInfo(const int pface,
                                       const int imesh, const int ibe,
                                       const int jmesh, const int jbe);
};

#endif
