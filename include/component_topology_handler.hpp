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

#ifndef COMPONENT_TOPOLOGY_HANDLER_HPP
#define COMPONENT_TOPOLOGY_HANDLER_HPP

#include "topology_handler.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

namespace mesh_config
{

static double trans[3], rotate[3];
typedef void TransformFunction(const Vector &, Vector &);

static void Transform2D(const Vector &x, Vector &y)
{
   assert(x.Size() == 2);
   y.SetSize(2);
   double sint = sin(mesh_config::rotate[0]);
   double cost = cos(mesh_config::rotate[0]);
   y[0] = cost * x[0] - sint * x[1];
   y[1] = sint * x[0] + cost * x[1];

   y[0] += mesh_config::trans[0];
   y[1] += mesh_config::trans[1];
}

static void InverseTransform2D(const Vector &x, Vector &y)
{
   assert(x.Size() == 2);
   y.SetSize(2);
   double tmp0 = x(0) - mesh_config::trans[0];
   double tmp1 = x(1) - mesh_config::trans[1];

   double sint = sin(- mesh_config::rotate[0]);
   double cost = cos(- mesh_config::rotate[0]);
   y(0) = cost * tmp0 - sint * tmp1;
   y(1) = sint * tmp0 + cost * tmp1;
}

static void Transform3D(const Vector &x, Vector &y)
{
   assert(x.Size() == 3);
   y.SetSize(3);

   for (int d = 0; d < 3; d++)
      y(d) = x(d) + mesh_config::trans[d];
}

static void InverseTransform3D(const Vector &x, Vector &y)
{
   assert(x.Size() == 3);
   y.SetSize(3);

   for (int d = 0; d < 3; d++)
      y(d) = x(d) - mesh_config::trans[d];
}

}

class BlockMesh : public Mesh
{
public:
   BlockMesh(const char *filename) : Mesh(filename) {};
   BlockMesh(const Mesh &mesh) : Mesh(mesh) {};

   // these protected members are needed to be public for GenerateInterfaces.
   using Mesh::GetTriOrientation;
   using Mesh::GetQuadOrientation;
};

class ComponentTopologyHandler : public TopologyHandler
{
public:
   struct PortData {
      int Component1, Component2;   // component mesh indexes.
      int Attr1, Attr2;             // boundary attribute of each component mesh sharing the port.

      std::unordered_map<int,int> vtx2to1;             // vertex mapping from component 2 to component 1.
      Array2D<int> be_pairs;  // boundary element pairs between component 1 and 2.
   };

protected:
//    // Global parent mesh that will be decomposed.
//    Mesh *pmesh;

   // Print out details.
   bool verbose = false;

   // Write built ports.
   bool write_ports = false;

   // Map from component name to array index.
   std::unordered_map<std::string, int> comp_name2idx;
   // Reference meshes for components.
   Array<BlockMesh*> components;

   // threshold for matching vertices in BuildPortDataFromInput.
   double vtx_gap_thrs = -1.0;

   struct MeshConfig {
      double trans[3];
      double rotate[3];
   };
   // configuration of each meshes
   Array<MeshConfig> mesh_configs;
   mesh_config::TransformFunction *tf_ptr = NULL;
   mesh_config::TransformFunction *inv_tf_ptr = NULL;

   // Meshes for global configuration.
   Array<Mesh*> meshes;

   // Reference ports between components.
   int num_ref_ports = -1;
   std::unordered_map<std::string, int> port_names;
   Array<PortData*> ref_ports;
   Array<Array<InterfaceInfo>*> ref_interfaces;

   // Global port configuration
   Array<int> port_types;

   // Boundary information for global configuration.
   Array<Array<int>*> bdr_c2g;   // size of meshes.Size(). component battr to global battr.
   Array<int> bdr_attributes;

public:
   ComponentTopologyHandler();

//    virtual ~SubMeshTopologyHandler();

   // access
   virtual Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   virtual Mesh* GetGlobalMesh()
   { mfem_error("ComponenetTopologyHandler does not support a global mesh!\n"); return NULL; }
   virtual const int GetNumRefPorts() { return num_ref_ports; }
   virtual const int GetPortType(const int &port_idx) { return port_types[port_idx]; }
   virtual PortData* GetPortData(const int r) { return ref_ports[r]; }
   virtual Mesh* GetComponentMesh(const int &c) { return components[c]; }
   virtual Array<InterfaceInfo>* const GetRefInterfaceInfos(const int &k) { return ref_interfaces[k]; }
   virtual Array<int>* GetBdrAttrComponentToGlobalMap(const int &m) { return bdr_c2g[m]; }

   // return component indexes for a reference port (ComponentTopologyHandler only)
   virtual void GetComponentPair(const int &ref_port_idx, int &comp1, int &comp2);

   // Export mesh pointers and interface info.
   virtual void ExportInfo(Array<Mesh*> &mesh_ptrs, TopologyData &topol_data);

   virtual void TransferToGlobal(Array<GridFunction*> &us, Array<GridFunction*> &global_u, const int &num_var)
   { mfem_error("ComponentTopologyHandler does not yet support global grid function/mesh!\n"); }

protected:
   // Get vertex orientation of face2 (from mesh2) with respect to face1 (mesh1).
   int GetOrientation(BlockMesh *comp1, const Element::Type &be_type, const Array<int> &vtx1, const Array<int> &vtx2);

   // Global configuration data
   // Read component list and names. not the actual meshes.
   void ReadComponentsFromFile(const std::string filename);
   // Read reference port list and names. not the actual port data.
   void ReadPortsFromFile(const std::string filename);
   // Read boundary attribute map between components and global.
   void ReadBoundariesFromFile(const std::string filename);

   // Reference port data
   void ReadPortDatasFromFile(const std::string filename);
   void BuildPortDataFromInput(const YAML::Node port_dict);
   void WritePortDataToFile(const PortData &port, const std::string &port_name, const std::string &filename);

   void SetupComponents();
   void SetupReferencePorts();
   void SetupMeshes();
   void SetupBdrAttributes();
   void SetupReferenceInterfaces();
   void SetupPorts();

   bool ComponentBdrAttrCheck(Mesh *comp);
};

#endif
