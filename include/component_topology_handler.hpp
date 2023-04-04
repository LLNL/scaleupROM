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

class BlockMesh : public Mesh
{
public:
   BlockMesh(const char *filename) : Mesh(filename) {};
   BlockMesh(const Mesh &mesh) : Mesh(mesh) {};

   // these protected members are needed to be public for GenerateInterfaces.
   using Mesh::GetTriOrientation;
   using Mesh::GetQuadOrientation;
   using Mesh::GetTetOrientation;
};

class ComponentTopologyHandler : public TopologyHandler
{
protected:
//    // Global parent mesh that will be decomposed.
//    Mesh *pmesh;

   int num_comp;  // number of components.
   // Map from component name to array index.
   std::unordered_map<std::string, int> comp_names;
   // Reference meshes for components.
   Array<BlockMesh*> components;

   struct MeshConfig {
      double trans[3];
      double rotate[3];
   };
   // configuration of each meshes
   Array<MeshConfig> mesh_configs;

   // Meshes for global configuration.
   Array<Mesh*> meshes;
   // Component index for each block.
   Array<int> mesh_types;

   struct PortData {
      int Component1, Component2;   // component mesh indexes.
      int Attr1, Attr2;             // boundary attribute of each component mesh sharing the port.

      std::unordered_map<int,int> vtx2to1;             // vertex mapping from component 2 to component 1.
      Array2D<int> be_pairs;  // boundary element pairs between component 1 and 2.
   };

   // Reference ports between components.
   int num_ref_ports = -1;
   std::unordered_map<std::string, int> port_names;
   Array<PortData*> ref_ports;
   Array<Array<InterfaceInfo>*> ref_interfaces;   // mesh indexes are replaced with component indexes.

   // Global port configuration
   Array<int> port_types;

   // Boundary information for global configuration.
   Array<std::unordered_map<int,int>*> bdr_c2g;
   Array<int> bdr_attributes;

public:
//    SubMeshTopologyHandler(Mesh* pmesh_);
//    // Read mesh file from input.
   ComponentTopologyHandler();

//    virtual ~SubMeshTopologyHandler();

   // access
   virtual Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   virtual Mesh* GetGlobalMesh()
   { mfem_error("ComponenetTopologyHandler does not support a global mesh!\n"); return NULL; }

   // Export mesh pointers and interface info.
   virtual void ExportInfo(Array<Mesh*> &mesh_ptrs, TopologyData &topol_data) {}

   virtual void TransferToGlobal(Array<GridFunction*> &us, GridFunction* &global_u) {}

protected:
   void ReadGlobalConfigFromFile(const std::string filename);
   void ReadPortsFromFile(const std::string filename);

   void SetupComponents();
   void SetupReferencePorts();
   void SetupMeshes();
   void SetupBoundaries();
   void SetupReferenceInterfaces();
   void SetupPorts();
};

namespace mesh_config
{

static double trans[3], rotate[3];

static void Transform2D(const Vector &x, Vector &y)
{
   assert(x.Size() == 2);
   y.SetSize(2);
   double sint = sin(mesh_config::rotate[0]);
   double cost = cos(mesh_config::rotate[0]);
   y(0) = cost * x(0) - sint * x(1);
   y(1) = sint * x(0) + cost * x(1);

   y(0) += mesh_config::trans[0];
   y(1) += mesh_config::trans[1];
}

static void Transform3D(const Vector &x, Vector &y)
{
   mfem_error("not implemented!\n");
}

}

#endif
