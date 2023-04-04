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

#include "component_topology_handler.hpp"
#include "hdf5.h"
#include "hdf5_utils.hpp"

using namespace std;
using namespace mfem;

ComponentTopologyHandler::ComponentTopologyHandler()
   : TopologyHandler()
{
   // read global file.
   std::string filename = config.GetRequiredOption<std::string>("mesh/component-wise/global_config");
   ReadGlobalConfigFromFile(filename);

   SetupComponents();

   // Assume all components have the same spatial dimension.
   dim = components[0]->Dimension();

   if (num_ref_ports > 0)
      SetupReferencePorts();

   // Do we really need to copy all meshes?
   SetupMeshes();

   SetupReferenceInterfaces();

   // Do we really need to set boundary attributes of all meshes?
   SetupBoundaries();
}

void ComponentTopologyHandler::SetupComponents()
{
   assert(num_comp > 0);

   YAML::Node comp_list = config.FindNode("mesh/component-wise/components");
   if (!comp_list) mfem_error("ComponentTopologyHandler: component list does not exist!\n");

   // We only read the components that are specified in the global configuration.
   components.SetSize(num_comp);
   components = NULL;
   for (int c = 0; c < comp_list.size(); c++)
   {
      std::string comp_name = config.GetRequiredOptionFromDict<std::string>("name", comp_list[c]);
      // do not read if this component is not used in the global config.
      if (!comp_names.count(comp_name)) continue;

      int idx = comp_names[comp_name];
      std::string filename = config.GetRequiredOptionFromDict<std::string>("file", comp_list[c]);
      components[idx] = new BlockMesh(filename.c_str());
   }

   for (int c = 0; c < components.Size(); c++) assert(components[c] != NULL);
}

void ComponentTopologyHandler::SetupReferencePorts()
{
   assert(num_ref_ports > 0);

   ref_ports.SetSize(num_ref_ports);
   ref_ports = NULL;

   YAML::Node port_list = config.FindNode("mesh/component-wise/ports");
   if (!port_list)
      mfem_error("ComponentTopologyHandler: port list does not exist!\n");
   else
   {
      for (int p = 0; p < port_list.size(); p++)
      {
         // Read hdf5 files.
         std::string filename = config.GetRequiredOptionFromDict<std::string>("file", port_list[p]);

         ReadPortsFromFile(filename);
      }
   }
   
   for (int p = 0; p < ref_ports.Size(); p++) assert(ref_ports[p] != NULL);
}

void ComponentTopologyHandler::ReadGlobalConfigFromFile(const std::string filename)
{
   hid_t file_id;
   hid_t grp_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);
   
   {  // Component list.
      // This line below currently does not work.
      // hdf5_utils::ReadDataset(file_id, "components", comp_names);
      grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
      assert(grp_id >= 0);

      hdf5_utils::ReadAttribute(grp_id, "number_of_components", num_comp);
      for (int c = 0; c < num_comp; c++)
      {
         std::string tmp;
         hdf5_utils::ReadAttribute(grp_id, std::to_string(c).c_str(), tmp);
         comp_names[tmp] = c;
      }

      // Mesh list.
      hdf5_utils::ReadDataset(grp_id, "meshes", mesh_types);

      // Configuration (translation, rotation) of meshes.
      Array2D<double> tmp;
      hdf5_utils::ReadDataset(grp_id, "configuration", tmp);
      assert(mesh_types.Size() == tmp.NumRows());
      numSub = mesh_types.Size();

      mesh_configs.SetSize(numSub);
      for (int m = 0; m < numSub; m++)
      {
         const double *m_cf = tmp.GetRow(m);
         for (int d = 0; d < 3; d++)
         {
            mesh_configs[m].trans[d] = m_cf[d];
            mesh_configs[m].rotate[d] = m_cf[d + 3];
         }
      }

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   {  // Port list.
      grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
      assert(grp_id >= 0);

      hdf5_utils::ReadAttribute(grp_id, "number_of_references", num_ref_ports);
      // // hdf5_utils::ReadDataset(file_id, "ports", ports);
      for (int p = 0; p < num_ref_ports; p++)
      {
         std::string tmp;
         hdf5_utils::ReadAttribute(grp_id, std::to_string(p).c_str(), tmp);
         port_names[tmp] = p;
      }

      // Global interface port data.
      Array2D<int> tmp;
      hdf5_utils::ReadDataset(grp_id, "interface", tmp);
      num_ports = tmp.NumRows();
      port_infos.SetSize(num_ports);
      port_types.SetSize(num_ports);

      for (int p = 0; p < num_ports; p++)
      {
         const int *p_data = tmp.GetRow(p);
         port_infos[p].Mesh1 = p_data[0];
         port_infos[p].Mesh2 = p_data[1];
         port_infos[p].Attr1 = p_data[2];
         port_infos[p].Attr2 = p_data[3];
         port_types[p] = p_data[4];
      }

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   {  // Boundary data.
      Array2D<int> tmp;
      hdf5_utils::ReadDataset(file_id, "boundary", tmp);
      bdr_c2g.SetSize(numSub);
      for (int m = 0; m < numSub; m++) bdr_c2g[m] = new std::unordered_map<int,int>;
      bdr_attributes.SetSize(0);

      for (int b = 0; b < tmp.NumRows(); b++)
      {
         const int *b_data = tmp.GetRow(b);
         int m = b_data[1];
         (*bdr_c2g[m])[b_data[2]] = b_data[0];

         int idx = bdr_attributes.Find(b_data[0]);
         if (idx < 0) bdr_attributes.Append(b_data[0]);
      }
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);
}

void ComponentTopologyHandler::ReadPortsFromFile(const std::string filename)
{
   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   // number of ports stored in the given hdf5 file.
   int size = -1;
   hdf5_utils::ReadAttribute(file_id, "number_of_ports", size);
   assert(size > 0);

   for (int k = 0; k < size; k++)
   {
      hid_t grp_id;
      grp_id = H5Gopen2(file_id, std::to_string(k).c_str(), H5P_DEFAULT);
      assert(grp_id >= 0);

      std::string port_name;
      hdf5_utils::ReadAttribute(grp_id, "name", port_name);
      // Read only the ports that are specified in the global configuration.
      if (!port_names.count(port_name))
      {
         errf = H5Gclose(grp_id);
         assert(errf >= 0);
         continue;
      }

      int battr1, battr2;
      hdf5_utils::ReadAttribute(grp_id, "bdr_attr1", battr1);
      hdf5_utils::ReadAttribute(grp_id, "bdr_attr2", battr2);
      printf("port: %d - %d\n", battr1, battr2);

      std::string name1, name2;
      hdf5_utils::ReadAttribute(grp_id, "component1", name1);
      hdf5_utils::ReadAttribute(grp_id, "component2", name2);

      int comp1 = comp_names[name1];
      int comp2 = comp_names[name2];
      assert((comp1 >= 0) && (comp2 >= 0));

      Array<int> vtx1, vtx2;
      hdf5_utils::ReadDataset(grp_id, "vtx1", vtx1);
      hdf5_utils::ReadDataset(grp_id, "vtx2", vtx2);
      assert(vtx1.Size() == vtx2.Size());

      printf("vtx1: ");
      for (int v = 0; v < vtx1.Size(); v++) printf("%d\t", vtx1[v]);
      printf("\n");
      printf("vtx2: ");
      for (int v = 0; v < vtx2.Size(); v++) printf("%d\t", vtx2[v]);
      printf("\n");

      Array<int> be1, be2;
      hdf5_utils::ReadDataset(grp_id, "be1", be1);
      hdf5_utils::ReadDataset(grp_id, "be2", be2);
      assert(be1.Size() == be2.Size());

      printf("be1: ");
      for (int v = 0; v < be1.Size(); v++) printf("%d\t", be1[v]);
      printf("\n");
      printf("be2: ");
      for (int v = 0; v < be2.Size(); v++) printf("%d\t", be2[v]);
      printf("\n");

      errf = H5Gclose(grp_id);
      assert(errf >= 0);

      int port_idx = port_names[port_name];
      ref_ports[port_idx] = new PortData;
      PortData *port = ref_ports[port_idx];
      port->Component1 = comp1;
      port->Component2 = comp2;
      port->Attr1 = battr1;
      port->Attr2 = battr2;
      for (int v = 0; v < vtx1.Size(); v++)
         port->vtx2to1[vtx2[v]] = vtx1[v];
      port->be_pairs.SetSize(be1.Size(), 2);
      for (int v = 0; v < be1.Size(); v++)
      {
         int *be_pair = port->be_pairs.GetRow(v);
         be_pair[0] = be1[v];
         be_pair[1] = be2[v];
      }
   }  // for (int k = 0; k < size; k++)

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

void ComponentTopologyHandler::SetupMeshes()
{
   assert(numSub > 0);
   assert(mesh_types.Size() == numSub);
   assert(mesh_configs.Size() == numSub);

   meshes.SetSize(numSub);
   meshes = NULL;

   for (int m = 0; m < numSub; m++)
   {
      meshes[m] = new Mesh(*components[mesh_types[m]]);

      for (int d = 0; d < 3; d++)
      {
         mesh_config::trans[d] = mesh_configs[m].trans[d];
         mesh_config::rotate[d] = mesh_configs[m].rotate[d];
      }
      meshes[m]->Transform(mesh_config::Transform2D);
   }

   for (int m = 0; m < numSub; m++) assert(meshes[m] != NULL);
}

void ComponentTopologyHandler::SetupBoundaries()
{
   assert(meshes.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      std::unordered_map<int,int> *c2g_map = bdr_c2g[m];
      for (int be = 0; be < meshes[m]->GetNBE(); be++)
      {
         int c_attr = meshes[m]->GetBdrAttribute(be);
         if(!c2g_map->count(c_attr)) continue;

         meshes[m]->SetBdrAttribute(be, (*c2g_map)[c_attr]);
      }

      UpdateBdrAttributes(*meshes[m]);
   }
}

void ComponentTopologyHandler::SetupReferenceInterfaces()
{
   ref_interfaces.SetSize(num_ref_ports);
   
   for (int i = 0; i < num_ref_ports; i++)
   {
      Array2D<int> *be_pairs = &(ref_ports[i]->be_pairs);
      std::unordered_map<int,int> *vtx2to1 = &(ref_ports[i]->vtx2to1);
      int comp1 = ref_ports[i]->Component1;
      int comp2 = ref_ports[i]->Component2;
      int attr1 = ref_ports[i]->Attr1;
      int attr2 = ref_ports[i]->Attr2;

      ref_interfaces[i] = new Array<InterfaceInfo>(0);

      for (int be = 0; be < be_pairs->NumRows(); be++)
      {
         InterfaceInfo tmp;

         int *pair = be_pairs->GetRow(be);
         tmp.BE1 = pair[0];
         tmp.BE2 = pair[1];

         // use the face index from each component mesh.
         int f1 = components[comp1]->GetBdrFace(tmp.BE1);
         int f2 = components[comp2]->GetBdrFace(tmp.BE2);
         int Inf1, Inf2, dump;
         components[comp1]->GetFaceInfos(f1, &Inf1, &dump);
         components[comp2]->GetFaceInfos(f2, &Inf2, &dump);
         tmp.Inf1 = Inf1;

         // determine orientation of the face with respect to mesh2/elem2.
         Array<int> vtx1, vtx2;
         components[comp1]->GetBdrElementVertices(tmp.BE1, vtx1);
         components[comp2]->GetBdrElementVertices(tmp.BE2, vtx2);
         for (int v = 0; v < vtx2.Size(); v++) vtx2[v] = (*vtx2to1)[vtx2[v]];
         switch (dim)
         {
            case 1:
            {
               break;
            }
            case 2:
            {
               if ((vtx1[1] == vtx2[0]) && (vtx1[0] == vtx2[1]))
               {
                  tmp.Inf2 = 64 * (Inf2 / 64) + 1;
               }
               else if ((vtx1[0] == vtx2[0]) && (vtx1[1] == vtx2[1]))
               {
                  tmp.Inf2 = 64 * (Inf2 / 64);
               }
               else
               {
                  mfem_error("orientation error!\n");
               }
               break;
            }
            case 3:
            {
               mfem_error("not implemented yet!\n");
               break;
            }
         }  // switch (dim)

         ref_interfaces[i]->Append(tmp);
      }  // for (int be = 0; be < be_pair->Size(); be++)

      std::string format = "";
      for (int k = 0; k < 8; k++) format += "%d\t";
      format += "%d\n";
      printf("Reference Interface %d informations\n", i);
      printf("Attr\tMesh1\tMesh2\tBE1\tBE2\tIdx1\tOri1\tIdx2\tOri2\n");
      for (int k = 0; k < ref_interfaces[i]->Size(); k++)
      {
         InterfaceInfo *tmp = &(*ref_interfaces[i])[k];
         printf(format.c_str(), -1, comp1, comp2,
                              tmp->BE1, tmp->BE2, tmp->Inf1 / 64,
                              tmp->Inf1 % 64, tmp->Inf2 / 64, tmp->Inf2 % 64);
      }
      printf("\n");
   }  // for (int i = 0; i < num_ref_ports; i++)
}