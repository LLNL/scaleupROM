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

#include <fstream>
#include <iostream>
#include "utils/HDFDatabase.h"
#include "mfem.hpp"
#include "topology_handler.hpp"

using namespace mfem;

class BlockMesh : public Mesh
{
public:
   BlockMesh(const char *filename) : Mesh(filename) {};
   BlockMesh(const Mesh &mesh) : Mesh(mesh) {};

   // these protected members are needed to be public for GenerateInterfaces.
   // using Mesh::AddPointFaceElement;
   // using Mesh::AddSegmentFaceElement;
   // using Mesh::AddTriangleFaceElement;
   // using Mesh::AddQuadFaceElement;
   using Mesh::GetTriOrientation;
   using Mesh::GetQuadOrientation;
   using Mesh::GetTetOrientation;

};

int main(int argc, char *argv[])
{
   // 2 Dimension
   {
      // const int nx_ = 2;
      // const int a_ = 2.0;
      // const int dx = 2;
      // Mesh mesh = Mesh::MakeCartesian2D(nx_, nx_, Element::QUADRILATERAL, false, a_, a_, false);
      Mesh mesh("meshes/2x2.mesh");

      // int numElem = mesh.GetNE();
      // for (int elem = 0; elem < numElem; elem++)
      // {
      //    mesh.SetAttribute(elem, elem+1);
      // }

      Array<Mesh*> meshes;
      Array<InterfaceInfo> *if_info = NULL;
      TopologyData topol_data;
      SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler(&mesh);
      submesh->ExportInfo(meshes, if_info, topol_data);

      for (int m = 0; m < meshes.Size(); m++)
      {
         printf("Mesh %d\n", m);
         printf("Vertex data.\n");
         printf("Vertex\tLocation\n");
         for (int v = 0; v < meshes[m]->GetNV(); v++)
         {
            const double *vtx = meshes[m]->GetVertex(v);
            printf("%d\t(%.3f\t%.3f)\n", v, vtx[0], vtx[1]);
         }
         printf("Boundary element data.\n");
         printf("Attr\tFace\tVertices\n");
         for (int be = 0; be < meshes[m]->GetNBE(); be++)
         {
            int battr = meshes[m]->GetBdrAttribute(be);
            int face = meshes[m]->GetBdrFace(be);
            Array<int> vtx;
            // comp.GetFaceVertices(face, vtx);
            meshes[m]->GetBdrElementVertices(be, vtx);

            printf("%d\t%d\t", battr, face);
            for (int v = 0; v < vtx.Size(); v++) printf("%d\t", vtx[v]);
            printf("\n");
         }
      }

      assert(if_info != NULL);
      printf("numSub: %d\n", topol_data.numSub);

      std::string format = "";
      for (int k = 0; k < 8; k++) format += "%d\t";
      format += "%d\n";
      printf("Interface informations\n");
      printf("Attr\tMesh1\tMesh2\tBE1\tBE2\tIdx1\tOri1\tIdx2\tOri2\n");
      for (int k = 0; k < if_info->Size(); k++)
      {
         printf(format.c_str(), (*if_info)[k].Attr, (*if_info)[k].Mesh1, (*if_info)[k].Mesh2,
                                (*if_info)[k].BE1, (*if_info)[k].BE2, (*if_info)[k].Inf1 / 64,
                                (*if_info)[k].Inf1 % 64, (*if_info)[k].Inf2 / 64, (*if_info)[k].Inf2 % 64);
      }
      printf("\n");

      // assert(nx_ % dx == 0);
      printf("Component mesh.\n");
      // BlockMesh comp(Mesh::MakeCartesian2D(nx_ / dx, nx_ / dx, Element::QUADRILATERAL, false, a_ / dx, a_ / dx, false));
      BlockMesh comp("meshes/1x1.mesh");

      // comp.UniformRefinement();

      printf("%d\n", comp.GetNBE());
      printf("Boundary attributes\n");
      for (int k = 0; k < comp.bdr_attributes.Size(); k++) printf("%d\t", comp.bdr_attributes[k]);
      printf("\n\n");

      // This extract a specific geometry-type boundary elements.
      int geom = Geometry::Type::SEGMENT;
      Array<int> bdr_elem_vtx, bdr_attr;
      comp.GetBdrElementData(geom, bdr_elem_vtx, bdr_attr);
      printf("Geometry: %d\n", geom);
      printf("BE vertices:\n");
      for (int k = 0; k < bdr_elem_vtx.Size(); k++) printf("%d\t", bdr_elem_vtx[k]);
      printf("\n");
      printf("BE attributes:\n");
      for (int k = 0; k < bdr_attr.Size(); k++) printf("%d\t", bdr_attr[k]);
      printf("\n\n");
      
      printf("Vertex data.\n");
      printf("Vertex\tLocation\n");
      for (int v = 0; v < comp.GetNV(); v++)
      {
         const double *vtx = comp.GetVertex(v);
         printf("%d\t(%.3f\t%.3f)\n", v, vtx[0], vtx[1]);
      }
      printf("Boundary element data.\n");
      printf("Attr\tFace\tVertices\n");
      for (int be = 0; be < comp.GetNBE(); be++)
      {
         int battr = comp.GetBdrAttribute(be);
         int face = comp.GetBdrFace(be);
         Array<int> vtx;
         // comp.GetFaceVertices(face, vtx);
         comp.GetBdrElementVertices(be, vtx);

         printf("%d\t%d\t", battr, face);
         for (int v = 0; v < vtx.Size(); v++) printf("%d\t", vtx[v]);
         printf("\n");
      }

      // Mesh does not have a method to collect all vertices of the same boundary attributes.
      int ref_bdr_attrs = comp.bdr_attributes.Size();
      Array<Array<int>*> bdr_vtx(comp.bdr_attributes.Size());
      for (int k = 0; k < bdr_vtx.Size(); k++) bdr_vtx[k] = new Array<int>(0);

      for (int be = 0; be < comp.GetNBE(); be++)
      {
         int battr = comp.GetBdrAttribute(be);
         int idx = comp.bdr_attributes.Find(battr);
         assert(idx >= 0);
         Array<int> vtx;
         comp.GetBdrElementVertices(be, vtx);
         for (int k = 0; k < vtx.Size(); k++)
         {
            int vtx_idx = bdr_vtx[idx]->Find(vtx[k]);
            if (vtx_idx < 0) bdr_vtx[idx]->Append(vtx[k]);
         }
      }

      printf("Boundary vertex data.\n");
      printf("Attr\tVertices\n");
      for (int k = 0; k < ref_bdr_attrs; k++)
      {
         printf("%d\t", comp.bdr_attributes[k]);
         for (int v = 0; v < bdr_vtx[k]->Size(); v++) printf("%d\t", (*bdr_vtx[k])[v]);
         printf("\n");
      }
      printf("\n");

      int n_interface_pairs = 2;
      Array<int> if_comp1(n_interface_pairs), if_comp2(n_interface_pairs);
      if_comp1 = 0;   // single component.
      if_comp2 = 0;

      Array<int> if_battr1(n_interface_pairs), if_battr2(n_interface_pairs);
      if_battr1[0] = 3;
      if_battr2[0] = 1;
      if_battr1[1] = 2;
      if_battr2[1] = 4;

      // These will be stored in a 'interface pair' file, with the comp/battr pair above.
      Array<std::map<int,int>*> vtx_2to1(n_interface_pairs);
      // Array<std::map<int,int>*> be_2to1(n_interface_pairs);
      Array<Array<std::pair<int,int>>*> be_pairs(n_interface_pairs);
      for (int k = 0; k < n_interface_pairs; k++)
      {
         int bidx1 = comp.bdr_attributes.Find(if_battr1[k]);
         int bidx2 = comp.bdr_attributes.Find(if_battr2[k]);
         assert((bidx1 >= 0) && (bidx2 >= 0));
         assert(bdr_vtx[bidx1]->Size() == bdr_vtx[bidx2]->Size());

         vtx_2to1[k] = new std::map<int,int>;
         be_pairs[k] = new Array<std::pair<int,int>>(0);
      }

      (*vtx_2to1[0])[1] = 3;
      (*vtx_2to1[0])[0] = 2;
      // (*be_pairs[0])[0] = 1;
      be_pairs[0]->Append({1, 0});

      (*vtx_2to1[1])[0] = 1;
      (*vtx_2to1[1])[2] = 3;
      // (*be_pairs[0])[2] = 3;
      be_pairs[1]->Append({3, 2});

      // These are the informations that will be stored in the 'global topology' file.
      Array<int> global_comps(topol_data.numSub);
      global_comps = 0;

      const int dim = topol_data.dim;
      const int numSub = topol_data.numSub;
      const int global_if_pairs = 4;
      Array<int> global_idx1(global_if_pairs), global_idx2(global_if_pairs),
                 global_battr1(global_if_pairs), global_battr2(global_if_pairs),
                 global_if_type(global_if_pairs);
      /*
            2  3
            0  1  
      */
      // interface between 0 and 1 - interface type 1 (between be2 and be4).
      global_idx1[0] = 0;
      global_idx2[0] = 1;
      global_battr1[0] = 2;
      global_battr2[0] = 4;
      global_if_type[0] = 1;
      // interface between 0 and 2 - interface type 0 (between be3 and be1).
      global_idx1[1] = 0;
      global_idx2[1] = 2;
      global_battr1[1] = 3;
      global_battr2[1] = 1;
      global_if_type[1] = 0;
      // interface between 1 and 3 - interface type 0 (between be3 and be1).
      global_idx1[2] = 1;
      global_idx2[2] = 3;
      global_battr1[2] = 3;
      global_battr2[2] = 1;
      global_if_type[2] = 0;
      // interface between 2 and 3 - interface type 1 (between be2 and be4).
      global_idx1[3] = 2;
      global_idx2[3] = 3;
      global_battr1[3] = 2;
      global_battr2[3] = 4;
      global_if_type[3] = 1;

      for (int m = 0; m < topol_data.numSub; m++)
         meshes[m] = new BlockMesh(comp);

      Array<InterfaceInfo> comp_if_info(0);
      {
         for (int i = 0; i < global_if_pairs; i++)
         {
            int if_pair_type = global_if_type[i];
            Array<std::pair<int,int>> *be_pair = be_pairs[if_pair_type];
            std::map<int,int> *if_vtx2to1 = vtx_2to1[if_pair_type];
            for (int be = 0; be < be_pair->Size(); be++)
            {
               InterfaceInfo tmp;

               // TODO: read value from hdf5 file.
               tmp.Attr = i + 5;
               tmp.Mesh1 = global_idx1[i];
               tmp.Mesh2 = global_idx2[i];

               tmp.BE1 = (*be_pair)[be].first;
               tmp.BE2 = (*be_pair)[be].second;

               // use the face index from each component mesh.
               int f1 = meshes[tmp.Mesh1]->GetBdrFace(tmp.BE1);
               int f2 = meshes[tmp.Mesh2]->GetBdrFace(tmp.BE2);
               int Inf1, Inf2, dump;
               meshes[tmp.Mesh1]->GetFaceInfos(f1, &Inf1, &dump);
               meshes[tmp.Mesh2]->GetFaceInfos(f2, &Inf2, &dump);
               tmp.Inf1 = Inf1;

               // determine orientation of the face with respect to mesh2/elem2.
               Array<int> vtx1, vtx2;
               meshes[tmp.Mesh1]->GetBdrElementVertices(tmp.BE1, vtx1);
               meshes[tmp.Mesh2]->GetBdrElementVertices(tmp.BE2, vtx2);
               for (int v = 0; v < vtx2.Size(); v++) vtx2[v] = (*if_vtx2to1)[vtx2[v]];
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
                     break;
                  }
               }  // switch (dim)
               comp_if_info.Append(tmp);
            }  // for (int be = 0; be < be_pair->Size(); be++)
         }  // for (int i = 0; i < global_if_pairs; i++)

         printf("Component Interface informations\n");
         printf("Attr\tMesh1\tMesh2\tBE1\tBE2\tIdx1\tOri1\tIdx2\tOri2\n");
         for (int k = 0; k < comp_if_info.Size(); k++)
         {
            printf(format.c_str(), comp_if_info[k].Attr, comp_if_info[k].Mesh1, comp_if_info[k].Mesh2,
                                 comp_if_info[k].BE1, comp_if_info[k].BE2, comp_if_info[k].Inf1 / 64,
                                 comp_if_info[k].Inf1 % 64, comp_if_info[k].Inf2 / 64, comp_if_info[k].Inf2 % 64);
         }
         printf("\n");
      }  // Array<InterfaceInfo> comp_if_info(0);


   }  // 2 dimension

   return 0;
}