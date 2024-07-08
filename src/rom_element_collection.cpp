// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "rom_element_collection.hpp"
#include "etc.hpp"

using namespace mfem;

ROMLinearElement::ROMLinearElement(
   TopologyHandler *topol_handler_, const Array<FiniteElementSpace *> &fes_, const bool separate_variable_)
   : ROMElementCollection(topol_handler_, fes_, separate_variable_)
{
   comp.SetSize(num_comp);
   mass.SetSize(num_comp);
   const int block_size = (separate_variable) ? num_var : 1;
   for (int c = 0; c < num_comp; c++)
   {
      comp[c] = new MatrixBlocks(block_size, block_size);
      mass[c] = new MatrixBlocks(block_size, block_size);
   }

   bdr.SetSize(num_comp);
   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      bdr[c] = new Array<MatrixBlocks *>(comp->bdr_attributes.Size());
      for (int b = 0; b < bdr[c]->Size(); b++)
         (*bdr[c])[b] = new MatrixBlocks(block_size, block_size);
   }
   port.SetSize(num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
      port[p] = new MatrixBlocks(2 * block_size, 2 * block_size);
}

ROMLinearElement::~ROMLinearElement()
{
   DeletePointers(comp);
   for (int c = 0; c < bdr.Size(); c++)
   {
      DeletePointers((*bdr[c]));
      delete bdr[c];
   }
   DeletePointers(port);
}

void ROMLinearElement::Save(const std::string &filename)
{
   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   // components + boundary
   SaveCompBdrElems(file_id);

   // (reference) ports
   SaveItfaceElems(file_id);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

void ROMLinearElement::SaveCompBdrElems(hid_t &file_id)
{
   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, "components", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::WriteAttribute(grp_id, "number_of_components", num_comp);

   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      dset_name = topol_handler->GetComponentName(c);

      hid_t comp_grp_id;
      comp_grp_id = H5Gcreate(grp_id, dset_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      hdf5_utils::WriteDataset(comp_grp_id, "domain", *comp[c]);

      hdf5_utils::WriteDataset(comp_grp_id, "mass", *mass[c]);

      // boundaries are saved for each component group.
      SaveBdrElems(comp_grp_id, c);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

void ROMLinearElement::SaveBdrElems(hid_t &comp_grp_id, const int &comp_idx)
{
   assert(comp_grp_id >= 0);
   herr_t errf;
   hid_t bdr_grp_id;
   bdr_grp_id = H5Gcreate(comp_grp_id, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(bdr_grp_id >= 0);

   const int num_bdr = bdr[comp_idx]->Size();
   Mesh *comp = topol_handler->GetComponentMesh(comp_idx);
   assert(num_bdr == comp->bdr_attributes.Size());

   hdf5_utils::WriteAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);
   
   Array<MatrixBlocks *> *bdr_c = bdr[comp_idx];
   for (int b = 0; b < num_bdr; b++)
      hdf5_utils::WriteDataset(bdr_grp_id, std::to_string(b), *((*bdr_c)[b]));

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void ROMLinearElement::SaveItfaceElems(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gcreate(file_id, "ports", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::WriteAttribute(grp_id, "number_of_ports", num_ref_ports);
   
   std::string dset_name;
   int c1, c2, a1, a2;
   for (int p = 0; p < num_ref_ports; p++)
   {
      topol_handler->GetRefPortInfo(p, c1, c2, a1, a2);
      dset_name = topol_handler->GetComponentName(c1) + ":" + topol_handler->GetComponentName(c2);
      dset_name += "-" + std::to_string(a1) + ":" + std::to_string(a2);
      hdf5_utils::WriteDataset(grp_id, dset_name, *port[p]);
   }

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

void ROMLinearElement::Load(const std::string &filename)
{
   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   // components
   LoadCompBdrElems(file_id);

   // (reference) ports
   LoadItfaceElems(file_id);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

void ROMLinearElement::LoadCompBdrElems(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   int num_comp_;
   hdf5_utils::ReadAttribute(grp_id, "number_of_components", num_comp_);
   assert(num_comp_ >= num_comp);

   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      dset_name = topol_handler->GetComponentName(c);

      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, dset_name.c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      hdf5_utils::ReadDataset(comp_grp_id, "domain", *comp[c]);

      hdf5_utils::ReadDataset(comp_grp_id, "mass", *mass[c]);

      // boundary
      LoadBdrElems(comp_grp_id, c);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void ROMLinearElement::LoadBdrElems(hid_t &comp_grp_id, const int &comp_idx)
{
   assert(comp_grp_id >= 0);
   herr_t errf;
   hid_t bdr_grp_id;
   bdr_grp_id = H5Gopen2(comp_grp_id, "boundary", H5P_DEFAULT);
   assert(bdr_grp_id >= 0);

   int num_bdr;
   hdf5_utils::ReadAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);

   Mesh *comp = topol_handler->GetComponentMesh(comp_idx);
   assert(num_bdr == comp->bdr_attributes.Size());
   assert(num_bdr = bdr[comp_idx]->Size());

   Array<MatrixBlocks *> *bdr_c = bdr[comp_idx];
   for (int b = 0; b < num_bdr; b++)
      hdf5_utils::ReadDataset(bdr_grp_id, std::to_string(b), *(*bdr_c)[b]);

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void ROMLinearElement::LoadItfaceElems(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
   assert(grp_id >= 0);

   int num_ref_ports_;
   hdf5_utils::ReadAttribute(grp_id, "number_of_ports", num_ref_ports_);
   assert(num_ref_ports_ >= num_ref_ports);

   std::string dset_name;
   int c1, c2, a1, a2;
   for (int p = 0; p < num_ref_ports; p++)
   {
      topol_handler->GetRefPortInfo(p, c1, c2, a1, a2);
      dset_name = topol_handler->GetComponentName(c1) + ":" + topol_handler->GetComponentName(c2);
      dset_name += "-" + std::to_string(a1) + ":" + std::to_string(a2);
      hdf5_utils::ReadDataset(grp_id, dset_name, *port[p]);
   }

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}