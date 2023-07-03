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
#include "component_topology_handler.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

MultiBlockSolver::MultiBlockSolver()
{
   ParseInputs();

   TopologyData topol_data;
   switch (topol_mode)
   {
      case TopologyHandlerMode::SUBMESH:
      {
         topol_handler = new SubMeshTopologyHandler();
         break;
      }
      case TopologyHandlerMode::COMPONENT:
      {
         topol_handler = new ComponentTopologyHandler();
         break;
      }
      default:
      {
         mfem_error("Unknown topology handler mode!\n");
         break;
      }
   }
   topol_handler->ExportInfo(meshes, topol_data);
   
   // Receive topology info
   numSub = topol_data.numSub;
   dim = topol_data.dim;
   global_bdr_attributes = *(topol_data.global_bdr_attributes);
   numBdr = global_bdr_attributes.Size();
}

MultiBlockSolver::~MultiBlockSolver()
{
   delete U;
   delete RHS;

   if (save_visual)
      for (int k = 0; k < paraviewColls.Size(); k++) delete paraviewColls[k];

   for (int k = 0; k < us.Size(); k++) delete us[k];
   for (int k = 0; k < fes.Size(); k++) delete fes[k];
   for (int k = 0; k < fec.Size(); k++) delete fec[k];

   for (int k = 0; k < global_fes.Size(); k++) delete global_fes[k];
   for (int k = 0; k < global_us_visual.Size(); k++) delete global_us_visual[k];

   for (int k = 0; k < bdr_markers.Size(); k++)
      delete bdr_markers[k];

   delete rom_handler;
   delete topol_handler;

   for (int c = 0; c < comp_mats.Size(); c++)
      delete comp_mats[c];

   for (int c = 0; c < bdr_mats.Size(); c++)
   {
      for (int b = 0; b < bdr_mats[c]->Size(); b++)
         delete (*bdr_mats[c])[b];

      delete bdr_mats[c];
   }

   for (int p = 0; p < port_mats.Size(); p++)
   {
      for (int i = 0; i < port_mats[p]->NumRows(); i++)
         for (int j = 0; j < port_mats[p]->NumCols(); j++)
            delete (*port_mats[p])(i,j);

      delete port_mats[p];
   }
}

void MultiBlockSolver::ParseInputs()
{
   std::string topol_str = config.GetOption<std::string>("mesh/type", "submesh");
   if (topol_str == "submesh")
   {
      topol_mode = TopologyHandlerMode::SUBMESH;
   }
   else if (topol_str == "component-wise")
   {
      topol_mode = TopologyHandlerMode::COMPONENT;
   }
   else
   {
      printf("%s\n", topol_str.c_str());
      mfem_error("Unknown topology handler mode!\n");
   }

   order = config.GetOption<int>("discretization/order", 1);
   full_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);

   // solver option;
   use_amg = config.GetOption<bool>("solver/use_amg", true);

   save_visual = config.GetOption<bool>("visualization/enabled", false);
   if (save_visual)
   {
      // Default file path if no input file name is provided.
      visual_dir = config.GetOption<std::string>("visualization/file_path/directory", ".");
      visual_prefix = config.GetOption<std::string>("visualization/file_path/prefix", "paraview_output");

      unified_paraview = config.GetOption<bool>("visualization/unified_paraview", false);
      if (unified_paraview)
      {
         if (topol_mode == TopologyHandlerMode::COMPONENT)
            mfem_error("ComponentTopologyHandler does not yet support unified paraview!\n");
      }
      else
      {
         visual_offset = config.GetOption<int>("visualization/domain_offset", 0);
         visual_freq = config.GetOption<int>("visualization/domain_frequency", 1);
      } 
   }

   // rom inputs.
   use_rom = config.GetOption<bool>("main/use_rom", false);

   // save solution if single run.
   save_sol = config.GetOption<bool>("save_solution/enabled", false);
   if (save_sol)
   {
      // Default file path if no input file name is provided.
      sol_dir = config.GetOption<std::string>("save_solution/file_path/directory", ".");
      sol_prefix = config.GetOption<std::string>("save_solution/file_path/prefix", "solution");
   }
}

void MultiBlockSolver::GetVariableVector(const int &var_idx, BlockVector &global, BlockVector &var)
{
   assert((var_idx >= 0) && (var_idx < num_var));
   assert(global.NumBlocks() == (num_var * numSub));
   assert(var.NumBlocks() == (numSub));

   for (int m = 0; m < numSub; m++)
   {
      int g_idx = num_var * m + var_idx;
      assert(var.BlockSize(m) == global.BlockSize(g_idx));

      Vector tmp;
      var.GetBlockView(m, tmp);
      tmp = global.GetBlock(g_idx);
   }
}

void MultiBlockSolver::SetVariableVector(const int &var_idx, BlockVector &var, BlockVector &global)
{
   assert((var_idx >= 0) && (var_idx < num_var));
   assert(global.NumBlocks() == (num_var * numSub));
   assert(var.NumBlocks() == (numSub));

   for (int m = 0; m < numSub; m++)
   {
      int g_idx = num_var * m + var_idx;
      assert(var.BlockSize(m) == global.BlockSize(g_idx));

      Vector tmp;
      global.GetBlockView(g_idx, tmp);
      tmp = var.GetBlock(m);
   }
}

void MultiBlockSolver::SetupBCVariables()
{
   // Set up boundary markers.
   int max_bdr_attr = -1;
   for (int m = 0; m < numSub; m++)
   {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
   }

   // TODO: technically this should be Array<Array2D<int>*> for each meshes.
   // Running with MFEM debug version will lead to error when assembling boundary integrators.
   bdr_markers.SetSize(global_bdr_attributes.Size());
   for (int k = 0; k < bdr_markers.Size(); k++) {
      int bdr_attr = global_bdr_attributes[k];
      assert((bdr_attr > 0) && (bdr_attr <= max_bdr_attr));
      bdr_markers[k] = new Array<int>(max_bdr_attr);
      (*bdr_markers[k]) = 0;
      (*bdr_markers[k])[bdr_attr-1] = 1;
   }
}

void MultiBlockSolver::AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
                                                FiniteElementSpace *fes1,
                                                FiniteElementSpace *fes2,
                                                InterfaceNonlinearFormIntegrator *interface_integ,
                                                Array<InterfaceInfo> *interface_infos,
                                                Array2D<SparseMatrix*> &mats)
{
   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);
      
      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *fe1, *fe2;
      Array<Array<int> *> vdofs(2);
      vdofs[0] = new Array<int>;
      vdofs[1] = new Array<int>;

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
         fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         interface_integ->AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);

         for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
               mats(i, j)->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}

void MultiBlockSolver::AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
   FiniteElementSpace *trial_fes1, FiniteElementSpace *trial_fes2,
   FiniteElementSpace *test_fes1, FiniteElementSpace *test_fes2,
   InterfaceNonlinearFormIntegrator *interface_integ,
   Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats)
{
   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);
      
      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;
      Array<Array<int> *> test_vdofs(2), trial_vdofs(2);
      trial_vdofs[0] = new Array<int>;
      trial_vdofs[1] = new Array<int>;
      test_vdofs[0] = new Array<int>;
      test_vdofs[1] = new Array<int>;

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         trial_fes1->GetElementVDofs(tr1->Elem1No, *trial_vdofs[0]);
         trial_fes2->GetElementVDofs(tr2->Elem1No, *trial_vdofs[1]);
         test_fes1->GetElementVDofs(tr1->Elem1No, *test_vdofs[0]);
         test_fes2->GetElementVDofs(tr2->Elem1No, *test_vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         trial_fe1 = trial_fes1->GetFE(tr1->Elem1No);
         trial_fe2 = trial_fes2->GetFE(tr2->Elem1No);
         test_fe1 = test_fes1->GetFE(tr1->Elem1No);
         test_fe2 = test_fes2->GetFE(tr2->Elem1No);

         interface_integ->AssembleInterfaceMatrix(
            *trial_fe1, *trial_fe2, *test_fe1, *test_fe2, *tr1, *tr2, elemmats);

         for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
               mats(i, j)->AddSubMatrix(*test_vdofs[i], *trial_vdofs[j], *elemmats(i,j), skip_zeros);
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}

void MultiBlockSolver::GetComponentFESpaces(Array<FiniteElementSpace *> &comp_fes)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(fec.Size() == num_var);
   assert(vdim.Size() == num_var);

   // Component domain system
   const int num_comp = topol_handler->GetNumComponents();
   comp_fes.SetSize(num_var * num_comp);
   comp_fes = NULL;

   for (int c = 0, idx = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      for (int v = 0; v < num_var; v++, idx++)
         comp_fes[idx] = new FiniteElementSpace(comp, fec[v], vdim[v]);
   }
}

void MultiBlockSolver::AllocateROMElements()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   const int num_comp = topol_handler->GetNumComponents();
   const int num_ref_ports = topol_handler->GetNumRefPorts();

   comp_mats.SetSize(num_comp);
   comp_mats = NULL;

   bdr_mats.SetSize(num_comp);
   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      bdr_mats[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats[c]) = NULL;
   }
   port_mats.SetSize(num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
   {
      port_mats[p] = new Array2D<SparseMatrix *>(2,2);
      (*port_mats[p]) = NULL;
   }
}

void MultiBlockSolver::BuildROMElements()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   // Component domain system
   const int num_comp = topol_handler->GetNumComponents();
   Array<FiniteElementSpace *> fes_comp;
   GetComponentFESpaces(fes_comp);

   BuildCompROMElement(fes_comp);

   // Boundary penalty matrixes
   BuildBdrROMElement(fes_comp);

   // Port penalty matrixes
   BuildInterfaceROMElement(fes_comp);

   for (int k = 0 ; k < fes_comp.Size(); k++) delete fes_comp[k];
}

void MultiBlockSolver::SaveROMElements(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   // components + boundary
   SaveCompBdrROMElement(file_id);

   // (reference) ports
   SaveInterfaceROMElement(file_id);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

void MultiBlockSolver::SaveCompBdrROMElement(hid_t &file_id)
{
   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, "components", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_mats.Size() == num_comp);
   assert(bdr_mats.Size() == num_comp);

   hdf5_utils::WriteAttribute(grp_id, "number_of_components", num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      hid_t comp_grp_id;
      comp_grp_id = H5Gcreate(grp_id, std::to_string(c).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      hdf5_utils::WriteSparseMatrix(comp_grp_id, "domain", comp_mats[c]);

      // boundaries are saved for each component group.
      SaveBdrROMElement(comp_grp_id, c);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

void MultiBlockSolver::SaveBdrROMElement(hid_t &comp_grp_id, const int &comp_idx)
{
   assert(comp_grp_id >= 0);
   herr_t errf;
   hid_t bdr_grp_id;
   bdr_grp_id = H5Gcreate(comp_grp_id, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(bdr_grp_id >= 0);

   const int num_bdr = bdr_mats[comp_idx]->Size();
   Mesh *comp = topol_handler->GetComponentMesh(comp_idx);
   assert(num_bdr == comp->bdr_attributes.Size());

   hdf5_utils::WriteAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);
   
   Array<SparseMatrix *> *bdr_mat_c = bdr_mats[comp_idx];
   for (int b = 0; b < num_bdr; b++)
      hdf5_utils::WriteSparseMatrix(bdr_grp_id, std::to_string(b), (*bdr_mat_c)[b]);

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void MultiBlockSolver::SaveInterfaceROMElement(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gcreate(file_id, "ports", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   assert(port_mats.Size() == num_ref_ports);

   hdf5_utils::WriteAttribute(grp_id, "number_of_ports", num_ref_ports);
   
   for (int p = 0; p < num_ref_ports; p++)
   {
      assert(port_mats[p]->NumRows() == 2);
      assert(port_mats[p]->NumCols() == 2);

      hid_t port_grp_id;
      port_grp_id = H5Gcreate(grp_id, std::to_string(p).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(port_grp_id >= 0);

      Array2D<SparseMatrix *> *port_mat = port_mats[p];
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            std::string dset_name = std::to_string(i) + std::to_string(j);
            hdf5_utils::WriteSparseMatrix(port_grp_id, dset_name, (*port_mat)(i,j));
         }
      
      errf = H5Gclose(port_grp_id);
      assert(errf >= 0);
   }  // for (int p = 0; p < num_ref_ports; p++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

void MultiBlockSolver::LoadROMElements(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   // components
   LoadCompBdrROMElement(file_id);

   // (reference) ports
   LoadInterfaceROMElement(file_id);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

void MultiBlockSolver::LoadCompBdrROMElement(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   int num_comp;
   hdf5_utils::ReadAttribute(grp_id, "number_of_components", num_comp);
   assert(num_comp == topol_handler->GetNumComponents());
   assert(comp_mats.Size() == num_comp);
   assert(bdr_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, std::to_string(c).c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      comp_mats[c] = hdf5_utils::ReadSparseMatrix(comp_grp_id, "domain");

      // boundary
      LoadBdrROMElement(comp_grp_id, c);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void MultiBlockSolver::LoadBdrROMElement(hid_t &comp_grp_id, const int &comp_idx)
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
   assert(num_bdr = bdr_mats[comp_idx]->Size());

   Array<SparseMatrix *> *bdr_mat_c = bdr_mats[comp_idx];
   for (int b = 0; b < num_bdr; b++)
      (*bdr_mat_c)[b] = hdf5_utils::ReadSparseMatrix(bdr_grp_id, std::to_string(b));

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void MultiBlockSolver::LoadInterfaceROMElement(hid_t &file_id)
{
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
   assert(grp_id >= 0);

   int num_ref_ports;
   hdf5_utils::ReadAttribute(grp_id, "number_of_ports", num_ref_ports);
   assert(num_ref_ports == topol_handler->GetNumRefPorts());
   assert(port_mats.Size() == num_ref_ports);

   for (int p = 0; p < num_ref_ports; p++)
   {
      assert(port_mats[p]->NumRows() == 2);
      assert(port_mats[p]->NumCols() == 2);

      hid_t port_grp_id;
      port_grp_id = H5Gopen2(grp_id, std::to_string(p).c_str(), H5P_DEFAULT);
      assert(port_grp_id >= 0);

      Array2D<SparseMatrix *> *port_mat = port_mats[p];
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            std::string dset_name = std::to_string(i) + std::to_string(j);
            (*port_mat)(i,j) = hdf5_utils::ReadSparseMatrix(port_grp_id, dset_name);
         }
      
      errf = H5Gclose(port_grp_id);
      assert(errf >= 0);
   }  // for (int p = 0; p < num_ref_ports; p++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void MultiBlockSolver::AssembleROM()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   const Array<int> *rom_block_offsets = rom_handler->GetBlockOffsets();
   BlockMatrix *romMat = new BlockMatrix(*rom_block_offsets);

   // component domain matrix.
   for (int m = 0; m < numSub; m++)
   {
      int c_type = topol_handler->GetMeshType(m);
      int num_basis = rom_handler->GetNumBasis(c_type);

      if (romMat->IsZeroBlock(m, m))
         romMat->SetBlock(m, m, new SparseMatrix(num_basis));

      romMat->GetBlock(m,m) += *(comp_mats[c_type]);

      // boundary matrixes of each component.
      Array<int> *bdr_c2g = topol_handler->GetBdrAttrComponentToGlobalMap(m);
      Array<SparseMatrix *> *bdr_mat = bdr_mats[c_type];

      for (int b = 0; b < bdr_c2g->Size(); b++)
      {
         int global_idx = global_bdr_attributes.Find((*bdr_c2g)[b]);
         if (global_idx < 0) continue;
         if (!BCExistsOnBdr(global_idx)) continue;

         romMat->GetBlock(m,m) += *(*bdr_mat)[b];
      }
   }

   // interface matrixes.
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);
      const int p_type = topol_handler->GetPortType(p);
      Array2D<SparseMatrix *> *port_mat = port_mats[p_type];

      const int m1 = pInfo->Mesh1;
      const int m2 = pInfo->Mesh2;
      const int c1 = topol_handler->GetMeshType(m1);
      const int c2 = topol_handler->GetMeshType(m2);
      const int num_basis1 = rom_handler->GetNumBasis(c1);
      const int num_basis2 = rom_handler->GetNumBasis(c2);

      Array<int> midx(2), num_basis(2);
      midx[0] = m1;
      midx[1] = m2;
      num_basis[0] = num_basis1;
      num_basis[1] = num_basis2;

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            if (romMat->IsZeroBlock(midx[i], midx[j]))
               romMat->SetBlock(midx[i], midx[j],
                                 new SparseMatrix(num_basis[i], num_basis[j]));
            romMat->GetBlock(midx[i], midx[j]) += *((*port_mat)(i, j));
         }
   }

   romMat->Finalize();
   rom_handler->LoadOperator(romMat);
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

   if (unified_paraview)
      InitUnifiedParaview(file_prefix);
   else
      InitIndividualParaview(file_prefix);
}

void MultiBlockSolver::InitIndividualParaview(const std::string& file_prefix)
{
   assert(var_names.size() == num_var);
   assert((visual_offset >= 0) && (visual_offset < numSub));
   assert((visual_freq > 0) && (visual_freq < numSub));
   paraviewColls.SetSize(numSub);
   paraviewColls = NULL;

   for (int m = 0; m < numSub; m++) {
      if ((m < visual_offset) || (m % visual_freq != 0)) continue;

      ostringstream oss;
      // Each subdomain needs to be save separately.
      oss << file_prefix << "_" << std::to_string(m);

      paraviewColls[m] = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
      paraviewColls[m]->SetLevelsOfDetail(order);
      paraviewColls[m]->SetHighOrderOutput(true);
      paraviewColls[m]->SetPrecision(8);

      for (int v = 0, idx = m * num_var; v < num_var; v++, idx++)
         paraviewColls[m]->RegisterField(var_names[v].c_str(), us[idx]);
      paraviewColls[m]->SetOwnData(false);
   }
}

void MultiBlockSolver::InitUnifiedParaview(const std::string& file_prefix)
{
   pmesh = topol_handler->GetGlobalMesh();

   global_fes.SetSize(num_var);
   global_us_visual.SetSize(num_var);
   for (int v = 0; v < num_var; v++)
   {
      global_fes[v] = new FiniteElementSpace(pmesh, fec[v], vdim[v]);
      global_us_visual[v] = new GridFunction(global_fes[v]);   
   }

   // TODO: For truly bottom-up case, when the parent mesh does not exist?
   mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
   paraviewColls.SetSize(1);

   paraviewColls[0] = new ParaViewDataCollection(file_prefix.c_str(), pmesh);
   paraviewColls[0]->SetLevelsOfDetail(order);
   paraviewColls[0]->SetHighOrderOutput(true);
   paraviewColls[0]->SetPrecision(8);

   for (int v = 0; v < num_var; v++)
      paraviewColls[0]->RegisterField(var_names[v].c_str(), global_us_visual[v]);
   paraviewColls[0]->SetOwnData(false);
}

void MultiBlockSolver::SaveVisualization()
{
   if (!save_visual) return;

   if (unified_paraview)
   {
      mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
      topol_handler->TransferToGlobal(us, global_us_visual, num_var);
   }
   else
   {
      assert((visual_offset >= 0) && (visual_offset < numSub));
      assert((visual_freq > 0) && (visual_freq < numSub));
   }

   for (int m = 0; m < paraviewColls.Size(); m++)
   {
      if ((!unified_paraview) && ((m < visual_offset) || (m % visual_freq != 0))) continue;
      assert(paraviewColls[m]);
      paraviewColls[m]->Save();
   }
};

void MultiBlockSolver::SaveSolution(std::string filename)
{
   if (!save_sol) return;

   if (filename == "")
   {
      filename = sol_dir + "/" + sol_prefix + ".h5";
   }
   printf("Saving the solution file %s ...", filename.c_str());

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   // TODO: currently we only need solution vector. But we can add more data as we need.
   hdf5_utils::WriteDataset(file_id, "solution", *U);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   printf("Done!\n");
}

void MultiBlockSolver::LoadSolution(const std::string &filename)
{
   // solution vector U must be instantiated beforehand.
   assert(U);
   printf("Loading the solution file %s ...", filename.c_str());
   if (!FileExists(filename))
      mfem_error("file does not exist!\n");
   

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   // Check the size of the solution in the file.
   hsize_t *dims = NULL;
   int ndims = hdf5_utils::GetDatasetSize(file_id, "solution", dims);
   bool size_match = (ndims == 1) && (dims[0] == U->Size());

   // Read the solution.
   if (size_match)
      hdf5_utils::ReadDataset(file_id, "solution", *U);

   // TODO: currently we only need solution vector. But we can add more data as we need.
   errf = H5Fclose(file_id);
   assert(errf >= 0);
   if (size_match)
      printf("Done!\n");
   else
      mfem_error("solution size does not match!\n");
}

void MultiBlockSolver::CopySolution(BlockVector *input_sol)
{
   assert(input_sol->NumBlocks() == U->NumBlocks());
   for (int b = 0; b < U->NumBlocks(); b++)
      assert(input_sol->BlockSize(b) == U->BlockSize(b));
   *U = *input_sol;
}

void MultiBlockSolver::InitROMHandler()
{
   std::string rom_handler_str = config.GetOption<std::string>("model_reduction/rom_handler_type", "base");
   bool separate_variable_basis = config.GetOption<bool>("model_reduction/separate_variable_basis", false);

   Array<int> rom_vdim;
   if (separate_variable_basis)
      rom_vdim = vdim;
   else
   {
      rom_vdim.SetSize(1);
      rom_vdim = vdim.Sum();
   }

   if (rom_handler_str == "base")
   {
      rom_handler = new ROMHandler(topol_handler, rom_vdim, num_vdofs);
   }
   else if (rom_handler_str == "mfem")
   {
      rom_handler = new MFEMROMHandler(topol_handler, rom_vdim, num_vdofs);
   }
   else
   {
      mfem_error("Unknown ROM handler type!\n");
   }
}

void MultiBlockSolver::PrepareSnapshots(BlockVector* &U_snapshots, std::vector<std::string> &basis_tags)
{
   // TODO: get offsets set by rom_handler.
   U_snapshots = new BlockVector(U->GetData(), domain_offsets); // View vector for U.
   rom_handler->GetBasisTags(basis_tags);
   assert(U_snapshots->NumBlocks() == basis_tags.size());
}

void MultiBlockSolver::ProjectRHSOnReducedBasis()
{
   BlockVector RHS_domain(RHS->GetData(), domain_offsets); // View vector for RHS.
   rom_handler->ProjectRHSOnReducedBasis(&RHS_domain);
}

void MultiBlockSolver::SolveROM()
{
   BlockVector U_domain(U->GetData(), domain_offsets); // View vector for U.
   rom_handler->Solve(&U_domain);
}

void MultiBlockSolver::ComputeSubdomainErrorAndNorm(GridFunction *fom_sol, GridFunction *rom_sol, double &error, double &norm)
{
   assert(fom_sol && rom_sol);
   const int vec_dim = fom_sol->FESpace()->GetVDim();

   if (vec_dim == 1)
   {
      ConstantCoefficient zero(0.0);
      GridFunctionCoefficient rom_sol_coeff(rom_sol);

      norm = fom_sol->ComputeLpError(2, zero);
      error = fom_sol->ComputeLpError(2, rom_sol_coeff);
   }
   else
   {
      Vector zero_v(vec_dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero(zero_v);
      VectorGridFunctionCoefficient rom_sol_coeff(rom_sol);

      norm = fom_sol->ComputeLpError(2, zero);
      error = fom_sol->ComputeLpError(2, rom_sol_coeff);
   }
}

double MultiBlockSolver::ComputeRelativeError(Array<GridFunction *> fom_sols, Array<GridFunction *> rom_sols)
{
   assert(fom_sols.Size() == (num_var * numSub));
   assert(rom_sols.Size() == (num_var * numSub));

   Vector norm(num_var), error(num_var);
   norm = 0.0; error = 0.0;
   for (int m = 0, idx = 0; m < numSub; m++)
   {
      for (int v = 0; v < num_var; v++, idx++)
      {
         assert(fom_sols[idx] && rom_sols[idx]);
         double var_norm, var_error;
         ComputeSubdomainErrorAndNorm(fom_sols[idx], rom_sols[idx], var_error, var_norm);
         norm[v] += var_norm * var_norm;
         error[v] += var_error * var_error;
      }
   }

   for (int v = 0; v < num_var; v++)
   {
      norm[v] = sqrt(norm[v]);
      error[v] = sqrt(error[v]);
      error[v] /= norm[v];
      printf("Variable %d relative error: %.5E\n", v, error[v]);
   }

   return error.Max();
}

double MultiBlockSolver::CompareSolution(BlockVector &test_U)
{
   assert(test_U.NumBlocks() == U->NumBlocks());
   for (int b = 0; b < U->NumBlocks(); b++)
      assert(test_U.BlockSize(b) == U->BlockSize(b));

   Array<GridFunction *> test_us;
   test_us.SetSize(num_var * numSub);
   for (int k = 0; k < test_us.Size(); k++)
   {
      test_us[k] = new GridFunction(fes[k], test_U.GetBlock(k), 0);

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      test_us[k]->SetTrueVector();
   }

   // Compare the solution.
   // Maximum L2-error / L2-norm over all variables.
   double error = ComputeRelativeError(us, test_us);
   printf("Relative error: %.5E\n", error);

   DeletePointers(test_us);

   return error;
}