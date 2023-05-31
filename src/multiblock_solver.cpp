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

   for (int k = 0; k < bdr_markers.Size(); k++)
      delete bdr_markers[k];

   delete rom_handler;
   delete topol_handler;
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
   }

   // rom inputs.
   use_rom = config.GetOption<bool>("main/use_rom", false);
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

   unified_paraview = config.GetOption<bool>("visualization/unified_paraview", true);

   if (unified_paraview)
      InitUnifiedParaview(file_prefix);
   else
      InitIndividualParaview(file_prefix);
}

void MultiBlockSolver::InitIndividualParaview(const std::string& file_prefix)
{
   paraviewColls.SetSize(numSub);

   for (int m = 0; m < numSub; m++) {
      ostringstream oss;
      // Each subdomain needs to be save separately.
      oss << file_prefix << "_" << std::to_string(m);

      paraviewColls[m] = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
      paraviewColls[m]->SetLevelsOfDetail(order);
      paraviewColls[m]->SetHighOrderOutput(true);
      paraviewColls[m]->SetPrecision(8);

      paraviewColls[m]->RegisterField("solution", us[m]);
      paraviewColls[m]->SetOwnData(false);
   }
}

void MultiBlockSolver::InitUnifiedParaview(const std::string& file_prefix)
{
   assert(pmesh != NULL);
   assert(global_us_visual != NULL);
   // TODO: For truly bottom-up case, when the parent mesh does not exist?
   mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
   paraviewColls.SetSize(1);

   paraviewColls[0] = new ParaViewDataCollection(file_prefix.c_str(), pmesh);
   paraviewColls[0]->SetLevelsOfDetail(order);
   paraviewColls[0]->SetHighOrderOutput(true);
   paraviewColls[0]->SetPrecision(8);

   paraviewColls[0]->RegisterField("solution", global_us_visual);
   paraviewColls[0]->SetOwnData(false);
}

void MultiBlockSolver::SaveVisualization()
{
   if (!save_visual) return;

   if (unified_paraview)
   {
      mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
      topol_handler->TransferToGlobal(us, global_us_visual);
   }

   for (int m = 0; m < paraviewColls.Size(); m++)
      paraviewColls[m]->Save();
};

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
