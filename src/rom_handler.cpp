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

#include "etc.hpp"
#include "input_parser.hpp"
#include "rom_handler.hpp"
#include "linalg_utils.hpp"
#include "hdf5_utils.hpp"
#include "block_smoother.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;

namespace mfem
{

ROMHandler::ROMHandler(TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_num_vdofs)
   : topol_handler(input_topol),
     numSub(input_topol->GetNumSubdomains()),
     vdim(input_vdim),
     fom_num_vdofs(input_num_vdofs),
     basis_loaded(false),
     operator_loaded(false)
{
   num_var = vdim.Size();
   udim = vdim.Sum();
   assert(fom_num_vdofs.Size() == (num_var * numSub));

   ParseInputs();

   SetBlockSizes();

   AllocROMMat();
}

ROMHandler::~ROMHandler()
{
   DeletePointers(carom_mats);
   delete reduced_rhs;
   delete romMat_inv;
}

void ROMHandler::ParseInputs()
{
   assert(numSub > 0);
   assert(topol_handler != NULL);

   std::string mode_str = config.GetOption<std::string>("main/mode", "");
   if (mode_str == "single_run")
   {
      mode = ROMHandlerMode::SINGLE_RUN;
   }
   else if (mode_str == "sample_generation")
   {
      mode = ROMHandlerMode::SAMPLE_GENERATION;
   }
   else if (mode_str == "build_rom")
   {
      mode = ROMHandlerMode::BUILD_ROM;
   }
   else
   {
      mfem_error("Unknown main mode!\n");
   }

   if ((mode == ROMHandlerMode::SAMPLE_GENERATION) || (mode == ROMHandlerMode::BUILD_ROM))
   {
      sample_dir = config.GetOption<std::string>("sample_generation/file_path/directory", ".");
      sample_prefix = config.GetOption<std::string>("sample_generation/file_path/prefix", "");
      if (sample_prefix == "")
         sample_prefix = config.GetRequiredOption<std::string>("parameterized_problem/name");
   }

   num_basis = config.GetRequiredOption<Array<int>>("model_reduction/number_of_basis");
   assert(num_basis.Size() > 0);
   for (int k = 0; k < num_basis.Size(); k++) assert(num_basis[k] > 0);

   basis_file_exists = config.GetOption<bool>("model_reduction/basis/file_exists", false);
   basis_prefix = config.GetOption<std::string>("model_reduction/basis/prefix", "basis");

   std::string save_op_str = config.GetOption<std::string>("model_reduction/save_operator/level", "none");
   if (save_op_str == "none")
   {
      save_operator = ROMBuildingLevel::NONE;
   }
   else
   {
      operator_prefix = config.GetRequiredOption<std::string>("model_reduction/save_operator/prefix");
      if (save_op_str == "global")
      {
         save_operator = ROMBuildingLevel::GLOBAL;
      }
      else if (save_op_str == "component")
      {
         save_operator = ROMBuildingLevel::COMPONENT;
      }
      else
      {
         mfem_error("Unknown ROM building level!\n");
      }
   }

   std::string train_mode_str = config.GetOption<std::string>("model_reduction/subdomain_training", "individual");
   if (train_mode_str == "individual")
   {
      train_mode = TrainMode::INDIVIDUAL;
      num_basis_sets = numSub;
   }
   else if (train_mode_str == "universal")
   {
      train_mode = TrainMode::UNIVERSAL;
      num_basis_sets = topol_handler->GetNumComponents();
   }
   else
   {
      mfem_error("Unknown subdomain training mode!\n");
   }
   
   // Adjust num_basis according to num_basis_sets.
   if (num_basis.Size() != num_basis_sets)
   {
      // Only take uniform number of basis for all components.
      assert(num_basis.Size() == 1);
      const int tmp = num_basis[0];
      num_basis.SetSize(num_basis_sets);
      num_basis = tmp;
   }

   // component_sampling = config.GetOption<bool>("sample_generation/component_sampling", false);
   // if ((train_mode == TrainMode::INDIVIDUAL) && component_sampling)
   //    mfem_error("Component sampling is only supported with universal basis!\n");

   // std::string proj_mode_str = config.GetOption<std::string>("model_reduction/projection_type", "lspg");
   // if (proj_mode_str == "galerkin")
   // {
   //    proj_mode = ProjectionMode::GALERKIN;
   // }
   // else if (proj_mode_str == "lspg")
   // {
   //    proj_mode = ProjectionMode::LSPG;
   // }
   // else
   // {
   //    mfem_error("Unknown projection mode!\n");
   // }

   // if (proj_mode == ProjectionMode::LSPG)
   //    save_lspg_basis = config.GetOption<bool>("model_reduction/lspg/save_lspg_basis", true);

   max_num_snapshots = config.GetOption<int>("sample_generation/maximum_number_of_snapshots", 100);
   update_right_SV = config.GetOption<bool>("model_reduction/svd/update_right_sv", false);

   save_sv = config.GetOption<bool>("model_reduction/svd/save_spectrum", false);

   save_basis_visual = config.GetOption<bool>("model_reduction/visualization/enabled", false);
}

void ROMHandler::FormReducedBasis()
{
   std::string basis_name, basis_tag;
   
   for (int c = 0; c < num_basis_sets; c++)
   {
      int basis_dim = -1;
      switch (train_mode)
      {
         case (INDIVIDUAL):
         {
            basis_tag = GetBasisTag(c);
            basis_dim = fom_num_vdofs[c];
            break;
         }
         case (UNIVERSAL):
         {
            basis_tag = GetBasisTagForComponent(c);
            for (int m = 0; m < numSub; m++)
               if (topol_handler->GetMeshType(m) == c)
               { basis_dim = fom_num_vdofs[m]; break; }
            break;
         }
      }
      basis_name = basis_prefix + "_" + basis_tag;
      assert(basis_dim > 0);

      rom_options = new CAROM::Options(basis_dim, max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);

      std::string filename = sample_dir + "/" + sample_prefix + "_sample";
      filename += "_" + basis_tag + "_snapshot";
      basis_generator->loadSamples(filename,"snapshot");

      basis_generator->endSamples(); // save the merged basis file
      SaveSV(basis_name, c);

      delete basis_generator;
      delete rom_options;
   }  // for (int c = 0; c < num_basis_sets; c++)
}

void ROMHandler::LoadReducedBasis()
{
   if (basis_loaded) return;

   std::string basis_name;
   int numRowRB, numColumnRB;

   carom_spatialbasis.SetSize(num_basis_sets);
   for (int k = 0; k < num_basis_sets; k++)
   {
      basis_name = basis_prefix + "_" + GetBasisTagForComponent(k);
      basis_reader = new CAROM::BasisReader(basis_name);

      carom_spatialbasis[k] = basis_reader->getSpatialBasis(0.0, num_basis[k]);
      numRowRB = carom_spatialbasis[k]->numRows();
      numColumnRB = carom_spatialbasis[k]->numColumns();
      printf("spatial basis-%d dimension is %d x %d\n", k, numRowRB, numColumnRB);

      delete basis_reader;
   }

   basis_loaded = true;
}

int ROMHandler::GetBasisIndexForSubdomain(const int &subdomain_index)
{
   int idx = -1;
   switch (train_mode)
   {
      case TrainMode::UNIVERSAL:
      { idx = topol_handler->GetMeshType(subdomain_index); break; }
      case TrainMode::INDIVIDUAL:
      { idx = subdomain_index; break; }
      default:
      { mfem_error("LoadBasis: unknown TrainMode!\n"); break; }
   }  // switch (train_mode)

   assert(idx >= 0);
   return idx;
}

void ROMHandler::GetBasis(const int &basis_index, const CAROM::Matrix* &basis)
{
   assert(num_basis_sets > 0);
   assert((basis_index >= 0) && (basis_index < num_basis_sets));

   basis = carom_spatialbasis[basis_index];
}

void ROMHandler::GetBasisOnSubdomain(const int &subdomain_index, const CAROM::Matrix* &basis)
{
   assert(basis_loaded);

   int idx = GetBasisIndexForSubdomain(subdomain_index);
   GetBasis(idx, basis);
   return;
}

void ROMHandler::ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats)
{
   printf("Project Operators on reduced basis.\n");
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   // Form inverse matrix
   for (int i = 0; i < numSub; i++)
   {
      int num_rows = rom_block_offsets[i+1] - rom_block_offsets[i];
      int basis_i = GetBasisIndexForSubdomain(i);
      for (int j = 0; j < numSub; j++)
      {
         int num_cols = rom_block_offsets[j+1] - rom_block_offsets[j];
         int basis_j = GetBasisIndexForSubdomain(j);

         assert(mats(i,j) != NULL);

         carom_mats(i,j) = new CAROM::Matrix(num_rows, num_cols, false);
         ProjectOperatorOnReducedBasis(basis_i, basis_j, mats(i,j), carom_mats(i,j));

         CAROM::SetBlock(*carom_mats(i,j), rom_block_offsets[i], rom_block_offsets[i+1],
                         rom_block_offsets[j], rom_block_offsets[j+1], *romMat_inv);
      }
   }

   romMat_inv->inverse();

   operator_loaded = true;
   if (save_operator == ROMBuildingLevel::GLOBAL) romMat_inv->write(operator_prefix);
}

void ROMHandler::SetBlockSizes()
{
   // TODO: non-uniform subdomain cases.
   rom_block_offsets.SetSize(numSub+1);
   rom_block_offsets = 0;

   for (int k = 1; k <= numSub; k++)
   {
      int c = topol_handler->GetMeshType(k-1);
      rom_block_offsets[k] = num_basis[c];
   }
   rom_block_offsets.PartialSum();
}

void ROMHandler::AllocROMMat()
{
   assert(rom_block_offsets.Size() == (numSub+1));

   carom_mats.SetSize(numSub, numSub);
   carom_mats = NULL;
   // TODO: parallelization.
   romMat_inv = new CAROM::Matrix(rom_block_offsets.Last(), rom_block_offsets.Last(), false);
}

void ROMHandler::ProjectRHSOnReducedBasis(const BlockVector* RHS)
{
   assert(RHS->NumBlocks() == numSub);

   printf("Project RHS on reduced basis.\n");
   reduced_rhs = new CAROM::Vector(rom_block_offsets.Last(), false);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      assert(RHS->GetBlock(i).Size() == fom_num_vdofs[i]);

      const CAROM::Matrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);

      CAROM::Vector block_rhs_carom(RHS->GetBlock(i).GetData(), RHS->GetBlock(i).Size(), true, false);
      CAROM::Vector *block_reduced_rhs = basis_i->transposeMult(&block_rhs_carom);

      CAROM::SetBlock(*block_reduced_rhs, rom_block_offsets[i], rom_block_offsets[i+1], *reduced_rhs);
   }
}

void ROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);
   assert(operator_loaded);

   printf("Solve ROM.\n");

   CAROM::Vector reduced_sol(rom_block_offsets.Last(), false);
   romMat_inv->mult(*reduced_rhs, reduced_sol);

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      const CAROM::Matrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);
      int c = topol_handler->GetMeshType(i);

      // 23. reconstruct FOM state
      CAROM::Vector block_reduced_sol(num_basis[c], false);
      const int offset = rom_block_offsets[i];
      for (int k = 0; k < num_basis[c]; k++)
         block_reduced_sol(k) = reduced_sol(k + offset);

      // This saves the data automatically to U.
      CAROM::Vector U_block_carom(U->GetBlock(i).GetData(), U->GetBlock(i).Size(), true, false);
      basis_i->mult(block_reduced_sol, U_block_carom);
   }
}

void ROMHandler::ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat, CAROM::Matrix *proj_mat)
{
   assert(proj_mat != NULL);
   assert((i >= 0) && (i < num_basis_sets));
   assert((j >= 0) && (j < num_basis_sets));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   const CAROM::Matrix *basis_i, *basis_j;
   int num_basis_i, num_basis_j;
   GetBasis(i, basis_i);
   num_basis_i = basis_i->numColumns();
   GetBasis(j, basis_j);
   num_basis_j = basis_j->numColumns();

   // TODO: multi-component case.
   proj_mat->setSize(num_basis_i, num_basis_j);
   CAROM::ComputeCtAB(*mat, *basis_j, *basis_i, *proj_mat);
   return;
}

void ROMHandler::LoadOperatorFromFile(const std::string input_prefix)
{
   assert(save_operator == ROMBuildingLevel::GLOBAL);

   std::string prefix;
   if (input_prefix == "")
      prefix = operator_prefix;
   else
      prefix = input_prefix;

   romMat_inv->read(prefix);
   operator_loaded = true;
}

const std::string ROMHandler::GetBasisTagForComponent(const int &comp_idx)
{
   switch (train_mode)
   {
      case (INDIVIDUAL):   { return "dom" + std::to_string(comp_idx); break; }
      case (UNIVERSAL):    { return topol_handler->GetComponentName(comp_idx); break; }
      default:
      {
         mfem_error("ROMHandler::GetBasisTagForComponent - Unknown training mode!\n");
         break;
      }
   }
   return "";
}

const std::string ROMHandler::GetBasisTag(const int &subdomain_index)
{
   switch (train_mode)
   {
      case (INDIVIDUAL):
      {
         return "dom" + std::to_string(subdomain_index);
         break;
      }
      case (UNIVERSAL):
      {
         int c_type = topol_handler->GetMeshType(subdomain_index);
         return topol_handler->GetComponentName(c_type);
         break;
      }
      default:
      {
         mfem_error("ROMHandler::GetBasisTag - Unknown training mode!\n");
         break;
      }
   }
   return "";
}

void ROMHandler::GetBasisTags(std::vector<std::string> &basis_tags)
{
   // TODO: support variable-separate basis case.
   basis_tags.resize(numSub);
   for (int m = 0; m < numSub; m++)
      basis_tags[m] = GetBasisTag(m);
}

void ROMHandler::SaveSV(const std::string& prefix, const int& basis_idx)
{
   if (!save_sv) return;
   assert(basis_generator != NULL);

   const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
   printf("Singular values: ");
   for (int d = 0; d < rom_sv->dim(); d++)
      printf("%.3E\t", rom_sv->item(d));
   printf("\n");

   double coverage = 0.0;
   double total = 0.0;

   for (int d = 0; d < rom_sv->dim(); d++)
   {
      if (d == num_basis[basis_idx]) coverage = total;
      total += rom_sv->item(d);
   }
   if (rom_sv->dim() == num_basis[basis_idx]) coverage = total;
   coverage /= total;
   printf("Coverage: %.7f%%\n", coverage * 100.0);

   // TODO: hdf5 format + parallel case.
   std::string filename = prefix + "_sv.txt";
   CAROM::PrintVector(*rom_sv, filename);
}

/*
   MFEMROMHandler
*/

MFEMROMHandler::MFEMROMHandler(TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_num_vdofs)
   : ROMHandler(input_topol, input_vdim, input_num_vdofs)
{
   romMat = new BlockMatrix(rom_block_offsets);
   romMat->owns_blocks = true;

   carom_mats.SetSize(0, 0);
}

MFEMROMHandler::~MFEMROMHandler()
{
   DeletePointers(spatialbasis);
   delete romMat, romMat_mono;
   delete reduced_rhs;
}

void MFEMROMHandler::LoadReducedBasis()
{
   ROMHandler::LoadReducedBasis();

   spatialbasis.SetSize(carom_spatialbasis.Size());
   spatialbasis = NULL;
   for (int k = 0; k < spatialbasis.Size(); k++)
   {
      assert(carom_spatialbasis[k] != NULL);
      spatialbasis[k] = new DenseMatrix(carom_spatialbasis[k]->numRows(), carom_spatialbasis[k]->numColumns());
      CAROM::CopyMatrix(*carom_spatialbasis[k], *spatialbasis[k]);
   }

   basis_loaded = true;
}

void MFEMROMHandler::GetBasis(const int &basis_index, DenseMatrix* &basis)
{
   assert(num_basis_sets > 0);
   assert((basis_index >= 0) && (basis_index < num_basis_sets));

   basis = spatialbasis[basis_index];
}

void MFEMROMHandler::GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis)
{
   assert(basis_loaded);

   int idx = GetBasisIndexForSubdomain(subdomain_index);
   GetBasis(idx, basis);
}

void MFEMROMHandler::ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats)
{
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);

   if (!basis_loaded) LoadReducedBasis();

   // This is pretty much the same as Assemble().
   // Each basis is applied to the same column blocks.
   int num_basis_i, num_basis_j;
   int basis_i, basis_j;
   for (int i = 0; i < numSub; i++)
   {
      num_basis_i = rom_block_offsets[i+1] - rom_block_offsets[i];
      basis_i = GetBasisIndexForSubdomain(i);

      for (int j = 0; j < numSub; j++)
      {
         num_basis_j = rom_block_offsets[j+1] - rom_block_offsets[j];
         basis_j = GetBasisIndexForSubdomain(j);
         
         SparseMatrix *elemmat = ProjectOperatorOnReducedBasis(basis_i, basis_j, mats(i,j));
         romMat->SetBlock(i, j, elemmat);
      }
   }  // for (int j = 0; j < numSub; j++)

   romMat->Finalize();
   operator_loaded = true;

   romMat_mono = romMat->CreateMonolithic();

   if (save_operator == ROMBuildingLevel::GLOBAL)
   {
      std::string filename = operator_prefix + ".h5";
      hid_t file_id;
      herr_t errf = 0;
      file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      assert(file_id >= 0);

      hdf5_utils::WriteBlockMatrix(file_id, "ROM_matrix", romMat);

      errf = H5Fclose(file_id);
      assert(errf >= 0);
   }
}

void MFEMROMHandler::ProjectRHSOnReducedBasis(const BlockVector* RHS)
{
   assert(RHS->NumBlocks() == numSub);

   printf("Project RHS on reduced basis.\n");
   reduced_rhs = new BlockVector(rom_block_offsets);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      assert(RHS->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);
      basis_i->MultTranspose(RHS->GetBlock(i).GetData(), reduced_rhs->GetBlock(i).GetData());
   }
}

void MFEMROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);
   assert(operator_loaded);

   printf("Solve ROM.\n");
   BlockVector reduced_sol(rom_block_offsets);

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);
   std::string solver_type = config.GetOption<std::string>("model_reduction/solver_type", "cg");
   std::string prec_str = config.GetOption<std::string>("model_reduction/preconditioner", "none");

   IterativeSolver *solver = SetSolver(solver_type, prec_str);
   HypreParMatrix *parRomMat = NULL;
   Solver *M = NULL;    // preconditioner.
   Operator *K = NULL;  // operator.
   HypreBoomerAMG *amgM = NULL;
   // GSSmoother *gsM = NULL;

   if (prec_str == "amg")
   {
      // TODO: need to change when the actual parallelization is implemented.
      HYPRE_BigInt glob_size = rom_block_offsets.Last();
      HYPRE_BigInt row_starts[2] = {0, rom_block_offsets.Last()};
      parRomMat = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, romMat_mono);
      K = parRomMat;
   }
   else if (prec_str == "gs")
      K = romMat_mono;
   else
      K = romMat;

   if (prec_str == "amg")
   {
      amgM = new HypreBoomerAMG(*parRomMat);
      amgM->SetPrintLevel(print_level);
      M = amgM;
   }
   else if (prec_str == "gs")
   {
      M = new GSSmoother(*romMat_mono);
   }
   else if (prec_str == "block_gs")
   {
      M = new BlockGSSmoother(*romMat);
   }
   else if (prec_str == "block_jacobi")
   {
      M = new BlockDSmoother(*romMat);
   }
   else if (prec_str != "none")
   {
      mfem_error("Unknown preconditioner for ROM!\n");
   }

   if (prec_str != "none")
      solver->SetPreconditioner(*M);
   solver->SetOperator(*K);
   
   solver->SetAbsTol(atol);
   solver->SetRelTol(rtol);
   solver->SetMaxIter(maxIter);
   solver->SetPrintLevel(print_level);

   reduced_sol = 0.0;
   // StopWatch solveTimer;
   // solveTimer.Start();
   solver->Mult(*reduced_rhs, reduced_sol);
   // solveTimer.Stop();
   // printf("ROM-solve-only time: %f seconds.\n", solveTimer.RealTime());

   for (int i = 0; i < numSub; i++)
   {
      assert(U->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);

      // 23. reconstruct FOM state
      basis_i->Mult(reduced_sol.GetBlock(i).GetData(), U->GetBlock(i).GetData());
   }

   // delete the created objects.
   if (prec_str == "amg")
      delete parRomMat;
   delete M;
   delete solver;
}

SparseMatrix* MFEMROMHandler::ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat)
{
   assert((i >= 0) && (i < num_basis_sets));
   assert((j >= 0) && (j < num_basis_sets));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   DenseMatrix *basis_i, *basis_j;
   int num_basis_i, num_basis_j;
   GetBasis(i, basis_i);
   num_basis_i = basis_i->NumCols();
   GetBasis(j, basis_j);
   num_basis_j = basis_j->NumCols();

   return mfem::RtAP(*basis_i, *mat, *basis_j);
}

void MFEMROMHandler::LoadOperatorFromFile(const std::string input_prefix)
{
   assert(save_operator == ROMBuildingLevel::GLOBAL);
   delete romMat, romMat_mono;
   
   std::string filename;
   if (input_prefix == "")
      filename = operator_prefix;
   else
      filename = input_prefix;
   filename += ".h5";

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);   

   romMat = hdf5_utils::ReadBlockMatrix(file_id, "ROM_matrix", rom_block_offsets);
   romMat_mono = romMat->CreateMonolithic();

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   operator_loaded = true;
}

void MFEMROMHandler::LoadOperator(BlockMatrix *input_mat)
{
   delete romMat, romMat_mono;
   romMat = input_mat;
   romMat_mono = romMat->CreateMonolithic();
   operator_loaded = true;
}

void MFEMROMHandler::SaveBasisVisualization(
   const Array<FiniteElementSpace *> &fes, const std::vector<std::string> &var_names)
{
   if (!save_basis_visual) return;
   assert(basis_loaded);

   const int num_var = var_names.size();
   assert(fes.Size() == num_var * numSub);

   std::string visual_prefix = config.GetRequiredOption<std::string>("model_reduction/visualization/prefix");
   if (train_mode == TrainMode::UNIVERSAL)
      visual_prefix += "_universal";

   for (int c = 0; c < num_basis_sets; c++)
   {
      std::string file_prefix = visual_prefix;
      file_prefix += "_" + std::to_string(c);

      int midx = -1;
      switch (train_mode) {
         case (TrainMode::INDIVIDUAL): midx = c; break;
         case (TrainMode::UNIVERSAL):
         {
            for (int m = 0; m < numSub; m++)
               if (topol_handler->GetMeshType(m) == c) { midx = m; break; }
            break;
         }
         default: mfem_error("Unknown train mode!\n"); break;
      }
      assert(midx >= 0);

      Mesh *mesh = fes[midx * num_var]->GetMesh();
      const int order = fes[midx * num_var]->FEColl()->GetOrder();
      ParaViewDataCollection coll = ParaViewDataCollection(file_prefix.c_str(), mesh);

      Array<int> var_offsets(num_basis[c] * num_var + 1);
      var_offsets[0] = 0;
      for (int k = 0, vidx = 1; k < num_basis[c]; k++)
         for (int v = 0, idx = midx * num_var; v < num_var; v++, idx++, vidx++)
            var_offsets[vidx] = fes[idx]->GetVSize();
      var_offsets.PartialSum();
      BlockVector basis_view(spatialbasis[c]->GetData(), var_offsets);

      Array<GridFunction*> basis_gf(num_basis[c] * num_var);
      basis_gf = NULL;
      for (int k = 0, idx = 0; k < num_basis[c]; k++)
      {
         for (int v = 0, fidx = midx * num_var; v < num_var; v++, idx++, fidx++)
         {
            std::string field_name = var_names[v] + "_basis_" + std::to_string(k);
            basis_gf[idx] = new GridFunction(fes[fidx], basis_view.GetBlock(idx), 0);
            coll.RegisterField(field_name.c_str(), basis_gf[idx]);   
         }
      }

      coll.SetLevelsOfDetail(order);
      coll.SetHighOrderOutput(true);
      coll.SetPrecision(8);
      coll.SetOwnData(false);
      coll.Save();

      for (int k = 0; k < basis_gf.Size(); k++) delete basis_gf[k];
   }
}

IterativeSolver* MFEMROMHandler::SetSolver(const std::string &solver_type, const std::string &prec_type)
{
   IterativeSolver *solver;
   if (solver_type == "cg")
   {
      if (prec_type == "amg") solver = new CGSolver(MPI_COMM_WORLD);
      else                    solver = new CGSolver();
   }
   else if (solver_type == "minres")
   {
      if (prec_type == "amg") solver = new MINRESSolver(MPI_COMM_WORLD);
      else                    solver = new MINRESSolver();
   }
   else
   {
      mfem_error("Unknown ROM solver type!\n");
   }

   return solver;
}

// void MFEMROMHandler::GetBlockSparsity(
//    const SparseMatrix *mat, const Array<int> &block_offsets, Array2D<bool> &mat_zero_blocks)
// {
//    assert(mat);
//    assert(block_offsets.Size() > 0);
//    assert(mat->NumRows() == block_offsets.Last());
//    assert(mat->NumCols() == block_offsets.Last());

//    const int num_blocks = block_offsets.Size() - 1;
//    mat_zero_blocks.SetSize(num_blocks, num_blocks);
//    mat_zero_blocks = true;

//    DenseMatrix tmp;
//    Array<int> rows, cols;
//    for (int i = 0; i < num_blocks; i++)
//    {
//       rows.SetSize(block_offsets[i+1] - block_offsets[i]);
//       for (int ii = block_offsets[i], idx = 0; ii < block_offsets[i+1]; ii++, idx++)
//          rows[idx] = ii;

//       for (int j = 0; j < num_blocks; j++)
//       {
//          cols.SetSize(block_offsets[j+1] - block_offsets[j]);
//          for (int jj = block_offsets[j], idx = 0; jj < block_offsets[j+1]; jj++, idx++)
//             cols[idx] = jj;

//          tmp.SetSize(rows.Size(), cols.Size());
//          tmp = 0.0;
//          mat->GetSubMatrix(rows, cols, tmp);
//          mat_zero_blocks(i, j) = CheckZeroBlock(tmp);
//       }  // for (int j = 0; j < num_blocks; j++)
//    }  // for (int i = 0; i < num_blocks; i++)
// }

// bool MFEMROMHandler::CheckZeroBlock(const DenseMatrix &mat)
// {
//    for (int i = 0; i < mat.NumRows(); i++)
//       for (int j = 0; j < mat.NumCols(); j++)
//          if (mat(i, j) != 0.0)
//             return false;

//    return true;
// }

}  // namespace mfem
