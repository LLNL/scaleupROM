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

#include "input_parser.hpp"
#include "rom_handler.hpp"
#include "linalg_utils.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;

namespace mfem
{

ROMHandler::ROMHandler(const int &input_numSub, const Array<int> &input_num_dofs)
   : numSub(input_numSub), fom_num_dofs(input_num_dofs), basis_loaded(false), proj_inv_loaded(false)
{
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
      sample_prefix = config.GetRequiredOption<std::string>("sample_generation/prefix");

   num_basis = config.GetRequiredOption<int>("model_reduction/number_of_basis");

   basis_prefix = config.GetOption<std::string>("model_reduction/basis_prefix", "basis");

   save_proj_inv = config.GetOption<bool>("model_reduction/save_projected_inverse", true);
   proj_inv_prefix = config.GetOption<std::string>("model_reduction/projected_inverse_filename", "proj_inv");

   std::string train_mode_str = config.GetOption<std::string>("model_reduction/subdomain_training", "individual");
   if (train_mode_str == "individual")
   {
      train_mode = TrainMode::INDIVIDUAL;
   }
   else if (train_mode_str == "universal")
   {
      train_mode = TrainMode::UNIVERSAL;
   }
   else
   {
      mfem_error("Unknown subdomain training mode!\n");
   }

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

   max_num_snapshots = config.GetOption<int>("model_reduction/svd/maximum_number_of_snapshots", 100);
   update_right_SV = config.GetOption<bool>("model_reduction/svd/update_right_sv", false);

   AllocROMMat();
}

void ROMHandler::SaveSnapshot(Array<GridFunction*> &us, const int &sample_index)
{
   assert(us.Size() == numSub);
   
   for (int m = 0; m < numSub; m++)
   {
      std::string filename(sample_prefix + "_sample" + std::to_string(sample_index) + "_dom" + std::to_string(m));
      rom_options = new CAROM::Options(fom_num_dofs[m], max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, filename);

      bool addSample = basis_generator->takeSample(us[m]->GetData(), 0.0, 0.01);
      basis_generator->writeSnapshot();

      delete basis_generator;
      delete rom_options;
   }
}

void ROMHandler::FormReducedBasis(const int &total_samples)
{
   std::string basis_name;

   if (train_mode == TrainMode::UNIVERSAL)
   {
      basis_name = basis_prefix + "_universal";
      rom_options = new CAROM::Options(fom_num_dofs[0], max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);   
   }

   for (int m = 0; m < numSub; m++)
   {
      if (train_mode == TrainMode::INDIVIDUAL)
      {
         basis_name = basis_prefix + "_dom" + std::to_string(m);
         rom_options = new CAROM::Options(fom_num_dofs[m], max_num_snapshots, 1, update_right_SV);
         basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);
      }

      for (int s = 0; s < total_samples; s++)
      {
         std::string filename(sample_prefix + "_sample" + std::to_string(s) + "_dom" + std::to_string(m) + "_snapshot");
         basis_generator->loadSamples(filename,"snapshot");
      }

      if (train_mode == TrainMode::INDIVIDUAL)
      {
         basis_generator->endSamples(); // save the merged basis file

         const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
         printf("Singular values: ");
         for (int d = 0; d < rom_sv->dim(); d++)
            printf("%.3f\t", rom_sv->item(d));
         printf("\n");

         delete basis_generator;
         delete rom_options;
      }
   }

   if (train_mode == TrainMode::UNIVERSAL)
   {
      basis_generator->endSamples(); // save the merged basis file

      const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
      printf("Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3E\t", rom_sv->item(d));
      printf("\n");

      delete basis_generator;
      delete rom_options;
   }
}

void ROMHandler::LoadReducedBasis()
{
   if (basis_loaded) return;

   std::string basis_name;
   int numRowRB, numColumnRB;

   switch (train_mode)
   {
      case TrainMode::UNIVERSAL:
      {  // TODO: when using more than one component domain.
         spatialbasis.SetSize(1);
         basis_name = basis_prefix + "_universal";
         basis_reader = new CAROM::BasisReader(basis_name);

         spatialbasis[0] = basis_reader->getSpatialBasis(0.0, num_basis);
         numRowRB = spatialbasis[0]->numRows();
         numColumnRB = spatialbasis[0]->numColumns();
         printf("spatial basis dimension is %d x %d\n", numRowRB, numColumnRB);

         delete basis_reader;
         break;
      }
      case TrainMode::INDIVIDUAL:
      {
         spatialbasis.SetSize(numSub);
         for (int j = 0; j < numSub; j++)
         {
            basis_name = basis_prefix + "_dom" + std::to_string(j);
            basis_reader = new CAROM::BasisReader(basis_name);

            spatialbasis[j] = basis_reader->getSpatialBasis(0.0, num_basis);
            numRowRB = spatialbasis[j]->numRows();
            numColumnRB = spatialbasis[j]->numColumns();
            printf("%d domain spatial basis dimension is %d x %d\n", j, numRowRB, numColumnRB);

            delete basis_reader;
         }
         break;
      }
      default:
      {
         mfem_error("LoadBasis: unknown TrainMode!\n");
         break;
      }
   }  // switch (train_mode)

   basis_loaded = true;
}

const CAROM::Matrix* ROMHandler::GetReducedBasis(const int &subdomain_index)
{
   MFEM_ASSERT(basis_loaded, "GetReducedBasis: reduced basis is not loaded!\n");

   switch (train_mode)
   {
      case TrainMode::UNIVERSAL:
      {
         // TODO: when using more than one component domain.
         return spatialbasis[0];
         break;
      }
      case TrainMode::INDIVIDUAL:
      {
         return spatialbasis[subdomain_index];
         break;
      }
      default:
      {
         mfem_error("LoadBasis: unknown TrainMode!\n");
         return NULL;
         break;
      }
   }  // switch (train_mode)
}

void ROMHandler::ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats)
{
   printf("Project Operators on reduced basis.\n");
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);

   if (!basis_loaded) LoadReducedBasis();

   // // Prepare matrixes.
   // AllocROMMat();

   // Each basis is applied to the same column blocks.
   const CAROM::Matrix *basis_i, *basis_j;
   for (int i = 0; i < numSub; i++)
   {
      basis_i = GetReducedBasis(i);

      for (int j = 0; j < numSub; j++)
      {
         basis_j = GetReducedBasis(j);

         // 21. form inverse ROM operator
         assert(mats(i,j) != NULL);
         assert(mats(i,j)->Finalized());

         carom_mats(i,j) = new CAROM::Matrix(num_basis, num_basis, false);
         CAROM::ComputeCtAB(*mats(i,j), *basis_j, *basis_i, *carom_mats(i,j));
      }
   }  // for (int j = 0; j < numSub; j++)

   // Form inverse matrix
   // TODO: which linear algbra utilities should I use? MFEM or CAROM?
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         CAROM::SetBlock(*carom_mats(i,j), rom_block_offsets[i], rom_block_offsets[i+1],
                         rom_block_offsets[j], rom_block_offsets[j+1], *romMat_inv);
      }
   }

   romMat_inv->inverse();

   proj_inv_loaded = true;
   if (save_proj_inv) romMat_inv->write(proj_inv_prefix);
}

void ROMHandler::AllocROMMat()
{
   // TODO: non-uniform subdomain cases.
   rom_block_offsets.SetSize(numSub+1);
   rom_block_offsets = 0;

   for (int k = 1; k <= numSub; k++)
   {
      rom_block_offsets[k] = num_basis;
   }
   rom_block_offsets.PartialSum();

   // TODO: If using MFEM linear algebra.
   // rom_mats.SetSize(numSub, numSub);
   // for (int i = 0; i < numSub; i++)
   // {
   //    for (int j = 0; j < numSub; j++)
   //    {
   //       rom_mats(i, j) = new SparseMatrix(num_basis, num_basis);
   //    }
   // }

   // TODO: If using MFEM linear algebra.
   // romMat = new BlockOperator(rom_block_offsets);
   // for (int i = 0; i < numSub; i++)
   //    for (int j = 0; j < numSub; j++)
   //       romMat->SetBlock(i, j, rom_mats(i, j));

   carom_mats.SetSize(numSub, numSub);
   // TODO: parallelization.
   romMat_inv = new CAROM::Matrix(numSub * num_basis, numSub * num_basis, false);
}

void ROMHandler::ProjectRHSOnReducedBasis(const BlockVector* RHS)
{
   printf("Project RHS on reduced basis.\n");
   reduced_rhs = new CAROM::Vector(numSub * num_basis, false);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      const CAROM::Matrix* basis_i = GetReducedBasis(i);

      CAROM::Vector block_rhs_carom(RHS->GetBlock(i).GetData(), RHS->GetBlock(i).Size(), true, false);
      CAROM::Vector *block_reduced_rhs = basis_i->transposeMult(&block_rhs_carom);

      CAROM::SetBlock(*block_reduced_rhs, i * num_basis, (i+1) * num_basis, *reduced_rhs);
   }
}

void ROMHandler::Solve(BlockVector* U)
{
   printf("Solve ROM.\n");
   if (!proj_inv_loaded)
   {
      romMat_inv->read(proj_inv_prefix);
      proj_inv_loaded = true;
   }

   CAROM::Vector reduced_sol(num_basis * numSub, false);
   romMat_inv->mult(*reduced_rhs, reduced_sol);

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      const CAROM::Matrix* basis_i = GetReducedBasis(i);

      // 23. reconstruct FOM state
      CAROM::Vector block_reduced_sol(num_basis, false);
      const int offset = i * num_basis;
      for (int k = 0; k < num_basis; k++)
         block_reduced_sol(k) = reduced_sol(k + offset);

      // This saves the data automatically to U.
      CAROM::Vector U_block_carom(U->GetBlock(i).GetData(), U->GetBlock(i).Size(), true, false);
      basis_i->mult(block_reduced_sol, U_block_carom);
   }
}

}
