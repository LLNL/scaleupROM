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

ROMHandler::ROMHandler(const int &input_numSub, const int &input_udim, const Array<int> &input_num_vdofs)
   : numSub(input_numSub),
     udim(input_udim),
     fom_num_vdofs(input_num_vdofs),
     basis_loaded(false),
     operator_loaded(false)
{
   assert(fom_num_vdofs.Size() == (numSub));

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

   num_basis = config.GetRequiredOption<int>("model_reduction/number_of_basis");

   basis_prefix = config.GetOption<std::string>("model_reduction/basis_prefix", "basis");

   save_operator = config.GetOption<bool>("model_reduction/save_operator/enabled", false);
   if (save_operator)
      operator_prefix = config.GetRequiredOption<std::string>("model_reduction/save_operator/prefix");
   // TODO: assemble on the fly if not save_operator.
   assert(save_operator);

   std::string train_mode_str = config.GetOption<std::string>("model_reduction/subdomain_training", "individual");
   if (train_mode_str == "individual")
   {
      train_mode = TrainMode::INDIVIDUAL;
      num_basis_sets = numSub;
   }
   else if (train_mode_str == "universal")
   {
      train_mode = TrainMode::UNIVERSAL;
      // TODO: multi-component basis.
      num_basis_sets = 1;
   }
   else
   {
      mfem_error("Unknown subdomain training mode!\n");
   }

   component_sampling = config.GetOption<bool>("sample_generation/component_sampling", false);
   if ((train_mode == TrainMode::INDIVIDUAL) && component_sampling)
      mfem_error("Component sampling is only supported with universal basis!\n");

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

   save_sv = config.GetOption<bool>("model_reduction/svd/save_spectrum", false);

   save_basis_visual = config.GetOption<bool>("model_reduction/visualization/enabled", false);

   AllocROMMat();
}

void ROMHandler::SaveSnapshot(Array<GridFunction*> &us, const int &sample_index)
{
   assert(us.Size() == numSub);
   
   for (int m = 0; m < numSub; m++)
   {
      const std::string filename = GetSnapshotPrefix(sample_index, m);
      rom_options = new CAROM::Options(fom_num_vdofs[m], max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, filename);

      bool addSample = basis_generator->takeSample(us[m]->GetData(), 0.0, 0.01);
      basis_generator->writeSnapshot();

      delete basis_generator;
      delete rom_options;
   }
}

void ROMHandler::FormReducedBasis(const int &total_samples)
{
   switch (train_mode)
   {
      case (TrainMode::UNIVERSAL):
      {
         FormReducedBasisUniversal(total_samples);
         break;
      }
      case (TrainMode::INDIVIDUAL):
      {
         FormReducedBasisIndividual(total_samples);
         break;
      }
      default:
      {
         mfem_error("ROMHandler: unknown train mode!\n");
      }
   }
}

void ROMHandler::FormReducedBasisUniversal(const int &total_samples)
{
   assert(train_mode == TrainMode::UNIVERSAL);
   std::string basis_name;

   basis_name = basis_prefix + "_universal";
   rom_options = new CAROM::Options(fom_num_vdofs[0], max_num_snapshots, 1, update_right_SV);
   basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);   

   int num_snapshot_sets = (component_sampling) ? num_basis_sets : numSub;
   for (int m = 0; m < num_snapshot_sets; m++)
   {
      for (int s = 0; s < total_samples; s++)
      {
         // TODO: we still need multi-component case adjustment for prefix.
         const std::string filename = GetSnapshotPrefix(s, m) + "_snapshot";
         basis_generator->loadSamples(filename,"snapshot");
      }
   }

   basis_generator->endSamples(); // save the merged basis file
   SaveSV(basis_name);

   delete basis_generator;
   delete rom_options;
}

void ROMHandler::FormReducedBasisIndividual(const int &total_samples)
{
   assert(train_mode == TrainMode::INDIVIDUAL);
   std::string basis_name;

   for (int m = 0; m < numSub; m++)
   {
      basis_name = basis_prefix + "_dom" + std::to_string(m);
      rom_options = new CAROM::Options(fom_num_vdofs[m], max_num_snapshots, 1, update_right_SV);
      basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);

      for (int s = 0; s < total_samples; s++)
      {
         const std::string filename = GetSnapshotPrefix(s, m) + "_snapshot";
         basis_generator->loadSamples(filename,"snapshot");
      }

      basis_generator->endSamples(); // save the merged basis file
      SaveSV(basis_name);

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
         carom_spatialbasis.SetSize(num_basis_sets);
         basis_name = basis_prefix + "_universal";
         basis_reader = new CAROM::BasisReader(basis_name);

         carom_spatialbasis[0] = basis_reader->getSpatialBasis(0.0, num_basis);
         numRowRB = carom_spatialbasis[0]->numRows();
         numColumnRB = carom_spatialbasis[0]->numColumns();
         printf("spatial basis dimension is %d x %d\n", numRowRB, numColumnRB);

         delete basis_reader;
         break;
      }
      case TrainMode::INDIVIDUAL:
      {
         carom_spatialbasis.SetSize(numSub);
         for (int j = 0; j < numSub; j++)
         {
            basis_name = basis_prefix + "_dom" + std::to_string(j);
            basis_reader = new CAROM::BasisReader(basis_name);

            carom_spatialbasis[j] = basis_reader->getSpatialBasis(0.0, num_basis);
            numRowRB = carom_spatialbasis[j]->numRows();
            numColumnRB = carom_spatialbasis[j]->numColumns();
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

void ROMHandler::GetReducedBasis(const int &subdomain_index, const CAROM::Matrix* &basis)
{
   MFEM_ASSERT(basis_loaded, "GetReducedBasis: reduced basis is not loaded!\n");

   switch (train_mode)
   {
      case TrainMode::UNIVERSAL:
      {
         // TODO: when using more than one component domain.
         basis = carom_spatialbasis[0];
         break;
      }
      case TrainMode::INDIVIDUAL:
      {
         basis = carom_spatialbasis[subdomain_index];
         break;
      }
      default:
      {
         mfem_error("LoadBasis: unknown TrainMode!\n");
         basis = NULL;
         break;
      }
   }  // switch (train_mode)

   return;
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
      GetReducedBasis(i, basis_i);

      for (int j = 0; j < numSub; j++)
      {
         GetReducedBasis(j, basis_j);

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

   operator_loaded = true;
   if (save_operator) romMat_inv->write(operator_prefix);
}

void ROMHandler::SetBlockSizes()
{
   // TODO: non-uniform subdomain cases.
   rom_block_offsets.SetSize(numSub+1);
   rom_block_offsets = 0;

   for (int k = 1; k <= numSub; k++)
   {
      rom_block_offsets[k] = num_basis;
   }
   rom_block_offsets.PartialSum();
}

void ROMHandler::AllocROMMat()
{
   SetBlockSizes();

   carom_mats.SetSize(numSub, numSub);
   // TODO: parallelization.
   romMat_inv = new CAROM::Matrix(numSub * num_basis, numSub * num_basis, false);
}

void ROMHandler::ProjectRHSOnReducedBasis(const BlockVector* RHS)
{
   assert(RHS->NumBlocks() == numSub);

   printf("Project RHS on reduced basis.\n");
   reduced_rhs = new CAROM::Vector(numSub * num_basis, false);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      assert(RHS->GetBlock(i).Size() == fom_num_vdofs[i]);

      const CAROM::Matrix* basis_i;
      GetReducedBasis(i, basis_i);

      CAROM::Vector block_rhs_carom(RHS->GetBlock(i).GetData(), RHS->GetBlock(i).Size(), true, false);
      CAROM::Vector *block_reduced_rhs = basis_i->transposeMult(&block_rhs_carom);

      CAROM::SetBlock(*block_reduced_rhs, i * num_basis, (i+1) * num_basis, *reduced_rhs);
   }
}

void ROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);

   printf("Solve ROM.\n");
   if (!operator_loaded)
   {  // TODO: assembling on the fly if not save_operator.
      assert(save_operator);
      romMat_inv->read(operator_prefix);
      operator_loaded = true;
   }

   CAROM::Vector reduced_sol(num_basis * numSub, false);
   romMat_inv->mult(*reduced_rhs, reduced_sol);

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      const CAROM::Matrix* basis_i;
      GetReducedBasis(i, basis_i);

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

void ROMHandler::SaveSV(const std::string& prefix)
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
      if (d == num_basis) coverage = total;
      total += rom_sv->item(d);
   }
   coverage /= total;
   printf("Coverage: %.7f%%\n", coverage * 100.0);

   // TODO: hdf5 format + parallel case.
   std::string filename = prefix + "_sv.txt";
   CAROM::PrintVector(*rom_sv, filename);
}

/*
   MFEMROMHandler
*/

MFEMROMHandler::MFEMROMHandler(const int &input_numSub, const int &input_udim, const Array<int> &input_num_vdofs)
   : ROMHandler(input_numSub, input_udim, input_num_vdofs)
{
   romMat = new SparseMatrix(numSub * num_basis, numSub * num_basis);
}

void MFEMROMHandler::LoadReducedBasis()
{
   ROMHandler::LoadReducedBasis();

   spatialbasis.SetSize(carom_spatialbasis.Size());
   for (int k = 0; k < spatialbasis.Size(); k++)
   {
      assert(carom_spatialbasis[k] != NULL);
      spatialbasis[k] = new DenseMatrix(carom_spatialbasis[k]->numRows(), carom_spatialbasis[k]->numColumns());
      CAROM::CopyMatrix(*carom_spatialbasis[k], *spatialbasis[k]);
   }

   basis_loaded = true;
}

void MFEMROMHandler::GetReducedBasis(const int &subdomain_index, DenseMatrix* &basis)
{
   MFEM_ASSERT(basis_loaded, "GetReducedBasis: reduced basis is not loaded!\n");

   switch (train_mode)
   {
      case TrainMode::UNIVERSAL:
      {
         // TODO: when using more than one component domain.
         basis = spatialbasis[0];
         break;
      }
      case TrainMode::INDIVIDUAL:
      {
         basis = spatialbasis[subdomain_index];
         break;
      }
      default:
      {
         mfem_error("LoadBasis: unknown TrainMode!\n");
         basis = NULL;
         break;
      }
   }  // switch (train_mode)

   return;
}

void MFEMROMHandler::ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats)
{
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);

   if (!basis_loaded) LoadReducedBasis();

   // This is pretty much the same as Assemble().
   // Each basis is applied to the same column blocks.
   DenseMatrix *basis_i, *basis_j;
   for (int i = 0; i < numSub; i++)
   {
      GetReducedBasis(i, basis_i);

      Array<int> vdof_i(num_basis);
      for (int k = 0; k < num_basis; k++) vdof_i[k] = k + i * num_basis;

      for (int j = 0; j < numSub; j++)
      {
         GetReducedBasis(j, basis_j);

         Array<int> vdof_j(num_basis);
         for (int k = 0; k < num_basis; k++) vdof_j[k] = k + j * num_basis;
         
         DenseMatrix elemmat(num_basis, num_basis);
         mfem::RtAP(*basis_i, *mats(i,j), *basis_j, elemmat);
         romMat->SetSubMatrix(vdof_i, vdof_j, elemmat);
      }
   }  // for (int j = 0; j < numSub; j++)

   romMat->Finalize();
   operator_loaded = true;

   if (save_operator)
   {
      std::string filename = operator_prefix + ".h5";
      WriteSparseMatrixToHDF(romMat, filename);
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
      GetReducedBasis(i, basis_i);
      basis_i->MultTranspose(RHS->GetBlock(i).GetData(), reduced_rhs->GetBlock(i).GetData());
   }
}

void MFEMROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);

   if (!operator_loaded)
   {  // TODO: option of assembling on the fly if not save_operator.
      assert(save_operator);
      std::string filename = operator_prefix + ".h5";
      romMat = ReadSparseMatrixFromHDF(filename);
      operator_loaded = true;
   }

   printf("Solve ROM.\n");
   BlockVector reduced_sol(rom_block_offsets);

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);
   bool use_amg = config.GetOption<bool>("solver/use_amg", true);

   CGSolver *solver = NULL;
   HypreParMatrix *parRomMat = NULL;
   HypreBoomerAMG *M = NULL;
   BlockDiagonalPreconditioner *globalPrec = NULL;

   if (use_amg)
   {
      solver = new CGSolver(MPI_COMM_WORLD);

      // TODO: need to change when the actual parallelization is implemented.
      HYPRE_BigInt glob_size = rom_block_offsets.Last();
      HYPRE_BigInt row_starts[2] = {0, rom_block_offsets.Last()};
      
      parRomMat = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, romMat);

      solver->SetOperator(*parRomMat);
      M = new HypreBoomerAMG(*parRomMat);
      solver->SetPreconditioner(*M);
   }
   else
   {
      solver = new CGSolver();
      solver->SetOperator(*romMat);
   }
   
   solver->SetAbsTol(atol);
   solver->SetRelTol(rtol);
   solver->SetMaxIter(maxIter);
   solver->SetPrintLevel(print_level);

   reduced_sol = 0.0;
   solver->Mult(*reduced_rhs, reduced_sol);

   for (int i = 0; i < numSub; i++)
   {
      assert(U->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetReducedBasis(i, basis_i);

      // 23. reconstruct FOM state
      basis_i->Mult(reduced_sol.GetBlock(i).GetData(), U->GetBlock(i).GetData());
   }
}

void MFEMROMHandler::SaveBasisVisualization(const Array<FiniteElementSpace *> &fes)
{
   if (!save_basis_visual) return;
   assert(basis_loaded);

   std::string visual_prefix = config.GetRequiredOption<std::string>("model_reduction/visualization/prefix");
   if (train_mode == TrainMode::UNIVERSAL)
      visual_prefix += "_universal";

   for (int m = 0; m < num_basis_sets; m++)
   {
      std::string file_prefix = visual_prefix;
      file_prefix += "_" + std::to_string(m);

      // TODO: Multi-component, universal basis case (index not necessarily matches the subdomain index.)
      Mesh *mesh = fes[m]->GetMesh();
      const int order = fes[m]->FEColl()->GetOrder();
      ParaViewDataCollection *coll = new ParaViewDataCollection(file_prefix.c_str(), mesh);
      coll->SetLevelsOfDetail(order);
      coll->SetHighOrderOutput(true);
      coll->SetPrecision(8);

      Array<GridFunction*> basis_gf(num_basis);
      basis_gf = NULL;
      for (int k = 0; k < num_basis; k++)
      {
         std::string field_name = "basis_" + std::to_string(k);
         basis_gf[k] = new GridFunction(fes[m], spatialbasis[m]->GetColumn(k));
         coll->RegisterField(field_name.c_str(), basis_gf[k]);
         coll->SetOwnData(false);
      }

      coll->Save();
   }
}

}
