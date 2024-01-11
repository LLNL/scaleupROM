// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "etc.hpp"
#include "input_parser.hpp"
#include "rom_handler.hpp"
#include "hdf5_utils.hpp"
#include "block_smoother.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;

namespace mfem
{

const TrainMode SetTrainMode()
{
   TrainMode train_mode = TrainMode::NUM_TRAINMODE;

   std::string train_mode_str = config.GetOption<std::string>("model_reduction/subdomain_training", "individual");
   if (train_mode_str == "individual")       train_mode = TrainMode::INDIVIDUAL;
   else if (train_mode_str == "universal")   train_mode = TrainMode::UNIVERSAL;
   else
      mfem_error("Unknown subdomain training mode!\n");

   return train_mode;
}

const std::string GetBasisTagForComponent(
   const int &comp_idx, const TrainMode &train_mode, const TopologyHandler *topol_handler)
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

const std::string GetBasisTag(
   const int &subdomain_index, const TrainMode &train_mode, const TopologyHandler *topol_handler)
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

ROMHandler::ROMHandler(const TrainMode &train_mode_, TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_var_offsets, const bool separate_variable_basis)
   : train_mode(train_mode_),
     topol_handler(input_topol),
     numSub(input_topol->GetNumSubdomains()),
     fom_vdim(input_vdim),
     fom_var_offsets(input_var_offsets),
     basis_loaded(false),
     operator_loaded(false),
     separate_variable(separate_variable_basis)
{
   num_var = fom_vdim.Size();
   udim = fom_vdim.Sum();

   fom_num_vdofs.SetSize(numSub);
   for (int m = 0, idx = 0; m < numSub; m++)
   {
      fom_num_vdofs[m] = 0;
      for (int v = 0; v < num_var; v++, idx++)
         fom_num_vdofs[m] += fom_var_offsets[idx+1] - fom_var_offsets[idx];
   }

   ParseInputs();

   SetBlockSizes();

   AllocROMMat();
}

ROMHandler::~ROMHandler()
{
   DeletePointers(carom_mats);
   delete reduced_rhs;
   delete reduced_sol;
   delete romMat_inv;
}

void ROMHandler::ParseInputs()
{
   assert(numSub > 0);
   assert(topol_handler != NULL);

   basis_prefix = config.GetOption<std::string>("basis/prefix", "basis");

   std::string save_op_str = config.GetOption<std::string>("model_reduction/save_operator/level", "none");
   if (save_op_str == "none")             save_operator = ROMBuildingLevel::NONE;
   else if (save_op_str == "global")      save_operator = ROMBuildingLevel::GLOBAL;
   else if (save_op_str == "component")   save_operator = ROMBuildingLevel::COMPONENT;
   else
      mfem_error("Unknown ROM building level!\n");

   if (save_operator != ROMBuildingLevel::NONE)
      operator_prefix = config.GetRequiredOption<std::string>("model_reduction/save_operator/prefix");

   num_rom_blocks = numSub;
   if (train_mode == TrainMode::INDIVIDUAL)        num_rom_comp_blocks = numSub;
   else if (train_mode == TrainMode::UNIVERSAL)    num_rom_comp_blocks = topol_handler->GetNumComponents();
   else
      mfem_error("ROMHandler - subdomain training mode is not set!\n");

   if (separate_variable)
   {
      num_rom_comp_blocks *= num_var;
      num_rom_blocks *= num_var;
   }
   
   const int comp_num_basis_default = config.GetOption<int>("basis/number_of_basis", -1);
   comp_num_basis.SetSize(num_rom_comp_blocks);
   comp_num_basis = comp_num_basis_default;
   assert(comp_num_basis.Size() > 0);

   YAML::Node basis_list = config.FindNode("basis/tags");
   if ((!basis_list) && (comp_num_basis_default <= 0))
      mfem_error("ROMHandler - cannot find the basis tag list, nor default number of basis is not set!\n");

   for (int c = 0; c < num_rom_comp_blocks; c++)
   {
      std::string basis_tag = GetBasisTagForComponent(c, train_mode, topol_handler);

      // Not so sure we need to explicitly list out all the needed basis tags here.
      for (int p = 0; p < basis_list.size(); p++)
         if (basis_tag == config.GetRequiredOptionFromDict<std::string>("name", basis_list[p]))
         {
            const int nb = config.GetOptionFromDict<int>("number_of_basis", comp_num_basis_default, basis_list[p]);
            comp_num_basis[c] = nb;
            break;
         }
      
      if (comp_num_basis[c] < 0)
      {
         printf("Cannot find the number of basis for %s!\n", basis_tag.c_str());
         mfem_error("Or specify the default number of basis!\n");
      }
   }
   
   for (int k = 0; k < comp_num_basis.Size(); k++)
      assert(comp_num_basis[k] > 0);

   max_num_snapshots = config.GetOption<int>("sample_generation/maximum_number_of_snapshots", 100);
   update_right_SV = config.GetOption<bool>("basis/svd/update_right_sv", false);

   save_sv = config.GetOption<bool>("basis/svd/save_spectrum", false);

   save_basis_visual = config.GetOption<bool>("basis/visualization/enabled", false);
}

// void ROMHandler::FormReducedBasis()
// {
//    std::string basis_name, basis_tag;
   
//    for (int c = 0; c < num_rom_comp_blocks; c++)
//    {
//       int basis_dim = -1;
//       switch (train_mode)
//       {
//          case (INDIVIDUAL):
//          {
//             basis_tag = GetBasisTag(c);
//             basis_dim = fom_num_vdofs[c];
//             break;
//          }
//          case (UNIVERSAL):
//          {
//             basis_tag = GetBasisTagForComponent(c);
//             for (int m = 0; m < numSub; m++)
//                if (topol_handler->GetMeshType(m) == c)
//                { basis_dim = fom_num_vdofs[m]; break; }
//             break;
//          }
//       }
//       basis_name = basis_prefix + "_" + basis_tag;
//       assert(basis_dim > 0);

//       rom_options = new CAROM::Options(basis_dim, max_num_snapshots, 1, update_right_SV);
//       basis_generator = new CAROM::BasisGenerator(*rom_options, incremental, basis_name);

//       std::string filename = sample_dir + "/" + sample_prefix + "_sample";
//       filename += "_" + basis_tag + "_snapshot";
//       basis_generator->loadSamples(filename,"snapshot");

//       basis_generator->endSamples(); // save the merged basis file
//       SaveSV(basis_name, c);

//       delete basis_generator;
//       delete rom_options;
//    }  // for (int c = 0; c < num_rom_comp_blocks; c++)
// }

void ROMHandler::LoadReducedBasis()
{
   if (basis_loaded) return;

   std::string basis_name;
   int numRowRB, numColumnRB;

   carom_comp_basis.SetSize(num_rom_comp_blocks);
   for (int k = 0; k < num_rom_comp_blocks; k++)
   {
      basis_name = basis_prefix + "_" + GetBasisTagForComponent(k, train_mode, topol_handler);
      basis_reader = new CAROM::BasisReader(basis_name);

      carom_comp_basis[k] = basis_reader->getSpatialBasis(0.0, comp_num_basis[k]);
      numRowRB = carom_comp_basis[k]->numRows();
      numColumnRB = carom_comp_basis[k]->numColumns();
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
   assert(num_rom_comp_blocks > 0);
   assert((basis_index >= 0) && (basis_index < num_rom_comp_blocks));

   basis = carom_comp_basis[basis_index];
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
      rom_block_offsets[k] = comp_num_basis[c];
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

void ROMHandler::ProjectVectorOnReducedBasis(const BlockVector* vec, CAROM::Vector*& rom_vec)
{
   assert(vec->NumBlocks() == numSub);
   // reset the rom_vec if initiated a priori.
   if (rom_vec) delete rom_vec;

   rom_vec = new CAROM::Vector(rom_block_offsets.Last(), false);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      assert(vec->GetBlock(i).Size() == fom_num_vdofs[i]);

      const CAROM::Matrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);

      CAROM::Vector block_vec_carom(vec->GetBlock(i).GetData(), vec->GetBlock(i).Size(), true, false);
      CAROM::Vector *block_reduced_vec = basis_i->transposeMult(&block_vec_carom);

      CAROM::SetBlock(*block_reduced_vec, rom_block_offsets[i], rom_block_offsets[i+1], *rom_vec);
   }
}

void ROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);
   assert(operator_loaded);

   printf("Solve ROM.\n");

   reduced_sol = new CAROM::Vector(rom_block_offsets.Last(), false);
   romMat_inv->mult(*reduced_rhs, *reduced_sol);

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      const CAROM::Matrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);
      int c = topol_handler->GetMeshType(i);

      // 23. reconstruct FOM state
      CAROM::Vector block_reduced_sol(comp_num_basis[c], false);
      const int offset = rom_block_offsets[i];
      for (int k = 0; k < comp_num_basis[c]; k++)
         block_reduced_sol(k) = reduced_sol->item(k + offset);

      // This saves the data automatically to U.
      CAROM::Vector U_block_carom(U->GetBlock(i).GetData(), U->GetBlock(i).Size(), true, false);
      basis_i->mult(block_reduced_sol, U_block_carom);
   }
}

void ROMHandler::ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat, CAROM::Matrix *proj_mat)
{
   assert(proj_mat != NULL);
   assert((i >= 0) && (i < num_rom_comp_blocks));
   assert((j >= 0) && (j < num_rom_comp_blocks));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   const CAROM::Matrix *basis_i, *basis_j;
   int comp_num_basis_i, comp_num_basis_j;
   GetBasis(i, basis_i);
   comp_num_basis_i = basis_i->numColumns();
   GetBasis(j, basis_j);
   comp_num_basis_j = basis_j->numColumns();

   // TODO: multi-component case.
   proj_mat->setSize(comp_num_basis_i, comp_num_basis_j);
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

// void ROMHandler::SaveSV(const std::string& prefix, const int& basis_idx)
// {
//    if (!save_sv) return;
//    assert(basis_generator != NULL);

//    const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
//    printf("Singular values: ");
//    for (int d = 0; d < rom_sv->dim(); d++)
//       printf("%.3E\t", rom_sv->item(d));
//    printf("\n");

//    double coverage = 0.0;
//    double total = 0.0;

//    for (int d = 0; d < rom_sv->dim(); d++)
//    {
//       if (d == comp_num_basis[basis_idx]) coverage = total;
//       total += rom_sv->item(d);
//    }
//    if (rom_sv->dim() == comp_num_basis[basis_idx]) coverage = total;
//    coverage /= total;
//    printf("Coverage: %.7f%%\n", coverage * 100.0);

//    // TODO: hdf5 format + parallel case.
//    std::string filename = prefix + "_sv.txt";
//    CAROM::PrintVector(*rom_sv, filename);
// }

/*
   MFEMROMHandler
*/

MFEMROMHandler::MFEMROMHandler(const TrainMode &train_mode_, TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_var_offsets, bool separate_variable_basis)
   : ROMHandler(train_mode_, input_topol, input_vdim, input_var_offsets, separate_variable_basis)
{
   romMat = new BlockMatrix(rom_block_offsets);
   romMat->owns_blocks = true;

   carom_mats.SetSize(0, 0);

   std::string solver_type_str = config.GetOption<std::string>("model_reduction/linear_solver_type", "cg");
   if (solver_type_str == "direct")       linsol_type = MFEMROMHandler::SolverType::DIRECT;
   else if (solver_type_str == "cg")      linsol_type = MFEMROMHandler::SolverType::CG;
   else if (solver_type_str == "minres")      linsol_type = MFEMROMHandler::SolverType::MINRES;
   else
   {
      mfem_error("Unknown ROM linear solver type!\n");
   }
   
   if (linsol_type == MFEMROMHandler::SolverType::DIRECT)
   {
      std::string mat_type_str = config.GetOption<std::string>("model_reduction/linear_system_type", "spd");
      if (mat_type_str == "spd")          mat_type = MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE;
      else if (mat_type_str == "sid")     mat_type = MUMPSSolver::MatType::SYMMETRIC_INDEFINITE;
      else if (mat_type_str == "us")      mat_type = MUMPSSolver::MatType::UNSYMMETRIC;
      else
      {
         mfem_error("Unknown ROM linear system type!\n");
      }
   }
}

MFEMROMHandler::~MFEMROMHandler()
{
   DeletePointers(comp_basis);
   delete romMat;
   delete romMat_mono;
   delete reduced_rhs;
   delete reduced_sol;
   delete romMat_hypre;
   delete mumps;
}

void MFEMROMHandler::LoadReducedBasis()
{
   ROMHandler::LoadReducedBasis();

   comp_basis.SetSize(carom_comp_basis.Size());
   comp_basis = NULL;
   for (int k = 0; k < comp_basis.Size(); k++)
   {
      assert(carom_comp_basis[k] != NULL);
      comp_basis[k] = new DenseMatrix(carom_comp_basis[k]->numRows(), carom_comp_basis[k]->numColumns());
      CAROM::CopyMatrix(*carom_comp_basis[k], *comp_basis[k]);
   }

   basis_loaded = true;
}

void MFEMROMHandler::GetBasis(const int &basis_index, DenseMatrix* &basis)
{
   assert(num_rom_comp_blocks > 0);
   assert((basis_index >= 0) && (basis_index < num_rom_comp_blocks));

   basis = comp_basis[basis_index];
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
   int comp_num_basis_i, comp_num_basis_j;
   int basis_i, basis_j;
   for (int i = 0; i < numSub; i++)
   {
      comp_num_basis_i = rom_block_offsets[i+1] - rom_block_offsets[i];
      basis_i = GetBasisIndexForSubdomain(i);

      for (int j = 0; j < numSub; j++)
      {
         comp_num_basis_j = rom_block_offsets[j+1] - rom_block_offsets[j];
         basis_j = GetBasisIndexForSubdomain(j);
         
         SparseMatrix *elemmat = ProjectOperatorOnReducedBasis(basis_i, basis_j, mats(i,j));
         romMat->SetBlock(i, j, elemmat);
      }
   }  // for (int j = 0; j < numSub; j++)

   romMat->Finalize();
   operator_loaded = true;

   romMat_mono = romMat->CreateMonolithic();

   if (linsol_type == SolverType::DIRECT) SetupDirectSolver();

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

void MFEMROMHandler::ProjectVectorOnReducedBasis(const BlockVector* vec, BlockVector*& rom_vec)
{
   assert(vec->NumBlocks() == numSub);
   // reset rom_vec if initiated a priori.
   if (rom_vec) delete rom_vec;

   rom_vec = new BlockVector(rom_block_offsets);

   if (!basis_loaded) LoadReducedBasis();

   // Each basis is applied to the same column blocks.
   for (int i = 0; i < numSub; i++)
   {
      assert(vec->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);
      basis_i->MultTranspose(vec->GetBlock(i).GetData(), rom_vec->GetBlock(i).GetData());
   }
}

void MFEMROMHandler::Solve(BlockVector* U)
{
   assert(U->NumBlocks() == numSub);
   assert(operator_loaded);

   printf("Solve ROM.\n");
   reduced_sol = new BlockVector(rom_block_offsets);
   (*reduced_sol) = 0.0;

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);   
   std::string prec_str = config.GetOption<std::string>("model_reduction/preconditioner", "none");

   if (linsol_type == SolverType::DIRECT)
   {
      assert(mumps);
      mumps->SetPrintLevel(print_level);
      mumps->Mult(*reduced_rhs, *reduced_sol);
   }
   else
   {
      IterativeSolver *solver = SetIterativeSolver(linsol_type, prec_str);
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
      else if ((prec_str == "gs") || (prec_str == "none"))
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

      // StopWatch solveTimer;
      // solveTimer.Start();
      solver->Mult(*reduced_rhs, *reduced_sol);
      // solveTimer.Stop();
      // printf("ROM-solve-only time: %f seconds.\n", solveTimer.RealTime());

      // delete the created objects.
      if (prec_str == "amg")
         delete parRomMat;
      delete M;
      delete solver;
   }

   for (int i = 0; i < numSub; i++)
   {
      assert(U->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);

      // 23. reconstruct FOM state
      basis_i->Mult(reduced_sol->GetBlock(i).GetData(), U->GetBlock(i).GetData());
   }
}

void MFEMROMHandler::NonlinearSolve(Operator &oper, BlockVector* U, Solver *prec)
{
   assert(U->NumBlocks() == numSub);

   printf("Solve ROM.\n");
   reduced_sol = new BlockVector(rom_block_offsets);
   bool use_restart = config.GetOption<bool>("solver/use_restart", false);
   if (use_restart)
   {
      for (int i = 0; i < numSub; i++)
      {
         assert(U->GetBlock(i).Size() == fom_num_vdofs[i]);

         DenseMatrix* basis_i;
         GetBasisOnSubdomain(i, basis_i);

         // Project from a FOM solution
         basis_i->MultTranspose(U->GetBlock(i).GetData(), reduced_sol->GetBlock(i).GetData());
      }
   }
   else
      (*reduced_sol) = 0.0;

   int maxIter = config.GetOption<int>("solver/max_iter", 100);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-10);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-10);
   int print_level = config.GetOption<int>("solver/print_level", 0);   

   int jac_maxIter = config.GetOption<int>("solver/jacobian/max_iter", 10000);
   double jac_rtol = config.GetOption<double>("solver/jacobian/relative_tolerance", 1.e-10);
   double jac_atol = config.GetOption<double>("solver/jacobian/absolute_tolerance", 1.e-10);
   int jac_print_level = config.GetOption<int>("solver/jacobian/print_level", -1);
   std::string prec_str = config.GetOption<std::string>("model_reduction/preconditioner", "none");
   if (prec_str != "none") assert(prec);

   Solver *J_solver = NULL;
   if (linsol_type == SolverType::DIRECT)
   {
      mumps = new MUMPSSolver();
      mumps->SetMatrixSymType(mat_type);
      mumps->SetPrintLevel(jac_print_level);
      J_solver = mumps;
   }
   else
   {
      IterativeSolver *iter_solver = SetIterativeSolver(linsol_type, prec_str);
      iter_solver->SetAbsTol(jac_atol);
      iter_solver->SetRelTol(jac_rtol);
      iter_solver->SetMaxIter(jac_maxIter);
      iter_solver->SetPrintLevel(jac_print_level);
      if (prec) iter_solver->SetPreconditioner(*prec);
      J_solver = iter_solver;
   }

   std::string nlin_solver = config.GetOption<std::string>("model_reduction/nonlinear_solver_type", "newton");

   if (nlin_solver == "newton")
   {
      NewtonSolver newton_solver;
      newton_solver.SetSolver(*J_solver);
      newton_solver.SetOperator(oper);
      newton_solver.SetPrintLevel(print_level); // print Newton iterations
      newton_solver.SetRelTol(rtol);
      newton_solver.SetAbsTol(atol);
      newton_solver.SetMaxIter(maxIter);

      newton_solver.Mult(*reduced_rhs, *reduced_sol);
   }
   else if (nlin_solver == "cg")
   {
      CGOptimizer optim;
      optim.SetOperator(oper);
      optim.SetPrintLevel(print_level); // print Newton iterations
      optim.SetRelTol(rtol);
      optim.SetAbsTol(atol);
      optim.SetMaxIter(maxIter);

      optim.Mult(*reduced_rhs, *reduced_sol);
   }
   else
      mfem_error("MFEMROMHandler::NonlinearSolve- Unknown ROM nonlinear solver type!\n");

   for (int i = 0; i < numSub; i++)
   {
      assert(U->GetBlock(i).Size() == fom_num_vdofs[i]);

      DenseMatrix* basis_i;
      GetBasisOnSubdomain(i, basis_i);

      // 23. reconstruct FOM state
      basis_i->Mult(reduced_sol->GetBlock(i).GetData(), U->GetBlock(i).GetData());
   }
}

SparseMatrix* MFEMROMHandler::ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat)
{
   assert((i >= 0) && (i < num_rom_comp_blocks));
   assert((j >= 0) && (j < num_rom_comp_blocks));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   DenseMatrix *basis_i, *basis_j;
   int comp_num_basis_i, comp_num_basis_j;
   GetBasis(i, basis_i);
   comp_num_basis_i = basis_i->NumCols();
   GetBasis(j, basis_j);
   comp_num_basis_j = basis_j->NumCols();

   return mfem::RtAP(*basis_i, *mat, *basis_j);
}

void MFEMROMHandler::LoadOperatorFromFile(const std::string input_prefix)
{
   assert(save_operator == ROMBuildingLevel::GLOBAL);
   delete romMat;
   delete romMat_mono;
   
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

   if (linsol_type == SolverType::DIRECT) SetupDirectSolver();

   operator_loaded = true;
}

void MFEMROMHandler::LoadOperator(BlockMatrix *input_mat)
{
   delete romMat;
   delete romMat_mono;
   romMat = input_mat;
   romMat_mono = romMat->CreateMonolithic();
   if (linsol_type == SolverType::DIRECT) SetupDirectSolver();
   operator_loaded = true;
}

void MFEMROMHandler::SaveBasisVisualization(
   const Array<FiniteElementSpace *> &fes, const std::vector<std::string> &var_names)
{
   if (!save_basis_visual) return;
   assert(basis_loaded);

   const int num_var = var_names.size();
   assert(fes.Size() == num_var * numSub);

   std::string visual_prefix = config.GetRequiredOption<std::string>("basis/visualization/prefix");
   if (train_mode == TrainMode::UNIVERSAL)
      visual_prefix += "_universal";

   for (int c = 0; c < num_rom_comp_blocks; c++)
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

      Array<int> var_offsets(comp_num_basis[c] * num_var + 1);
      var_offsets[0] = 0;
      for (int k = 0, vidx = 1; k < comp_num_basis[c]; k++)
         for (int v = 0, idx = midx * num_var; v < num_var; v++, idx++, vidx++)
            var_offsets[vidx] = fes[idx]->GetVSize();
      var_offsets.PartialSum();
      BlockVector basis_view(comp_basis[c]->GetData(), var_offsets);

      Array<GridFunction*> basis_gf(comp_num_basis[c] * num_var);
      basis_gf = NULL;
      for (int k = 0, idx = 0; k < comp_num_basis[c]; k++)
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

IterativeSolver* MFEMROMHandler::SetIterativeSolver(const MFEMROMHandler::SolverType &linsol_type_, const std::string &prec_type)
{
   IterativeSolver *solver;
   switch (linsol_type_)
   {
      case (SolverType::CG):
      {
         if (prec_type == "amg") solver = new CGSolver(MPI_COMM_WORLD);
         else                    solver = new CGSolver();
         break;
      }
      case (SolverType::MINRES):
      {
         if (prec_type == "amg") solver = new MINRESSolver(MPI_COMM_WORLD);
         else                    solver = new MINRESSolver();
         break;
      }
      case (SolverType::GMRES):
      {
         if (prec_type == "amg") solver = new GMRESSolver(MPI_COMM_WORLD);
         else                    solver = new GMRESSolver();
         break;
      }
      default:
      {
         mfem_error("Unknown ROM iterative linear solver type!\n");
         break;
      }
   }

   return solver;
}

void MFEMROMHandler::SetupDirectSolver()
{
   // If nonlinear mode, Jacobian will keep changing within Solve, thus no need of initial LU factorization.
   if ((linsol_type != MFEMROMHandler::SolverType::DIRECT) || nonlinear_mode)
      return;

   assert(romMat_mono);
   delete romMat_hypre, mumps;

   // TODO: need to change when the actual parallelization is implemented.
   sys_glob_size = romMat_mono->NumRows();
   sys_row_starts[0] = 0;
   sys_row_starts[1] = romMat_mono->NumRows();
   romMat_hypre = new HypreParMatrix(MPI_COMM_WORLD, sys_glob_size, sys_row_starts, romMat_mono);

   mumps = new MUMPSSolver();
   mumps->SetMatrixSymType(mat_type);
   mumps->SetOperator(*romMat_hypre);
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
