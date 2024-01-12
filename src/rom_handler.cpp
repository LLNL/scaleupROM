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
   const int &comp_idx, const TrainMode &train_mode, const TopologyHandler *topol_handler, const std::string var_name)
{
   std::string tag;
   switch (train_mode)
   {
      case (INDIVIDUAL):   { tag = "dom" + std::to_string(comp_idx); break; }
      case (UNIVERSAL):    { tag = topol_handler->GetComponentName(comp_idx); break; }
      default:
      {
         mfem_error("ROMHandler::GetBasisTagForComponent - Unknown training mode!\n");
         break;
      }
   }
   if (var_name != "")
      tag += "_" + var_name;
   return tag;
}

const std::string GetBasisTag(
   const int &subdomain_index, const TrainMode &train_mode, const TopologyHandler *topol_handler, const std::string var_name)
{
   std::string tag;
   switch (train_mode)
   {
      case (INDIVIDUAL):
      {
         tag = "dom" + std::to_string(subdomain_index);
         break;
      }
      case (UNIVERSAL):
      {
         int c_type = topol_handler->GetMeshType(subdomain_index);
         tag = topol_handler->GetComponentName(c_type);
         break;
      }
      default:
      {
         mfem_error("ROMHandler::GetBasisTag - Unknown training mode!\n");
         break;
      }
   }
   if (var_name != "")
      tag += "_" + var_name;
   return tag;
}

ROMHandlerBase::ROMHandlerBase(
   const TrainMode &train_mode_, TopologyHandler *input_topol, const Array<int> &input_var_offsets,
   const std::vector<std::string> &var_names, const bool separate_variable_basis)
   : train_mode(train_mode_),
     topol_handler(input_topol),
     numSub(input_topol->GetNumSubdomains()),
     fom_var_names(var_names),
     fom_var_offsets(input_var_offsets),
     basis_loaded(false),
     operator_loaded(false),
     separate_variable(separate_variable_basis)
{
   num_var = fom_var_names.size();

   fom_num_vdofs.SetSize(numSub);
   for (int m = 0, idx = 0; m < numSub; m++)
   {
      fom_num_vdofs[m] = 0;
      for (int v = 0; v < num_var; v++, idx++)
         fom_num_vdofs[m] += fom_var_offsets[idx+1] - fom_var_offsets[idx];
   }

   ParseInputs();

   SetBlockSizes();
}

ROMHandlerBase::~ROMHandlerBase()
{
   delete reduced_rhs;
   delete reduced_sol;
}

void ROMHandlerBase::ParseInputs()
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
   if (train_mode == TrainMode::INDIVIDUAL)        num_rom_ref_blocks = numSub;
   else if (train_mode == TrainMode::UNIVERSAL)    num_rom_ref_blocks = topol_handler->GetNumComponents();
   else
      mfem_error("ROMHandler - subdomain training mode is not set!\n");

   num_rom_comp = num_rom_ref_blocks;

   if (separate_variable)
   {
      num_rom_ref_blocks *= num_var;
      num_rom_blocks *= num_var;
   }
   
   const int num_ref_basis_default = config.GetOption<int>("basis/number_of_basis", -1);
   num_ref_basis.SetSize(num_rom_ref_blocks);
   num_ref_basis = num_ref_basis_default;
   assert(num_ref_basis.Size() > 0);

   YAML::Node basis_list = config.FindNode("basis/tags");
   if ((!basis_list) && (num_ref_basis_default <= 0))
      mfem_error("ROMHandler - cannot find the basis tag list, nor default number of basis is not set!\n");

   for (int b = 0; b < num_rom_ref_blocks; b++)
   {
      std::string basis_tag;
      if (separate_variable)
         basis_tag = GetBasisTagForComponent(b / num_var, train_mode, topol_handler, fom_var_names[b % num_var]);
      else
         basis_tag = GetBasisTagForComponent(b, train_mode, topol_handler);

      num_ref_basis[b] = config.LookUpFromDict("name", basis_tag, "number_of_basis", num_ref_basis_default, basis_list);

      if (num_ref_basis[b] < 0)
      {
         printf("Cannot find the number of basis for %s!\n", basis_tag.c_str());
         mfem_error("Or specify the default number of basis!\n");
      }
   }
   
   for (int k = 0; k < num_ref_basis.Size(); k++)
      assert(num_ref_basis[k] > 0);

   max_num_snapshots = config.GetOption<int>("sample_generation/maximum_number_of_snapshots", 100);
   update_right_SV = config.GetOption<bool>("basis/svd/update_right_sv", false);

   save_sv = config.GetOption<bool>("basis/svd/save_spectrum", false);

   save_basis_visual = config.GetOption<bool>("basis/visualization/enabled", false);
}

void ROMHandlerBase::LoadReducedBasis()
{
   if (basis_loaded) return;

   std::string basis_name;
   int numRowRB, numColumnRB;

   carom_ref_basis.SetSize(num_rom_ref_blocks);
   for (int k = 0; k < num_rom_ref_blocks; k++)
   {
      basis_name = basis_prefix + "_";
      if (separate_variable)
         basis_name += GetBasisTagForComponent(k / num_var, train_mode, topol_handler, fom_var_names[k % num_var]);
      else
         basis_name += GetBasisTagForComponent(k, train_mode, topol_handler);
      basis_reader = new CAROM::BasisReader(basis_name);

      carom_ref_basis[k] = basis_reader->getSpatialBasis(0.0, num_ref_basis[k]);
      numRowRB = carom_ref_basis[k]->numRows();
      numColumnRB = carom_ref_basis[k]->numColumns();
      printf("spatial basis-%d dimension is %d x %d\n", k, numRowRB, numColumnRB);

      delete basis_reader;
   }

   basis_loaded = true;
}

int ROMHandlerBase::GetBasisIndexForSubdomain(const int &subdomain_index)
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

void ROMHandlerBase::SetBlockSizes()
{
   rom_block_offsets.SetSize(num_rom_blocks+1);
   rom_varblock_offsets.SetSize(num_rom_blocks+1);
   rom_comp_block_offsets.SetSize(num_rom_ref_blocks+1);
   
   rom_block_offsets = 0;
   rom_varblock_offsets = 0;
   rom_comp_block_offsets = 0;

   if (separate_variable)
   {
      for (int c = 0, idx = 0; c < num_rom_comp; c++)
         for (int v = 0; v < num_var; v++, idx++)
            rom_comp_block_offsets[idx+1] = num_ref_basis[idx];

      for (int k = 0, idx = 0; k < numSub; k++)
      {
         int c = topol_handler->GetMeshType(k);
         for (int v = 0; v < num_var; v++, idx++)
         {
            rom_block_offsets[idx+1] = num_ref_basis[c * num_var + v];
            rom_varblock_offsets[1 + k + v * numSub] = num_ref_basis[c * num_var + v];
         }
      }
   }
   else
   {
      for (int c = 0; c < num_rom_comp; c++)
         rom_comp_block_offsets[c+1] = num_ref_basis[c];

      for (int k = 0; k < numSub; k++)
      {
         int c = topol_handler->GetMeshType(k);
         rom_block_offsets[k+1] = num_ref_basis[c];
      }
      rom_varblock_offsets = rom_block_offsets;
   }

   rom_block_offsets.PartialSum();
   rom_comp_block_offsets.PartialSum();
   rom_varblock_offsets.PartialSum();
}

/*
   MFEMROMHandler
*/

MFEMROMHandler::MFEMROMHandler(const TrainMode &train_mode_, TopologyHandler *input_topol, const Array<int> &input_var_offsets, const std::vector<std::string> &var_names, const bool separate_variable_basis)
   : ROMHandlerBase(train_mode_, input_topol, input_var_offsets, var_names, separate_variable_basis)
{
   romMat = new BlockMatrix(rom_block_offsets);
   romMat->owns_blocks = true;

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
   DeletePointers(ref_basis);
   delete romMat;
   delete romMat_mono;
   delete reduced_rhs;
   delete reduced_sol;
   delete romMat_hypre;
   delete mumps;
}

void MFEMROMHandler::LoadReducedBasis()
{
   ROMHandlerBase::LoadReducedBasis();

   ref_basis.SetSize(carom_ref_basis.Size());
   ref_basis = NULL;
   for (int k = 0; k < ref_basis.Size(); k++)
   {
      assert(carom_ref_basis[k] != NULL);
      ref_basis[k] = new DenseMatrix(carom_ref_basis[k]->numRows(), carom_ref_basis[k]->numColumns());
      CAROM::CopyMatrix(*carom_ref_basis[k], *ref_basis[k]);
   }

   dom_basis.SetSize(num_rom_blocks);
   for (int k = 0; k < num_rom_blocks; k++)
   {
      if (train_mode == TrainMode::INDIVIDUAL)
      {
         dom_basis[k] = ref_basis[k];
         continue;
      }

      int m = (separate_variable) ? k / num_var : k;
      int c = topol_handler->GetMeshType(m);
      int v = (separate_variable) ? k % num_var : 0;
      int idx = (separate_variable) ? c * num_var + v : c;
      dom_basis[k] = ref_basis[idx];
   }

   basis_loaded = true;
}

void MFEMROMHandler::GetReferenceBasis(const int &basis_index, DenseMatrix* &basis)
{
   assert(num_rom_ref_blocks > 0);
   assert((basis_index >= 0) && (basis_index < num_rom_ref_blocks));

   basis = ref_basis[basis_index];
}

void MFEMROMHandler::GetDomainBasis(const int &basis_index, DenseMatrix* &basis)
{
   assert(basis_loaded);
   assert((basis_index >= 0) && (basis_index < num_rom_blocks));

   basis = dom_basis[basis_index];
}

void MFEMROMHandler::GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis)
{
   assert(basis_loaded);

   int idx = GetBasisIndexForSubdomain(subdomain_index);
   GetReferenceBasis(idx, basis);
}

void MFEMROMHandler::ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats)
{
   assert(mats.NumRows() == num_rom_blocks);
   assert(mats.NumCols() == num_rom_blocks);
   assert(romMat->NumRowBlocks() == num_rom_blocks);
   assert(romMat->NumColBlocks() == num_rom_blocks);
   assert(basis_loaded);

   Array2D<SparseMatrix *> rom_mats;
   ProjectGlobalToDomainBasis(mats, rom_mats);
   for (int i = 0; i < num_rom_blocks; i++)
      for (int j = 0; j < num_rom_blocks; j++)
         romMat->SetBlock(i, j, rom_mats(i, j));

   romMat->Finalize();
   SetRomMat(romMat);
}

void MFEMROMHandler::SaveOperator(const std::string input_prefix)
{
   assert(save_operator == ROMBuildingLevel::GLOBAL);
   assert(operator_loaded && romMat);

   std::string filename;
   if (input_prefix == "")
      filename = operator_prefix;
   else
      filename = input_prefix;
   filename += ".h5";

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   hdf5_utils::WriteBlockMatrix(file_id, "ROM_matrix", romMat);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
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

SparseMatrix* MFEMROMHandler::ProjectToRefBasis(const int &i, const int &j, const Operator *mat)
{
   assert((i >= 0) && (i < num_rom_ref_blocks));
   assert((j >= 0) && (j < num_rom_ref_blocks));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   DenseMatrix *basis_i, *basis_j;
   GetReferenceBasis(i, basis_i);
   GetReferenceBasis(j, basis_j);

   assert(basis_i->NumRows() == mat->Height());
   assert(basis_j->NumRows() == mat->Width());

   if (mat)
      return mfem::SparseRtAP(*basis_i, *mat, *basis_j);
   else
      return new SparseMatrix(basis_i->NumCols(), basis_j->NumCols());
}

SparseMatrix* MFEMROMHandler::ProjectToDomainBasis(const int &i, const int &j, const Operator *mat)
{
   assert((i >= 0) && (i < num_rom_blocks));
   assert((j >= 0) && (j < num_rom_blocks));
   // assert(mat->Finalized());
   assert(basis_loaded);
   
   DenseMatrix *basis_i, *basis_j;
   GetDomainBasis(i, basis_i);
   GetDomainBasis(j, basis_j);

   assert(basis_i->NumRows() == mat->Height());
   assert(basis_j->NumRows() == mat->Width());

   if (mat)
      return mfem::SparseRtAP(*basis_i, *mat, *basis_j);
   else
      return new SparseMatrix(basis_i->NumCols(), basis_j->NumCols());
}

void MFEMROMHandler::ProjectToRefBasis(
   const Array<int> &idx_i, const Array<int> &idx_j, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   assert(basis_loaded);
   assert((idx_i.Min() >= 0) && (idx_i.Max() < num_rom_ref_blocks));
   assert((idx_j.Min() >= 0) && (idx_j.Max() < num_rom_ref_blocks));
   assert(idx_i.Size() == mats.NumRows());
   assert(idx_j.Size() == mats.NumCols());

   /* Clean up existing rom_mats */
   for (int i = 0; i < rom_mats.NumRows(); i++)
      for (int j = 0; j < rom_mats.NumCols(); j++)
         if (rom_mats(i, j)) delete rom_mats(i, j);
   
   rom_mats.SetSize(mats.NumRows(), mats.NumCols());
   for (int i = 0; i < rom_mats.NumRows(); i++)
      for (int j = 0; j < rom_mats.NumCols(); j++)
         rom_mats(i, j) = ProjectToRefBasis(idx_i[i], idx_j[j], mats(i, j));
}

void MFEMROMHandler::ProjectToDomainBasis(
   const Array<int> &idx_i, const Array<int> &idx_j, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   assert(basis_loaded);
   assert((idx_i.Min() >= 0) && (idx_i.Max() < num_rom_blocks));
   assert((idx_j.Min() >= 0) && (idx_j.Max() < num_rom_blocks));
   assert(idx_i.Size() == mats.NumRows());
   assert(idx_j.Size() == mats.NumCols());

   /* Clean up existing rom_mats */
   for (int i = 0; i < rom_mats.NumRows(); i++)
      for (int j = 0; j < rom_mats.NumCols(); j++)
         if (rom_mats(i, j)) delete rom_mats(i, j);
   
   rom_mats.SetSize(mats.NumRows(), mats.NumCols());
   for (int i = 0; i < rom_mats.NumRows(); i++)
      for (int j = 0; j < rom_mats.NumCols(); j++)
         rom_mats(i, j) = ProjectToDomainBasis(idx_i[i], idx_j[j], mats(i, j));
}

void MFEMROMHandler::ProjectComponentToRefBasis(
   const int &c, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   int size = (separate_variable) ? num_var : 1;
   Array<int> idx(size);
   idx = c;
   if (separate_variable)
      for (int v = 0; v < num_var; v++)
      {
         idx[v] *= num_var;
         idx[v] += v;
      }

   assert((mats.NumRows() == size) && (mats.NumCols() == size));

   ProjectToRefBasis(idx, idx, mats, rom_mats);
}

void MFEMROMHandler::ProjectComponentToDomainBasis(
   const int &c, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   int size = (separate_variable) ? num_var : 1;
   Array<int> idx(size);
   idx = c;
   if (separate_variable)
      for (int v = 0; v < num_var; v++)
      {
         idx[v] *= num_var;
         idx[v] += v;
      }

   assert((mats.NumRows() == size) && (mats.NumCols() == size));

   ProjectToDomainBasis(idx, idx, mats, rom_mats);
}

void MFEMROMHandler::ProjectInterfaceToRefBasis(
   const int &c1, const int &c2, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   int size = (separate_variable) ? 2 * num_var : 2;
   Array<int> idx(size);
   if (separate_variable)
      for (int v = 0; v < num_var; v++)
      {
         idx[v] = c1 * num_var + v;
         idx[v + num_var] = c2 * num_var + v;
      }
   else
   {
      idx[0] = c1;
      idx[1] = c2;
   }

   assert((mats.NumRows() == size) && (mats.NumCols() == size));

   ProjectToRefBasis(idx, idx, mats, rom_mats);
}

void MFEMROMHandler::ProjectInterfaceToDomainBasis(
   const int &c1, const int &c2, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   int size = (separate_variable) ? 2 * num_var : 2;
   Array<int> idx(size);
   if (separate_variable)
      for (int v = 0; v < num_var; v++)
      {
         idx[v] = c1 * num_var + v;
         idx[v + num_var] = c2 * num_var + v;
      }
   else
   {
      idx[0] = c1;
      idx[1] = c2;
   }

   assert((mats.NumRows() == size) && (mats.NumCols() == size));

   ProjectToDomainBasis(idx, idx, mats, rom_mats);
}

void MFEMROMHandler::ProjectVariableToDomainBasis(
   const int &vi, const int &vj, const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   assert(separate_variable);

   int size = numSub;
   Array<int> idx_i(size), idx_j(size);
   for (int m = 0; m < numSub; m++)
   {
      idx_i[m] = m * num_var + vi;
      idx_j[m] = m * num_var + vj;
   }

   assert((mats.NumRows() == size) && (mats.NumCols() == size));

   ProjectToDomainBasis(idx_i, idx_j, mats, rom_mats);
}

void MFEMROMHandler::ProjectGlobalToDomainBasis(const Array2D<Operator*> &mats, Array2D<SparseMatrix *> &rom_mats)
{
   assert((mats.NumRows() == num_rom_blocks) && (mats.NumCols() == num_rom_blocks));

   Array<int> idx(num_rom_blocks);
   for (int k = 0; k < num_rom_blocks; k++) idx[k] = k;

   ProjectToDomainBasis(idx, idx, mats, rom_mats);
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
   SetRomMat(romMat);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
}

void MFEMROMHandler::SetRomMat(BlockMatrix *input_mat)
{
   if (romMat != input_mat)
   {
      delete romMat;
      romMat = input_mat;
   }

   delete romMat_mono;
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

   for (int c = 0; c < num_rom_ref_blocks; c++)
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

      Array<int> var_offsets(num_ref_basis[c] * num_var + 1);
      var_offsets[0] = 0;
      for (int k = 0, vidx = 1; k < num_ref_basis[c]; k++)
         for (int v = 0, idx = midx * num_var; v < num_var; v++, idx++, vidx++)
            var_offsets[vidx] = fes[idx]->GetVSize();
      var_offsets.PartialSum();
      BlockVector basis_view(ref_basis[c]->GetData(), var_offsets);

      Array<GridFunction*> basis_gf(num_ref_basis[c] * num_var);
      basis_gf = NULL;
      for (int k = 0, idx = 0; k < num_ref_basis[c]; k++)
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
