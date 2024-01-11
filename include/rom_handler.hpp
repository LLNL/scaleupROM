// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef ROM_HANDLER_HPP
#define ROM_HANDLER_HPP

#include "mfem.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "mfem/Utilities.hpp"
#include "topology_handler.hpp"
#include "linalg_utils.hpp"

namespace mfem
{

enum TrainMode
{
   INDIVIDUAL,
   UNIVERSAL,
   NUM_TRAINMODE
};

enum ROMBuildingLevel
{
   NONE,
   COMPONENT,
   GLOBAL,
   NUM_BLD_LVL
};

const TrainMode SetTrainMode();

const std::string GetBasisTagForComponent(const int &comp_idx, const TrainMode &train_mode, const TopologyHandler *topol_handler, const std::string var_name="");
const std::string GetBasisTag(const int &subdomain_index, const TrainMode &train_mode, const TopologyHandler *topol_handler, const std::string var_name="");

class ROMHandlerBase
{
protected:
// public:
   int numSub = -1;          // number of subdomains.
   int num_var = -1;         // number of variables for which POD is performed.
   int num_rom_blocks = -1;  // number of ROM blocks for the global domain.
   int num_rom_ref_blocks = -1;  // number of ROM reference component blocks.
   int num_rom_comp = -1;     // number of ROM reference components.
   std::vector<std::string> fom_var_names;          // dimension of each variable.
   Array<int> fom_var_offsets;
   Array<int> fom_num_vdofs;

   // rom options.
   bool save_sv = false;
   bool save_basis_visual = false;
   bool component_sampling = false;
   bool save_lspg_basis = false;
   ROMBuildingLevel save_operator = NUM_BLD_LVL;
   TrainMode train_mode = NUM_TRAINMODE;
   bool nonlinear_mode = false;
   bool separate_variable = false;
   // ProjectionMode proj_mode = NUM_PROJMODE;

   // file names.
   std::string sample_dir;
   std::string sample_prefix;
   std::string basis_prefix;
   std::string operator_prefix;

   // topology handler
   TopologyHandler *topol_handler = NULL;

   // component rom variables.
   /*
      number of columns in the basis for a reference component.
      For i-th reference component and j-th variable,
         index = i * num_var + j
   */
   Array<int> num_ref_basis;
   Array<const CAROM::Matrix*> carom_ref_basis;
   Array<int> rom_comp_block_offsets;
   bool basis_loaded;
   bool operator_loaded;

   // domain rom variables.
   /*
      offset for the global domain ROM blocks.
      For i-th subdomain and j-th variable,
         index = i * num_var + j
   */
   Array<int> num_basis;
   Array<const CAROM::Matrix*> carom_basis; // This is only the pointers to carom_ref_basis. no need of deleting.
   Array<int> rom_block_offsets;
   
   CAROM::Vector *reduced_rhs = NULL;
   CAROM::Vector *reduced_sol = NULL;
   
   CAROM::Options* rom_options;
   CAROM::BasisGenerator *basis_generator;
   CAROM::BasisReader *basis_reader;

   int max_num_snapshots = 100;
   bool update_right_SV = false;
   bool incremental = false;

   void ParseInputs();
public:
   ROMHandlerBase(const TrainMode &train_mode_, TopologyHandler *input_topol,
              const Array<int> &input_var_offsets, const std::vector<std::string> &var_names, const bool separate_variable_basis);

   virtual ~ROMHandlerBase();

   // access
   const int GetNumSubdomains() { return numSub; }
   const TrainMode GetTrainMode() { return train_mode; }
   const int GetNumROMRefBlocks() { return num_rom_ref_blocks; }
   const int GetComponentNumBasis(const int &basis_idx) { return num_ref_basis[basis_idx]; }
   const ROMBuildingLevel SaveOperator() { return save_operator; }
   const bool BasisLoaded() { return basis_loaded; }
   const bool OperatorLoaded() { return operator_loaded; }
   const std::string GetOperatorPrefix() { return operator_prefix; }
   const Array<int>* GetBlockOffsets() { return &rom_block_offsets; }
   virtual SparseMatrix* GetOperator() = 0;
   const bool GetNonlinearMode() { return nonlinear_mode; }
   void SetNonlinearMode(const bool nl_mode) { nonlinear_mode = nl_mode; }

   virtual void LoadReducedBasis();

   int GetBasisIndexForSubdomain(const int &subdomain_index);
   virtual void GetReferenceBasis(const int &basis_index, DenseMatrix* &basis) = 0;
   virtual void GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis) = 0;
   virtual void SetBlockSizes();

   // virtual void ReferenceRTAP(const Array<int> &ridxs, const Array<int> &pidxs, const Array2D<Operator*> &mats, Array2D<CAROM::Matrix *> &rom_mats);
   // virtual void DomainRTAP(const Array<int> &ridxs, const Array<int> &pidxs, const Array2D<Operator*> &mats, Array2D<CAROM::Matrix *> &rom_mats);

   virtual void ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats) = 0;
   virtual void ProjectVectorOnReducedBasis(const BlockVector* vec, mfem::BlockVector*& rom_vec) = 0;
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS) = 0;

   virtual void Solve(BlockVector* U) = 0;
   virtual void NonlinearSolve(Operator &oper, BlockVector* U, Solver *prec=NULL) = 0;

   // P_i^T * mat * P_j
   virtual SparseMatrix* ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat) = 0;

   virtual void LoadOperatorFromFile(const std::string input_prefix="") = 0;
   virtual void LoadOperator(BlockMatrix *input_mat) = 0;

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes, const std::vector<std::string> &var_names) = 0;

   virtual void SaveReducedSolution(const std::string &filename) = 0;
   virtual void SaveReducedRHS(const std::string &filename) = 0;
};

class MFEMROMHandler : public ROMHandlerBase
{
protected:
   // type for linear system solver.
   enum SolverType
   {
      DIRECT,
      CG,
      MINRES,
      GMRES,
      NUM_SOLVERTYPE
   } linsol_type;
   MUMPSSolver::MatType mat_type;

   // component rom variables.
   Array<DenseMatrix*> ref_basis;

   // domain rom variables.
   Array<DenseMatrix*> basis; // This is only the pointers to ref_basis. no need of deleting.

   BlockMatrix *romMat = NULL;
   SparseMatrix *romMat_mono = NULL;

   // variables needed for direct solve
   HYPRE_BigInt sys_glob_size;
   HYPRE_BigInt sys_row_starts[2];
   HypreParMatrix *romMat_hypre = NULL;
   MUMPSSolver *mumps = NULL;
   
   mfem::BlockVector *reduced_rhs = NULL;
   mfem::BlockVector *reduced_sol = NULL;

public:
   MFEMROMHandler(const TrainMode &train_mode_, TopologyHandler *input_topol,
                  const Array<int> &input_var_offsets, const std::vector<std::string> &var_names, const bool separate_variable_basis);

   virtual ~MFEMROMHandler();

   virtual SparseMatrix* GetOperator() override
   { assert(romMat_mono); return romMat_mono; }
   
   virtual void LoadReducedBasis();
   virtual void GetReferenceBasis(const int &basis_index, DenseMatrix* &basis) override;
   virtual void GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis) override;

   virtual void ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats);
   virtual void ProjectVectorOnReducedBasis(const BlockVector* vec, mfem::BlockVector*& rom_vec);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS) override
   { ProjectVectorOnReducedBasis(RHS, reduced_rhs); }
   virtual void Solve(BlockVector* U);
   virtual void NonlinearSolve(Operator &oper, BlockVector* U, Solver *prec=NULL) override;
   
   // P_i^T * mat * P_j
   virtual SparseMatrix* ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat);

   virtual void LoadOperatorFromFile(const std::string input_prefix="");
   virtual void LoadOperator(BlockMatrix *input_mat);

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes, const std::vector<std::string> &var_names);

   virtual void SaveReducedSolution(const std::string &filename) override
   { PrintVector(*reduced_sol, filename); }
   virtual void SaveReducedRHS(const std::string &filename) override
   { PrintVector(*reduced_rhs, filename); }

private:
   IterativeSolver* SetIterativeSolver(const MFEMROMHandler::SolverType &linsol_type_, const std::string &prec_type);
   // void GetBlockSparsity(const SparseMatrix *mat, const Array<int> &block_offsets, Array2D<bool> &mat_zero_blocks);
   // bool CheckZeroBlock(const DenseMatrix &mat);
   void SetupDirectSolver();
};


} // namespace mfem

#endif
