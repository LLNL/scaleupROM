// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

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

enum ROMHandlerMode
{
   SAMPLE_GENERATION,
   TRAIN_ROM,
   BUILD_ROM,
   SINGLE_RUN,
   NUM_HANDLERMODE
};

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

// enum ProjectionMode
// {
//    GALERKIN,
//    LSPG,
//    NUM_PROJMODE
// };

class ROMHandler
{
protected:
// public:
   int numSub = -1;          // number of subdomains.
   int udim = -1;            // solution dimension.
   int num_var = -1;         // number of variables for which linear subspace method is applied.
   int num_basis_sets = -1;  // number of the basis sets. for individual case, ==numSub. for universal case, == number of components.
   Array<int> vdim;          // dimension of each variable.
   Array<int> fom_num_vdofs;

   // rom options.
   bool save_sv = false;
   bool save_basis_visual = false;
   bool component_sampling = false;
   bool save_lspg_basis = false;
   ROMBuildingLevel save_operator = NUM_BLD_LVL;
   ROMHandlerMode mode = NUM_HANDLERMODE;
   TrainMode train_mode = NUM_TRAINMODE;
   // ProjectionMode proj_mode = NUM_PROJMODE;

   // file names.
   std::string sample_dir;
   std::string sample_prefix;
   std::string basis_prefix;
   std::string operator_prefix;

   // topology handler
   TopologyHandler *topol_handler = NULL;

   // rom variables.
   Array<int> num_basis;    // number of columns in a basis set
   Array<const CAROM::Matrix*> carom_spatialbasis;
   bool basis_loaded;
   bool operator_loaded;

   // SparseMatrix *romMat;
   Array<int> rom_block_offsets;
   
   CAROM::Vector *reduced_rhs = NULL;
   CAROM::Vector *reduced_sol = NULL;

   Array2D<CAROM::Matrix *> carom_mats;
   CAROM::Matrix *romMat_inv = NULL;
   
   CAROM::Options* rom_options;
   CAROM::BasisGenerator *basis_generator;
   CAROM::BasisReader *basis_reader;

   int max_num_snapshots = 100;
   bool update_right_SV = false;
   bool incremental = false;

   void ParseInputs();
public:
   ROMHandler(TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_num_vdofs);

   virtual ~ROMHandler();

   // access
   const int GetNumSubdomains() { return numSub; }
   const ROMHandlerMode GetMode() { return mode; }
   const TrainMode GetTrainMode() { return train_mode; }
   const int GetNumBasisSets() { return num_basis_sets; }
   const int GetNumBasis(const int &basis_idx) { return num_basis[basis_idx]; }
   const ROMBuildingLevel SaveOperator() { return save_operator; }
   const bool BasisLoaded() { return basis_loaded; }
   const bool OperatorLoaded() { return operator_loaded; }
   const std::string GetOperatorPrefix() { return operator_prefix; }
   const Array<int>* GetBlockOffsets() { return &rom_block_offsets; }

   // virtual void FormReducedBasis();
   virtual void LoadReducedBasis();

   int GetBasisIndexForSubdomain(const int &subdomain_index);
   void GetBasis(const int &basis_index, const CAROM::Matrix* &basis);
   virtual void GetBasisOnSubdomain(const int &subdomain_index, const CAROM::Matrix* &basis);
   virtual void SetBlockSizes();
   virtual void AllocROMMat();  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats);
   virtual void ProjectVectorOnReducedBasis(const BlockVector* vec, CAROM::Vector*& rom_vec);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS)
   {
      printf("Project RHS on reduced basis.\n");
      ProjectVectorOnReducedBasis(RHS, reduced_rhs);
   }
   virtual void Solve(BlockVector* U);

   // P_i^T * mat * P_j
   virtual void ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat, CAROM::Matrix *proj_mat);
   virtual SparseMatrix* ProjectOperatorOnReducedBasis(const int &i, const int &j, const Operator *mat)
   { mfem_error("ROMHandler::ProjectOperatorOnReducedBasis(...)\n"); return NULL; }

   virtual void LoadOperatorFromFile(const std::string input_prefix="");
   virtual void LoadOperator(BlockMatrix *input_mat)
   { mfem_error("ROMHandler::LoadOperator is not supported!\n"); }

   const std::string GetBasisTagForComponent(const int &comp_idx);
   const std::string GetBasisTag(const int &subdomain_index);
   void GetBasisTags(std::vector<std::string> &basis_tags);

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes, const std::vector<std::string> &var_names)
   { if (save_basis_visual) mfem_error("Base ROMHandler does not support saving visualization!\n"); }

   // virtual void SaveSV(const std::string& prefix, const int& basis_idx);
   virtual void SaveReducedSolution(const std::string &filename)
   { CAROM::PrintVector(*reduced_sol, filename); }
   virtual void SaveReducedRHS(const std::string &filename)
   { CAROM::PrintVector(*reduced_rhs, filename); }
};

class MFEMROMHandler : public ROMHandler
{
protected:
   // rom variables.
   Array<DenseMatrix*> spatialbasis;

   BlockMatrix *romMat = NULL;
   SparseMatrix *romMat_mono = NULL;
   
   mfem::BlockVector *reduced_rhs = NULL;
   mfem::BlockVector *reduced_sol = NULL;

public:
   MFEMROMHandler(TopologyHandler *input_topol, const Array<int> &input_vdim, const Array<int> &input_num_vdofs);

   virtual ~MFEMROMHandler();
   
   // cannot do const GridFunction* due to librom function definitions.
   // virtual void FormReducedBasis(const int &total_samples);
   virtual void LoadReducedBasis();
   void GetBasis(const int &basis_index, DenseMatrix* &basis);
   void GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis);
   // virtual void AllocROMMat();  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<Operator*> &mats);
   virtual void ProjectVectorOnReducedBasis(const BlockVector* vec, CAROM::Vector*& rom_vec) override
   { mfem_error("MFEMROMHandler::ProjectVectorOnReducedBasis - base class method is called!\n"); }
   virtual void ProjectVectorOnReducedBasis(const BlockVector* vec, mfem::BlockVector*& rom_vec);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS) override
   { ProjectVectorOnReducedBasis(RHS, reduced_rhs); }
   virtual void Solve(BlockVector* U);
   
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
   IterativeSolver* SetSolver(const std::string &solver_type, const std::string &prec_type);
   // void GetBlockSparsity(const SparseMatrix *mat, const Array<int> &block_offsets, Array2D<bool> &mat_zero_blocks);
   // bool CheckZeroBlock(const DenseMatrix &mat);
};


} // namespace mfem

#endif
