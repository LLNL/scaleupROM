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

namespace mfem
{

enum ROMHandlerMode
{
   SAMPLE_GENERATION,
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
   int num_basis_sets = -1;  // number of the basis sets. for individual case, ==numSub. for universal case, == number of components.
   Array<int> fom_num_vdofs;

   // rom options.
   bool save_sv = false;
   bool save_basis_visual = false;
   bool component_sampling = false;
   bool save_lspg_basis = false;
   bool basis_file_exists = false;
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
   // TODO: need Array<int> for multi-component basis.
   int num_basis;    // number of columns in a basis set
   Array<const CAROM::Matrix*> carom_spatialbasis;
   bool basis_loaded;
   bool operator_loaded;

   // SparseMatrix *romMat;
   Array<int> rom_block_offsets;
   
   CAROM::Vector *reduced_rhs;

   Array2D<CAROM::Matrix *> carom_mats;
   CAROM::Matrix *romMat_inv;
   
   CAROM::Options* rom_options;
   CAROM::BasisGenerator *basis_generator;
   CAROM::BasisReader *basis_reader;

   int max_num_snapshots = 100;
   bool update_right_SV = false;
   bool incremental = false;

   void ParseInputs();
public:
   ROMHandler(TopologyHandler *input_topol, const int &input_udim, const Array<int> &input_num_vdofs);

   virtual ~ROMHandler() {};

   // access
   const int GetNumSubdomains() { return numSub; }
   const ROMHandlerMode GetMode() { return mode; }
   const TrainMode GetTrainMode() { return train_mode; }
   // TODO: multi-component case
   const int GetNumBasis(const int &basis_idx) { return num_basis; }
   const bool UseExistingBasis() { return basis_file_exists; }
   const ROMBuildingLevel SaveOperator() { return save_operator; }
   const bool BasisLoaded() { return basis_loaded; }
   const bool OperatorLoaded() { return operator_loaded; }
   const std::string GetOperatorPrefix() { return operator_prefix; }
   
   // cannot do const GridFunction* due to librom function definitions.
   virtual void SaveSnapshot(Array<GridFunction*> &us, const int &sample_index);

   virtual void FormReducedBasis(const int &total_samples);
   virtual void FormReducedBasisUniversal(const int &total_samples);
   virtual void FormReducedBasisIndividual(const int &total_samples);

   virtual void LoadReducedBasis();
   virtual void GetBasisOnSubdomain(const int &subdomain_index, const CAROM::Matrix* &basis);
   virtual void SetBlockSizes();
   virtual void AllocROMMat();  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS);
   virtual void Solve(BlockVector* U);
   // void CompareSolution();

   // P_i^T * mat * P_j
   virtual void ProjectOperatorOnReducedBasis(const int &i, const int &j, const SparseMatrix *mat, DenseMatrix *proj_mat)
   { mfem_error("ROMHandler::ProjectOperatorOnReducedBasis(const int &, const int &, SparseMatrix *) is not supported!\n"); }

   virtual void LoadOperatorFromFile(const std::string input_prefix="");
   virtual void LoadOperator(SparseMatrix *input_mat)
   { mfem_error("ROMHandler::LoadOperator is not supported!\n"); }

   const std::string GetSnapshotPrefix(const int &sample_idx, const int &subdomain_idx);

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes)
   { if (save_basis_visual) mfem_error("Base ROMHandler does not support saving visualization!\n"); }

   virtual void SaveSV(const std::string& prefix);
};

class MFEMROMHandler : public ROMHandler
{
protected:
   // rom variables.
   Array<DenseMatrix*> spatialbasis;

   SparseMatrix *romMat;
   
   mfem::BlockVector *reduced_rhs;

public:
   MFEMROMHandler(TopologyHandler *input_topol, const int &input_udim, const Array<int> &input_num_vdofs);

   virtual ~MFEMROMHandler() {}; 
   
   // cannot do const GridFunction* due to librom function definitions.
   // virtual void FormReducedBasis(const int &total_samples);
   virtual void LoadReducedBasis();
   virtual void GetBasis(const int &basis_index, DenseMatrix* &basis);
   virtual void GetBasisOnSubdomain(const int &subdomain_index, DenseMatrix* &basis);
   // virtual void AllocROMMat() override;  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS);
   virtual void Solve(BlockVector* U);
   // void CompareSolution();
   
   // P_i^T * mat * P_j
   virtual void ProjectOperatorOnReducedBasis(const int &i, const int &j, const SparseMatrix *mat, DenseMatrix *proj_mat);

   virtual void LoadOperatorFromFile(const std::string input_prefix="");
   virtual void LoadOperator(SparseMatrix *input_mat);

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes);
};


} // namespace mfem

#endif
