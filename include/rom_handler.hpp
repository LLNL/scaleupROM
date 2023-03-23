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
   int numSub;          // number of subdomains.
   int num_basis_sets;  // number of the basis sets.
   Array<int> fom_num_dofs;

   // rom options.
   bool save_proj_inv = false;
   bool save_lspg_basis = false;
   ROMHandlerMode mode = NUM_HANDLERMODE;
   TrainMode train_mode = NUM_TRAINMODE;
   // ProjectionMode proj_mode = NUM_PROJMODE;

   // file names.
   std::string sample_dir;
   std::string sample_prefix;
   std::string basis_prefix;
   std::string proj_inv_prefix;

   // rom variables.
   // TODO: need Array<int> for multi-component basis.
   int num_basis;    // number of columns in a basis set
   Array<const CAROM::Matrix*> carom_spatialbasis;
   bool basis_loaded;
   bool proj_inv_loaded;

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

public:
   ROMHandler(const int &input_numSub, const Array<int> &input_num_dofs);

   virtual ~ROMHandler() {};

   // access
   const int GetNumSubdomains() { return numSub; }
   
   // cannot do const GridFunction* due to librom function definitions.
   virtual void SaveSnapshot(Array<GridFunction*> &us, const int &sample_index);
   virtual void FormReducedBasis(const int &total_samples);
   virtual void LoadReducedBasis();
   virtual void GetReducedBasis(const int &subdomain_index, const CAROM::Matrix* &basis);
   virtual void SetBlockSizes();
   virtual void AllocROMMat();  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS);
   virtual void Solve(BlockVector* U);
   // void CompareSolution();

   const std::string GetSnapshotPrefix(const int &sample_idx, const int &subdomain_idx)
   { return sample_dir + "/" + sample_prefix + "_sample" + std::to_string(sample_idx) + "_dom" + std::to_string(subdomain_idx); }

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes)
   { mfem_error("Base ROMHandler does not support saving visualization!\n"); }
};

class MFEMROMHandler : public ROMHandler
{
protected:
   // rom variables.
   Array<DenseMatrix*> spatialbasis;

   SparseMatrix *romMat;
   
   mfem::BlockVector *reduced_rhs;

public:
   MFEMROMHandler(const int &input_numSub, const Array<int> &input_num_dofs);

   virtual ~MFEMROMHandler() {};
   
   // cannot do const GridFunction* due to librom function definitions.
   // virtual void FormReducedBasis(const int &total_samples);
   virtual void LoadReducedBasis();
   virtual void GetReducedBasis(const int &subdomain_index, DenseMatrix* &basis);
   // virtual void AllocROMMat() override;  // allocate matrixes for rom.
   // TODO: extension to nonlinear operators.
   virtual void ProjectOperatorOnReducedBasis(const Array2D<SparseMatrix*> &mats);
   virtual void ProjectRHSOnReducedBasis(const BlockVector* RHS);
   virtual void Solve(BlockVector* U);
   // void CompareSolution();

   virtual void SaveBasisVisualization(const Array<FiniteElementSpace *> &fes);
};


} // namespace mfem

#endif
