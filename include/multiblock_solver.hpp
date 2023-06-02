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

#ifndef SCALEUPROM_MULTIBLOCK_SOLVER_HPP
#define SCALEUPROM_MULTIBLOCK_SOLVER_HPP

#include "topology_handler.hpp"
#include "interfaceinteg.hpp"
#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "rom_handler.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class MultiBlockSolver
{

friend class ParameterizedProblem;

protected:
   /*
      Base variables needed for all systems (potentially)
   */
   int order = 1;

   bool full_dg = true;
   int skip_zeros = 1;

   TopologyHandlerMode topol_mode = NUM_TOPOL_MODE;
   TopologyHandler *topol_handler = NULL;

   // MultiBlockSolver does not own these. Owned by TopologyHandler.
   Mesh *pmesh = NULL;  // parent mesh. only available from SubMeshTopologyHandler.
   Array<Mesh*> meshes;

   // Informations received from Topology Handler.
   int numSub;   // number of subdomains.
   int dim;      // Spatial dimension.
   Array<int> global_bdr_attributes;   // boundary attributes of global system.

   // Solution dimension, by default -1 (scalar).
   int udim = -1;       // vector dimension of the entire solution variable
   int num_var = -1;    // number of variables
   Array<int> vdim;     // vector dimension of each variable
   std::vector<std::string> var_names;

   Array<int> block_offsets;  // Size(numSub * udim + 1). each block corresponds to a component of vector solution.
   Array<int> var_offsets; // Size(num_var * numSub + 1). each block corresponds to the variable of the solution in each domain.
   Array<int> domain_offsets; // Size(numSub + 1). each block corresponds to the vector solution in each domain.
   Array<int> num_vdofs;       // Size(numSub). number of vdofs of the vector solution in each subdomain.
   BlockVector *U, *RHS;

   // Each Variable is separated by distinct FiniteElementSpace.
   Array<FiniteElementCollection *> fec;  // Size(num_var);
   Array<FiniteElementSpace *> fes;       // Size(num_var * numSub);
   Array<GridFunction *> us;              // Size(num_var * numSub);

   int max_bdr_attr;
   int numBdr;
   Array<Array<int> *> bdr_markers;

   // MFEM solver options
   bool use_amg;

   // visualization variables
   bool save_visual = false;
   bool unified_paraview = false;
   std::string visual_dir = ".";
   std::string visual_prefix;
   Array<ParaViewDataCollection *> paraviewColls;
   // Used only for the unified visualization. Size(num_var).
   Array<FiniteElementSpace *> global_fes;
   Array<GridFunction *> global_us_visual;

   // rom variables.
   ROMHandler *rom_handler = NULL;
   bool use_rom = false;

public:
   MultiBlockSolver();

   virtual ~MultiBlockSolver();

   // Parse some base input options. 
   void ParseInputs();

   // access
   const int GetDim() const { return dim; }
   const int GetNumSubdomains() const { return numSub; }
   Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   GridFunction* GetGridFunction(const int k) { return us[k]; }
   const int GetDiscretizationOrder() const { return order; }
   const bool UseRom() const { return use_rom; }
   ROMHandler* GetROMHandler() const { return rom_handler; }
   const bool IsVisualizationSaved() const { return save_visual; }
   const std::string GetVisualizationPrefix() const { return visual_prefix; }
   const TopologyHandlerMode GetTopologyMode() const { return topol_mode; }
   ParaViewDataCollection* GetParaViewColl(const int &k) { return paraviewColls[k]; }

   void GetVariableVector(const int &var_idx, BlockVector &global, BlockVector &var);
   void SetVariableVector(const int &var_idx, BlockVector &var, BlockVector &global);

   virtual void SetupBCVariables();
   virtual void AddBCFunction(std::function<double(const Vector &)> F, const int battr = -1)
   { mfem_error("Abstract method MultiBlockSolver::AddBCFunction!\n"); }
   virtual void AddBCFunction(const double &F, const int battr = -1)
   { mfem_error("Abstract method MultiBlockSolver::AddBCFunction!\n"); }
   virtual void AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr = -1)
   { mfem_error("Abstract method MultiBlockSolver::AddBCFunction!\n"); }
   virtual void AddBCFunction(const Vector &F, const int battr = -1)
   { mfem_error("Abstract method MultiBlockSolver::AddBCFunction!\n"); }
   virtual void InitVariables() = 0;

   virtual void BuildOperators() = 0;
   virtual void BuildRHSOperators() = 0;
   virtual void BuildDomainOperators() = 0;
   
   virtual void SetupBCOperators() = 0;
   virtual void SetupRHSBCOperators() = 0;
   virtual void SetupDomainBCOperators() = 0;

   virtual void AddRHSFunction(std::function<double(const Vector &)> F)
   { mfem_error("Abstract method MultiBlockSolver::AddRHSFunction!\n"); }
   virtual void AddRHSFunction(const double F)
   { mfem_error("Abstract method MultiBlockSolver::AddRHSFunction!\n"); }
   virtual void AddRHSFunction(std::function<void(const Vector &, Vector &)> F)
   { mfem_error("Abstract method MultiBlockSolver::AddRHSFunction!\n"); }
   virtual void AddRHSFunction(const Vector &F)
   { mfem_error("Abstract method MultiBlockSolver::AddRHSFunction!\n"); }

   virtual void Assemble() = 0;
   virtual void AssembleRHS() = 0;
   virtual void AssembleOperator() = 0;
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrixes() = 0;

   // BilinearForm interface operator.
   void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *fes1, FiniteElementSpace *fes2,
      InterfaceNonlinearFormIntegrator *interface_integ,
      Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats);
      
   // MixedBilinearForm interface operator.
   void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *trial_fes1, FiniteElementSpace *trial_fes2,
      FiniteElementSpace *test_fes1, FiniteElementSpace *test_fes2, 
      InterfaceNonlinearFormIntegrator *interface_integ,
      Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats);

   // Component-wise assembly
   void GetComponentFESpaces(Array<FiniteElementSpace *> &comp_fes);
   virtual void AllocateROMElements() = 0;
   virtual void BuildROMElements() = 0;
   virtual void SaveROMElements(const std::string &filename) = 0;
   virtual void LoadROMElements(const std::string &filename) = 0;
   virtual void AssembleROM() = 0;

   virtual void Solve() = 0;

   virtual void InitVisualization(const std::string& output_dir = "");
   virtual void InitUnifiedParaview(const std::string &file_prefix);
   virtual void InitIndividualParaview(const std::string &file_prefix);
   virtual void SaveVisualization();

   void InitROMHandler();
   virtual void SaveSnapshot(const int &sample_index);
   void FormReducedBasis(const int &total_samples)
   { rom_handler->FormReducedBasis(total_samples); }
   void LoadReducedBasis() { rom_handler->LoadReducedBasis(); }
   virtual void ProjectOperatorOnReducedBasis() = 0;
   virtual void ProjectRHSOnReducedBasis();
   virtual void SolveROM();
   virtual void SaveBasisVisualization()
   { rom_handler->SaveBasisVisualization(fes, var_names); }

   virtual void SetParameterizedProblem(ParameterizedProblem *problem) = 0;

   void ComputeSubdomainErrorAndNorm(GridFunction *fom_sol, GridFunction *rom_sol, double &error, double &norm);
   double ComputeRelativeError(Array<GridFunction *> fom_sols, Array<GridFunction *> rom_sols);
   double CompareSolution();
};

#endif
