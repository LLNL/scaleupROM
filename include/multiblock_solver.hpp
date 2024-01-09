// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_MULTIBLOCK_SOLVER_HPP
#define SCALEUPROM_MULTIBLOCK_SOLVER_HPP

#include "topology_handler.hpp"
#include "interfaceinteg.hpp"
#include "interface_form.hpp"
#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "rom_handler.hpp"
#include "hdf5_utils.hpp"

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
   bool direct_solve = false;

   // Saving solution in single run
   bool save_sol = false;
   std::string sol_dir = ".";
   std::string sol_prefix;

   // visualization variables
   bool save_visual = false;
   bool unified_paraview = false;
   int visual_offset = -1, visual_freq = -1;    // for individual_paraview, support partial visualization.
   std::string visual_dir = ".";
   std::string visual_prefix;
   Array<ParaViewDataCollection *> paraviewColls;
   // Used only for the unified visualization. Size(num_var).
   Array<FiniteElementSpace *> global_fes;
   Array<GridFunction *> global_us_visual;

   // rom variables.
   ROMHandler *rom_handler = NULL;
   TrainMode train_mode = NUM_TRAINMODE;
   bool use_rom = false;
   bool separate_variable_basis = false;

   // Used for bottom-up building, only with ComponentTopologyHandler.
   // For now, assumes ROM basis represents the entire vector solution.
   Array<SparseMatrix *> comp_mats;     // Size(num_components);
   // boundary condition is enforced via forcing term.
   Array<Array<SparseMatrix *> *> bdr_mats;
   Array<Array2D<SparseMatrix *> *> port_mats;   // reference ports.
   // DenseTensor objects from nonlinear operators
   // will be defined per each derived MultiBlockSolver.

public:
   MultiBlockSolver();

   virtual ~MultiBlockSolver();

   // Parse some base input options. 
   void ParseInputs();

   // access
   const int GetDim() const { return dim; }
   const int GetNumSubdomains() const { return numSub; }
   const int GetNumBdr() const { return numBdr; }
   Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   GridFunction* GetGridFunction(const int k) { return us[k]; }
   const int GetDiscretizationOrder() const { return order; }
   const bool UseRom() const { return use_rom; }
   ROMHandler* GetROMHandler() const { return rom_handler; }
   const TrainMode GetTrainMode() { return train_mode; }
   const bool IsVisualizationSaved() const { return save_visual; }
   const std::string GetSolutionFilePrefix() const { return sol_prefix; }
   const std::string GetVisualizationPrefix() const { return visual_prefix; }
   const TopologyHandlerMode GetTopologyMode() const { return topol_mode; }
   ParaViewDataCollection* GetParaViewColl(const int &k) { return paraviewColls[k]; }
   BlockVector* GetSolution() { return U; }
   BlockVector* GetSolutionCopy() { return new BlockVector(*U); }

   void GetVariableVector(const int &var_idx, BlockVector &global, BlockVector &var);
   void SetVariableVector(const int &var_idx, BlockVector &var, BlockVector &global);

   void SortBySubdomains(BlockVector &by_var, BlockVector &by_sub);
   void SortByVariables(BlockVector &by_sub, BlockVector &by_var);

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
   
   virtual bool BCExistsOnBdr(const int &global_battr_idx) = 0;
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

   // Global ROM operator Loading.
   virtual void LoadROMOperatorFromFile(const std::string input_prefix="")
   { rom_handler->LoadOperatorFromFile(input_prefix); }

   // Component-wise assembly
   void GetComponentFESpaces(Array<FiniteElementSpace *> &comp_fes);
   void AllocateROMElements();

   void BuildROMElements();
   virtual void BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp) = 0;
   virtual void BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp) = 0;
   // TODO(kevin): part of this can be transferred to InterfaceForm.
   virtual void BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp) = 0;

   void SaveROMElements(const std::string &filename);
   // Save ROM Elements in a hdf5-format file specified with file_id.
   // TODO: add more arguments to support more general data structures of ROM Elements.
   virtual void SaveCompBdrROMElement(hid_t &file_id);
   void SaveBdrROMElement(hid_t &comp_grp_id, const int &comp_idx);
   void SaveInterfaceROMElement(hid_t &file_id);

   void LoadROMElements(const std::string &filename);
   // Load ROM Elements in a hdf5-format file specified with file_id.
   // TODO: add more arguments to support more general data structures of ROM Elements.
   virtual void LoadCompBdrROMElement(hid_t &file_id);
   void LoadBdrROMElement(hid_t &comp_grp_id, const int &comp_idx);
   void LoadInterfaceROMElement(hid_t &file_id);

   void AssembleROM();

   virtual bool Solve() = 0;

   virtual void InitVisualization(const std::string& output_dir = "");
   virtual void InitUnifiedParaview(const std::string &file_prefix);
   virtual void InitIndividualParaview(const std::string &file_prefix);
   virtual void SaveVisualization();

   void SaveSolution(std::string filename = "");
   void LoadSolution(const std::string &filename);
   void CopySolution(BlockVector *input_sol);

   void InitROMHandler();
   void GetBasisTags(std::vector<std::string> &basis_tags);

   virtual void PrepareSnapshots(BlockVector* &U_snapshots, std::vector<std::string> &basis_tags);
   void LoadReducedBasis() { rom_handler->LoadReducedBasis(); }
   virtual void ProjectOperatorOnReducedBasis() = 0;
   virtual void ProjectRHSOnReducedBasis();
   virtual void SolveROM();
   virtual void SaveBasisVisualization()
   { rom_handler->SaveBasisVisualization(fes, var_names); }

   virtual void SetParameterizedProblem(ParameterizedProblem *problem) = 0;

   void ComputeSubdomainErrorAndNorm(GridFunction *fom_sol, GridFunction *rom_sol, double &error, double &norm);
   double ComputeRelativeError(Array<GridFunction *> fom_sols, Array<GridFunction *> rom_sols);
   double CompareSolution(BlockVector &test_U);
};

#endif
