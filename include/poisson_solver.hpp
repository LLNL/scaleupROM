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

#ifndef SCALEUPROM_POISSON_SOLVER_HPP
#define SCALEUPROM_POISSON_SOLVER_HPP

// #include "input_parser.hpp"
// #include "topology_handler.hpp"
#include "multiblock_solver.hpp"
#include "interfaceinteg.hpp"
#include "mfem.hpp"
// #include "parameterized_problem.hpp"
// #include "rom_handler.hpp"
// #include "linalg/BasisGenerator.h"
// #include "linalg/BasisReader.h"
// #include "mfem/Utilities.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class PoissonSolver : public MultiBlockSolver
{

friend class ParameterizedProblem;
friend class Poisson0;
friend class PoissonComponent;
friend class PoissonSpiral;

protected:
   /*
      Base variables needed for all systems (potentially)
   */
   // int order = 1;
   // Finite element collection for all fe spaces.
   FiniteElementCollection *fec;
   // Finite element spaces
   Array<FiniteElementSpace *> fes;

   // bool full_dg = true;

   // TopologyHandlerMode topol_mode = NUM_TOPOL_MODE;
   // TopologyHandler *topol_handler = NULL;

   // // MultiBlockSolver does not own these. Owned by TopologyHandler.
   // Array<Mesh*> meshes;

   // // Informations received from Topology Handler.
   // int numSub;   // number of subdomains.
   // int dim;      // Spatial dimension.
   // Array<int> global_bdr_attributes;   // boundary attributes of global system.

   // // Solution dimension, by default 1 (scalar).
   // int udim = -1;       // vector dimension of the entire solution variable
   // int num_var = -1;    // number of variables
   // Array<int> vdim;     // vector dimension of each variable

   // 
   // Array<int> block_offsets;  // Size(numSub * udim + 1). each block corresponds to a component of vector solution.
   // Array<int> domain_offsets; // Size(numSub + 1). each block corresponds to the vector solution.
   // Array<int> num_vdofs;       // Size(numSub). number of vdofs of the vector solution in each subdomain.
   // BlockVector *U, *RHS;

   // Array<GridFunction *> us;

   // // boundary infos
   // bool strong_bc = false;
   // Array<Array<int> *> ess_attrs;
   // Array<Array<int> *> ess_tdof_lists;

   // int max_bdr_attr;
   // Array<Array<int> *> bdr_markers;

   // // MFEM solver options
   // bool use_amg;

   // // visualization variables
   // bool save_visual = false;
   // bool unified_paraview = false;
   // std::string visual_dir = ".";
   // std::string visual_prefix;
   // Array<ParaViewDataCollection *> paraviewColls;
   // // Used only for the unified visualization.
   // FiniteElementSpace *global_fes = NULL;
   // GridFunction *global_us_visual = NULL;

   // // rom variables.
   // ROMHandler *rom_handler = NULL;
   // bool use_rom = false;

   /*
      System-specific variables (will separated to derived classes)
   */

   // interface integrator
   InterfaceNonlinearFormIntegrator *interface_integ;
   // int skip_zeros = 1;

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;
   // For nonlinear problem
   // BlockOperator *globalMat;
   BlockMatrix *globalMat;
   SparseMatrix *globalMat_mono;

   // operators
   Array<LinearForm *> bs;
   Array<BilinearForm *> as;

   // rhs coefficients
   // The solution dimension is 1 by default, for which using VectorCoefficient is not allowed. (in LinearForm Assemble.)
   // For a derived class for vector solution, this is the first one needs to be changed to Array<VectorCoefficient*>.
   Array<Coefficient *> rhs_coeffs;
   Array<Coefficient *> bdr_coeffs;

   // DG parameters specific to Poisson equation.
   double sigma = -1.0;
   double kappa = -1.0;

   // Used for bottom-up building, only with ComponentTopologyHandler.
   Array<DenseMatrix *> comp_mats;
   // boundary condition is enforced via forcing term.
   Array<Array<DenseMatrix *> *> bdr_mats;
   Array<Array2D<DenseMatrix *> *> port_mats;   // reference ports.

public:
   PoissonSolver();

   virtual ~PoissonSolver();

   // // Parse some base input options. 
   // void ParseInputs();

   // // access
   // const int GetNumSubdomains() { return numSub; }
   // Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   // GridFunction* GetGridFunction(const int k) { return us[k]; }
   // const int GetDiscretizationOrder() { return order; }
   // const bool UseRom() { return use_rom; }
   // ROMHandler* GetROMHandler() { return rom_handler; }
   // const bool IsVisualizationSaved() { return save_visual; }
   // const std::string GetVisualizationPrefix() { return visual_prefix; }
   // const TopologyHandlerMode GetTopologyMode() { return topol_mode; }

   virtual void SetupBCVariables() override;
   virtual void AddBCFunction(std::function<double(const Vector &)> F, const int battr = -1);
   virtual void AddBCFunction(const double &F, const int battr = -1);
   virtual void InitVariables();

   virtual void BuildOperators();
   virtual void BuildRHSOperators();
   virtual void BuildDomainOperators();
   
   virtual void SetupBCOperators();
   virtual void SetupRHSBCOperators();
   virtual void SetupDomainBCOperators();

   virtual void AddRHSFunction(std::function<double(const Vector &)> F)
   { rhs_coeffs.Append(new FunctionCoefficient(F)); }
   virtual void AddRHSFunction(const double F)
   { rhs_coeffs.Append(new ConstantCoefficient(F)); }

   virtual void Assemble();
   virtual void AssembleRHS();
   virtual void AssembleOperator();
   // For bilinear case.
   // system-specific.
   virtual void AssembleInterfaceMatrixes();
   // // universal operator.
   // void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
   //                               FiniteElementSpace *fes1,
   //                               FiniteElementSpace *fes2,
   //                               Array<InterfaceInfo> *interface_infos,
   //                               Array2D<SparseMatrix*> &mats);

   // Component-wise assembly
   virtual void AllocateROMElements();
   virtual void BuildROMElements();
   virtual void SaveROMElements(const std::string &filename);
   virtual void LoadROMElements(const std::string &filename);
   virtual void AssembleROM();

   virtual void Solve();

   // void InitVisualization(const std::string& output_dir = "");
   void InitUnifiedParaview(const std::string &file_prefix) override;
   // void InitIndividualParaview(const std::string &file_prefix);
   // void SaveVisualization();

   // void InitROMHandler();
   // void SaveSnapshot(const int &sample_index)
   // { rom_handler->SaveSnapshot(us, sample_index); }
   // void FormReducedBasis(const int &total_samples)
   // { rom_handler->FormReducedBasis(total_samples); }
   // void LoadReducedBasis() { rom_handler->LoadReducedBasis(); }
   virtual void ProjectOperatorOnReducedBasis()
   { rom_handler->ProjectOperatorOnReducedBasis(mats); }
   // void ProjectRHSOnReducedBasis()
   // { rom_handler->ProjectRHSOnReducedBasis(RHS); }
   // void SolveROM() { rom_handler->Solve(U); }
   virtual double CompareSolution();
   virtual void SaveBasisVisualization()
   { rom_handler->SaveBasisVisualization(fes); }

   void SanityCheckOnCoeffs();

   virtual void SetParameterizedProblem(ParameterizedProblem *problem);
};

#endif
