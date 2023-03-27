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

#include "input_parser.hpp"
#include "topology_handler.hpp"
#include "interfaceinteg.hpp"
#include "mfem.hpp"
// #include "parameterized_problem.hpp"
#include "rom_handler.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "mfem/Utilities.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

// enum DecompositionMode
// {
//    NODD,       // no decomposition
//    IP,         // interior penalty
//    FETI,       // finite-element tearing and interconnecting
//    NUM_DDMODE
// };

class MultiBlockSolver
{

friend class ParameterizedProblem;
friend class Poisson0;
friend class PoissonComponent;
friend class PoissonSpiral;

// public:
   // struct InterfaceInfo {
   //    int Attr;
   //    int Mesh1, Mesh2;
   //    int BE1, BE2;

   //    // Inf = 64 * LocalFaceIndex + FaceOrientation
   //    // From the parent mesh.
   //    // Boundary face only have Elem1, and its orientation is always 0 by convention.
   //    // This causes a problem for interface between two meshes.
   //    // Thus stores orientation information from the parent mesh.
   //    int Inf1, Inf2;
   // };

protected:
   /*
      Base variables needed for all systems (potentially)
   */
   int order = 1;
   // Finite element collection for all fe spaces.
   FiniteElementCollection *fec;
   // Finite element spaces
   Array<FiniteElementSpace *> fes;

   bool full_dg = true;
   DecompositionMode dd_mode;

   // Global parent mesh that will be decomposed.
   Mesh *pmesh;

   // SubMesh does not allow creating Array of its pointers. Use std::shared_ptr.
   std::vector<std::shared_ptr<SubMesh>> meshes;
   int numSub;   // number of subdomains.

   // Spatial dimension.
   int dim;
   // Solution dimension, by default 1 (scalar).
   int udim = 1;

   // face/element map from each subdomain to parent mesh.
   Array<Array<int> *> parent_face_map;
   Array<Array<int> *> parent_elem_map;

   Array<InterfaceInfo> interface_infos;
   // Array<int> interface_parent;

   // interface integrator
   InterfaceNonlinearFormIntegrator *interface_integ;
   int skip_zeros = 1;

   Array<int> block_offsets;  // Size(numSub * udim + 1). each block corresponds to a component of vector solution.
   Array<int> domain_offsets; // Size(numSub + 1). each block corresponds to the vector solution.
   Array<int> num_vdofs;       // Size(numSub). number of vdofs of the vector solution in each subdomain.
   BlockVector *U, *RHS;
   // For nonlinear problem
   // BlockOperator *globalMat;
   BlockMatrix *globalMat;
   SparseMatrix *globalMat_mono;

   Array<GridFunction *> us;

   // boundary infos
   bool strong_bc = false;
   Array<Array<int> *> ess_attrs;
   Array<Array<int> *> ess_tdof_lists;

   int max_bdr_attr;
   Array<Array<int> *> bdr_markers;

   // MFEM solver options
   bool use_monolithic;

   // visualization variables
   bool save_visual = false;
   bool unified_paraview = false;
   std::string visual_dir = ".";
   std::string visual_prefix;
   Array<ParaViewDataCollection *> paraviewColls;
   // Used only for the unified visualization.
   FiniteElementSpace *global_fes = NULL;
   GridFunction *global_us_visual = NULL;

   // rom variables.
   ROMHandler *rom_handler = NULL;
   bool use_rom = false;

   /*
      System-specific variables (will separated to derived classes)
   */

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;

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

public:
   MultiBlockSolver();

   // // constructor using command line inputs.
   // MultiBlockSolver(int argc, char *argv[]);
   // RubberOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
   //                Array<int> &block_trueOffsets, double rel_tol, double abs_tol,
   //                int iter, Coefficient &mu);

   // // Required to use the native newton solver
   // virtual Operator &GetGradient(const Vector &xp) const;
   // virtual void Mult(const Vector &k, Vector &y) const;

   // // Driver for the newton solver
   // void Solve(Vector &xp) const;

   virtual ~MultiBlockSolver();

   // Parse some base input options. 
   void ParseInputs();

   // access
   const int GetNumSubdomains() { return numSub; }
   Mesh* GetMesh(const int k) { return &(*meshes[k]); }
   GridFunction* GetGridFunction(const int k) { return us[k]; }
   const int GetDiscretizationOrder() { return order; }
   const bool UseRom() { return use_rom; }
   ROMHandler* GetROMHandler() { return rom_handler; }
   const bool IsVisualizationSaved() { return save_visual; }
   const std::string GetVisualizationPrefix() { return visual_prefix; }

   // SubMesh does not support face mapping for 2d meshes.
   Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm);
   void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map=NULL);
   void UpdateBdrAttributes(Mesh& m);

   void BuildInterfaceInfos();
   Array<int> FindParentInterfaceInfo(const int pface,
                                       const int imesh, const int ibe,
                                       const int jmesh, const int jbe);

   void SetupBCVariables();
   void AddBCFunction(std::function<double(const Vector &)> F, const int battr = -1);
   void AddBCFunction(const double &F, const int battr = -1);
   void InitVariables();

   void BuildOperators();
   // TODO: support non-homogeneous Neumann condition.
   void SetupBCOperators();

   void AddRHSFunction(std::function<double(const Vector &)> F)
   { rhs_coeffs.Append(new FunctionCoefficient(F)); }
   void AddRHSFunction(const double F)
   { rhs_coeffs.Append(new ConstantCoefficient(F)); }

   void Assemble();
   // For bilinear case.
   void AssembleInterfaceMatrix();

   // Mesh sets face element transformation based on the face_info.
   // For boundary face, the adjacent element is always on element 1, and its orientation is "by convention" always zero.
   // This is a problem for the interface between two meshes, where both element orientations are zero.
   // At least one element should reflect a relative orientation with respect to the other.
   // Currently this is done by hijacking global mesh face information in the beginning.
   // If we would want to do more flexible global mesh building, e.g. rotating component submeshes,
   // then we will need to figure out how to actually determine relative orientation.
   void GetInterfaceTransformations(Mesh *m1, Mesh *m2, const InterfaceInfo *if_info,
                                    FaceElementTransformations* &tr1, FaceElementTransformations* &tr2);

   void Solve();

   void InitVisualization(const std::string& output_dir = "");
   void InitUnifiedParaview(const std::string &file_prefix);
   void InitIndividualParaview(const std::string &file_prefix);
   void SaveVisualization();

   void InitROMHandler();
   void SaveSnapshot(const int &sample_index)
   { rom_handler->SaveSnapshot(us, sample_index); }
   void FormReducedBasis(const int &total_samples)
   { rom_handler->FormReducedBasis(total_samples); }
   void LoadReducedBasis() { rom_handler->LoadReducedBasis(); }
   void ProjectOperatorOnReducedBasis()
   { rom_handler->ProjectOperatorOnReducedBasis(mats); }
   void ProjectRHSOnReducedBasis()
   { rom_handler->ProjectRHSOnReducedBasis(RHS); }
   void SolveROM() { rom_handler->Solve(U); }
   double CompareSolution();
   void SaveBasisVisualization()
   { rom_handler->SaveBasisVisualization(fes); }

   void SanityCheckOnCoeffs();
};

#endif
