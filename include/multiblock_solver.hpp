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
#include "interfaceinteg.hpp"
#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "rom_handler.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "mfem/Utilities.hpp"

namespace mfem
{

enum DecompositionMode
{
   NODD,       // no decomposition
   IP,         // interior penalty
   FETI,       // finite-element tearing and interconnecting
   NUM_DDMODE
};

class MultiBlockSolver
{
public:
   struct InterfaceInfo {
      int Attr;
      int Mesh1, Mesh2;
      int BE1, BE2;

      // Inf = 64 * LocalFaceIndex + FaceOrientation
      // From the parent mesh.
      // Boundary face only have Elem1, and its orientation is always 0 by convention.
      // This causes a problem for interface between two meshes.
      // Thus stores orientation information from the parent mesh.
      int Inf1, Inf2;
   };

protected:
// public:
   int order = 1;
   // Finite element collection for all fe spaces.
   FiniteElementCollection *fec;
   // Finite element spaces
   Array<FiniteElementSpace *> fes;

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;

   bool full_dg = true;
   DecompositionMode dd_mode;

   // Global parent mesh that will be decomposed.
   Mesh *pmesh;

   // SubMesh does not allow creating Array of its pointers. Use std::shared_ptr.
   std::vector<std::shared_ptr<SubMesh>> meshes;
   int numSub;   // number of subdomains.

   // Spatial dimension.
   int dim;

   // face/element map from each subdomain to parent mesh.
   Array<Array<int> *> parent_face_map;
   Array<Array<int> *> parent_elem_map;

   Array<InterfaceInfo> interface_infos;
   // Array<int> interface_parent;

   bool strong_bc = false;
   Array<Array<int> *> ess_attrs;
   Array<Array<int> *> ess_tdof_lists;
   Array<Coefficient *> bdr_coeffs;

   int max_bdr_attr;
   Array<Array<int> *> bdr_markers;

   Array<int> block_offsets, num_dofs;
   BlockVector *U, *RHS;
   // For nonlinear problem
   // BlockOperator *globalMat;
   BlockMatrix *globalMat;
   SparseMatrix *globalMat_mono;

   Array<GridFunction *> us;

   // operators
   Array<LinearForm *> bs;
   Array<BilinearForm *> as;

   // interface integrator
   InterfaceNonlinearFormIntegrator *interface_integ;
   int skip_zeros = 1;

   // rhs coefficients
   Array<Coefficient *> rhs_coeffs;

   // DG parameters specific to Poisson equation.
   double sigma = -1.0;
   double kappa = -1.0;

   // MFEM solver options
   bool use_monolithic;

   // visualization variables
   bool save_visual = false;
   std::string visual_output;
   Array<ParaViewDataCollection *> paraviewColls;

   // rom variables.
   ROMHandler *rom_handler = NULL;
   bool use_rom = false;

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

   void InitVisualization();
   void SaveVisualization()
   { if (!save_visual) return; for (int m = 0; m < numSub; m++) paraviewColls[m]->Save(); };

   // TODO: some other form of interface?
   void SetParameterizedProblem(ParameterizedProblem *problem);

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
   void CompareSolution();

   void SanityCheckOnCoeffs();
};


} // namespace mfem

#endif
