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

#include "interfaceinteg.hpp"
#include "mfem.hpp"

namespace mfem
{

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

// protected:
public:
   int order = 1;
   // Finite element collection for all fe spaces.
   FiniteElementCollection *fec;
   // Finite element spaces
   Array<FiniteElementSpace *> fes;

   // System matrix for Bilinear case.
   Array2D<SparseMatrix *> mats;

   bool full_dg = true;

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
   // This class does not own these Coefficient objects!
   Array<Coefficient *> bdr_coeffs;

   int max_bdr_attr;
   Array<Array<int> *> bdr_markers;

   Array<int> block_offsets;
   BlockVector *U, *RHS;
   BlockOperator *globalMat;

   Array<GridFunction *> us;

   // operators
   Array<LinearForm *> bs;
   Array<BilinearForm *> as;

   // interface integrator
   InterfaceNonlinearFormIntegrator *interface_integ;
   int skip_zeros = 1;

   // rhs coefficients
   Array<Coefficient *> rhs_coeffs;

   // parameters specific to Poisson equation.
   double sigma = -1.0;
   double kappa = -1.0;

   Array<ParaViewDataCollection *> paraviewColls;

public:
   MultiBlockSolver() {};

   // constructor using command line inputs.
   MultiBlockSolver(int argc, char *argv[]);
   // RubberOperator(Array<FiniteElementSpace *> &fes, Array<Array<int> *>&ess_bdr,
   //                Array<int> &block_trueOffsets, double rel_tol, double abs_tol,
   //                int iter, Coefficient &mu);

   // // Required to use the native newton solver
   // virtual Operator &GetGradient(const Vector &xp) const;
   // virtual void Mult(const Vector &k, Vector &y) const;

   // // Driver for the newton solver
   // void Solve(Vector &xp) const;

   virtual ~MultiBlockSolver();

   // SubMesh does not support face mapping for 2d meshes.
   Array<int> BuildFaceMap2D(const Mesh& pm, const SubMesh& sm);
   void BuildSubMeshBoundary2D(const Mesh& pm, SubMesh& sm, Array<int> *parent_face_map=NULL);
   void UpdateBdrAttributes(Mesh& m);

   void BuildInterfaceInfos();
   Array<int> FindParentInterfaceInfo(const int pface,
                                       const int imesh, const int ibe,
                                       const int jmesh, const int jbe);

   // This class does not own these Coefficient objects!
   void SetupBoundaryConditions(Array<Coefficient *> &bdr_coeffs_in);
   void InitVariables();

   void BuildOperators();
   void SetupBCOperators();

   void Assemble();
   // For bilinear case.
   void AssembleInterfaceMatrix();

   void GetInterfaceTransformations(Mesh *m1, Mesh *m2, const InterfaceInfo *if_info,
                                    FaceElementTransformations *tr1, FaceElementTransformations *tr2);

   void Solve();

   void InitVisualization();
   void SaveVisualization()
   { for (int m = 0; m < numSub; m++) paraviewColls[m]->Save(); };
};


} // namespace mfem

#endif
