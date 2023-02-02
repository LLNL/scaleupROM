// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//          -------------------------------------------------------
//          Twist Miniapp:  Generate simple twisted periodic meshes
//          -------------------------------------------------------
//
// This miniapp generates simple periodic meshes to demonstrate MFEM's handling
// of periodic domains. MFEM's strategy is to use a discontinuous vector field
// to define the mesh coordinates on a topologically periodic mesh. It works by
// defining a stack of individual elements and stitching together the top and
// bottom of the mesh. The stack can also be twisted so that the vertices of the
// bottom and top can be joined with any integer offset (for tetrahedral and
// wedge meshes only even offsets are supported).
//
// Compile with: make twist
//
// Sample runs:  twist
//               twist -no-pm
//               twist -nt -2 -no-pm
//               twist -nt 2 -e 4
//               twist -nt 2 -e 6
//               twist -nt 3 -e 8
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Element::Type el_type_ = Element::QUADRILATERAL;
static int    order_ = 3;
static int    nx_    = 5;
static int    ny_    = 3;
static double a_     = 5.0;
static double b_     = 1.0;
static int    ser_ref_levels = 3;
static int    periodic_dir = 2; // 0 for x only, 1 for y only, 2 for both
// static double c_     = 3.0;

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{

   int el_type = 4;
   bool per_mesh = true;
   bool dg_mesh  = true;
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx_, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny_, "-ny", "--num-unit-elements",
                  "Number of elements in y-direction.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&a_, "-a", "--base-x",
                  "Width of the base in x-direction.");
   args.AddOption(&b_, "-b", "--base-y",
                  "Width of the base in y-direction.");
   args.AddOption(&periodic_dir, "-dir", "--direction",
                  "direction for periodicity for periodic mesh.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 4 - Quadrilateral, 6 - Triangle.");
   args.AddOption(&per_mesh, "-pm", "--periodic-mesh",
                  "-no-pm", "--non-periodic-mesh",
                  "Enforce periodicity.");
   // args.AddOption(&dg_mesh, "-dm", "--discont-mesh", "-cm", "--cont-mesh",
   //                "Use discontinuous or continuous space for the mesh nodes.");
   // args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
   //                "--no-visualization",
   //                "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // The output mesh could be tetrahedra, hexahedra, or prisms
   switch (el_type)
   {
      case 4:
         el_type_ = Element::QUADRILATERAL;
         break;
      case 6:
         el_type_ = Element::TRIANGLE;
         break;
      default:
         cout << "Unsupported element type" << endl;
         exit(1);
         break;
   }

   // Define the mesh
   std::cout << a_ << ", " << b_ << std::endl;
   Mesh orig_mesh = Mesh::MakeCartesian2D(nx_, ny_, el_type_, false, a_, b_, false);

   Mesh mesh;
   if (per_mesh) {
     std::vector<int> v2v(orig_mesh.GetNV());
     std::vector<Vector> translations;
     switch (periodic_dir) {
       case 0:
         translations = {Vector({a_,0.0})};
         break;
       case 1:
         translations = {Vector({0.0,b_})};
         break;
       case 2:
         translations = {Vector({a_,0.0}), Vector({0.0,b_})};
         break;
       default:
         cout << "Unsupported periodic direction" << endl;
         exit(1);
         break;
     }
     v2v = orig_mesh.CreatePeriodicVertexMapping(translations);
     mesh = Mesh::MakePeriodic(orig_mesh,v2v);
     std::cout << "created periodic mesh." << std::endl;
   } else {
     mesh = Mesh(orig_mesh);
   }

  // Promote to high order mesh
  if (order_ > 1 || dg_mesh || per_mesh)
  {
     orig_mesh.SetCurvature(order_, dg_mesh || per_mesh, 2, Ordering::byVDIM);
  }

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      if (el_type_ == Element::QUADRILATERAL)
      {
         oss << "rectangle-quad";
      }
      else
      {
         oss << "rectangle-tri";
      }
      oss << "-o" << order_;
      if (ser_ref_levels > 0)
      {
         oss << "-r" << ser_ref_levels;
      }
      if (per_mesh)
      {
        switch (periodic_dir) {
          case 0:
            oss << "-xp";
            break;
          case 1:
            oss << "-yp";
            break;
          case 2:
            oss << "-p";
            break;
        }
      }
      else if (dg_mesh)
      {
         oss << "-d";
      }
      else
      {
         oss << "-c";
      }

      oss << ".mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh.Print(ofs);
      ofs.close();
   }

   // Clean up and exit
   return 0;
}
