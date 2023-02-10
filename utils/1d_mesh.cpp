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

static int    nx_    = 5;
static double a_     = 5.0;
static int    ser_ref_levels = 3;
static bool per_mesh = true;

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{

   const char *meshFileString = "";
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx_, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&a_, "-a", "--base-x",
                  "Width of the base in x-direction.");
   args.AddOption(&per_mesh, "-pm", "--periodic-mesh",
                  "-no-pm", "--non-periodic-mesh",
                  "Enforce periodicity.");
   args.AddOption(&meshFileString, "-m", "--mesh-name",
                  "File name for the output mesh.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Define the mesh
   Mesh orig_mesh = Mesh::MakeCartesian1D(nx_, a_);

   Mesh mesh;
   if (per_mesh) {
     std::vector<int> v2v(orig_mesh.GetNV());
     std::vector<Vector> translations;
     translations = {Vector({a_,0.0})};
     v2v = orig_mesh.CreatePeriodicVertexMapping(translations);
     mesh = Mesh::MakePeriodic(orig_mesh,v2v);
     std::cout << "created periodic mesh." << std::endl;
   } else {
     mesh = Mesh(orig_mesh);
   }

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   int numElem = mesh.GetNE();
   printf("number of elements: %d\n", numElem);
   for (int el = 0; el < numElem; el++) {
     int attr = mesh.GetAttribute(el);
     Vector center(2);
     mesh.GetElementCenter(el, center);
     printf("Element %d: %d - (%f, %f)\n", el, attr, center(0), center(1));
   }

   // Output the resulting mesh to a file
   if (strlen(meshFileString) == 0) {
      ostringstream oss;
      oss << "1d";
      if (ser_ref_levels > 0)
      {
         oss << "-r" << ser_ref_levels;
      }
      if (per_mesh)
      {
        oss << "-p";
      }

      oss << ".mesh";
      meshFileString = oss.str().c_str();
   }
   ofstream ofs(meshFileString);
   ofs.precision(8);
   mesh.Print(ofs);
   ofs.close();

   // Clean up and exit
   return 0;
}
