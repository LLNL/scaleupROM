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

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{
   const char *meshFileString = "test.msh";
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&meshFileString, "-m", "--mesh-name",
                  "File name for the output mesh.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(meshFileString);

   std::string outputFile(meshFileString);
   outputFile += ".mfem";
   ofstream ofs(outputFile.c_str());
   ofs.precision(15);
   mesh.Print(ofs);
   ofs.close();

   // Clean up and exit
   return 0;
}
