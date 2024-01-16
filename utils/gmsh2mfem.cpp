// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);
static int    order_ = 3;

int main(int argc, char *argv[])
{
   const char *meshFileString = "test.msh";
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&meshFileString, "-m", "--mesh-name",
                  "File name for the output mesh.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(meshFileString);

   // Promote to high order mesh
   if (order_ > 1)
      mesh.SetCurvature(order_, true, 2, Ordering::byVDIM);

   std::string outputFile(meshFileString);
   outputFile += ".mfem";
   ofstream ofs(outputFile.c_str());
   ofs.precision(8);
   mesh.Print(ofs);
   ofs.close();

   // Clean up and exit
   return 0;
}
