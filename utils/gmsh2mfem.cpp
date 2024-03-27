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
static bool    force_nc_ = false;

int main(int argc, char *argv[])
{
   const char *meshFileString = "test.msh";
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&meshFileString, "-m", "--mesh-name",
                  "File name for the output mesh.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&force_nc_, "-fnc", "--force-non-conforming", "-nfnc", "--noforce-non-conforming",
                  "Sets whether to force the output mesh to be nonconforming. Default behavior is no enforcing.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh(meshFileString);

   // Promote to high order mesh and ensure discontinuity
   if (order_ >1)
      mesh.SetCurvature(order_, true, 2, Ordering::byVDIM);
   
   // Force mesh to be nonconforming
   if (force_nc_)
   {
      mesh.SetCurvature(mesh.GetNodalFESpace()->GetMaxElementOrder(), true, 2, Ordering::byVDIM);
   }
   
   std::string outputFile(meshFileString);
   outputFile += ".mfem";
   ofstream ofs(outputFile.c_str());
   ofs.precision(8);
   mesh.Print(ofs);
   ofs.close();

   // Clean up and exit
   return 0;
}
