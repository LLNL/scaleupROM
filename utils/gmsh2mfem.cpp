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
