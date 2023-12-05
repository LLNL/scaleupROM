// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

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
static int    ser_ref_levels = 0;
static int    periodic_dir = 2; // 0 for x only, 1 for y only, 2 for both
static int    dx_    = 1; // number of subdomains in x direction
static int    dy_    = 1; // number of subdomains in y direction

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);

int main(int argc, char *argv[])
{

   int el_type = 4;
   bool per_mesh = true;
   bool dg_mesh  = true;
   const char *meshFileString = "test.mesh";
   //bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx_, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&ny_, "-ny", "--num-elements-y",
                  "Number of elements in y-direction.");
   args.AddOption(&dx_, "-dx", "--num-subdomains-x",
                  "Number of subdomains in x-direction.");
   args.AddOption(&dy_, "-dy", "--num-subdomains-y",
                  "Number of subdomains in y-direction.");
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
   args.AddOption(&meshFileString, "-m", "--mesh-name",
                  "File name for the output mesh.");
printf("%s\n", meshFileString);
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

   // check all subdomains will have the same size.
   if ((nx_ % dx_ != 0) or (ny_ % dy_ != 0)) {
     cout << "Each domain should have the same size, i.e. nx_ % dx_ == 0 and ny_ % dy_ == 0!" << endl;
     exit(1);
   }

   // Define the mesh
   std::cout << a_ << ", " << b_ << std::endl;
   Mesh orig_mesh = Mesh::MakeCartesian2D(nx_, ny_, el_type_, false, a_, b_, false);

   // Promote to high order mesh
   if (order_ > 1 || dg_mesh || per_mesh)
   {
      orig_mesh.SetCurvature(order_, dg_mesh || per_mesh, 2, Ordering::byVDIM);
   }

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

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Set element attributes based on subdomains.
   int Nx = nx_ * pow(2, ser_ref_levels);
   int Ny = ny_ * pow(2, ser_ref_levels);
   int snx = Nx / dx_;
   int sny = Ny / dy_;
   double sa = a_ / dx_;
   double sb = b_ / dy_;
   int numElem = mesh.GetNE();
   printf("number of elements: %d\n", numElem);
   printf("number of elements in each subdomain: (%d, %d)\n", snx, sny);
   for (int el = 0; el < numElem; el++) {
       Vector center(2);
       mesh.GetElementCenter(el, center);

       int skx = floor(center(0) / sa);
       int sky = floor(center(1) / sb);

       int attr = 1 + skx + sky * dx_; // Attribute starts from 1.
       // printf("Element %d attribute: %d\n", el, attr);
       mesh.SetAttribute(el, attr);
   }

   for (int el = 0; el < numElem; el++) {
     int attr = mesh.GetAttribute(el);
     Vector center(2);
     mesh.GetElementCenter(el, center);
     printf("Element %d: %d - (%f, %f)\n", el, attr, center(0), center(1));
   }

   // Output the resulting mesh to a file
   if (strlen(meshFileString) == 0) {
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
      meshFileString = oss.str().c_str();
   }
   ofstream ofs(meshFileString);
   ofs.precision(8);
   mesh.Print(ofs);
   ofs.close();

   // Clean up and exit
   return 0;
}
