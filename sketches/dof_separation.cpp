// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "etc.hpp"

using namespace std;
using namespace mfem;

struct DofTag
{
   Array<int> battr;

   DofTag(const Array<int> &battrs)
      : battr(battrs)
   { battr.Sort(); }

   bool operator==(const DofTag &dtag) const
   {
      if (battr.Size() != dtag.battr.Size())
         return false;

      for (int k = 0; k < battr.Size(); k++)
         if (battr[k] != dtag.battr[k])
            return false;

      return true;
   }

   bool operator<(const DofTag &dtag) const
   {
      if (battr.Size() == dtag.battr.Size())
      {
         for (int k = 0; k < battr.Size(); k++)
            if (battr[k] != dtag.battr[k])
               return (battr[k] < dtag.battr[k]);
         
         return false;
      }
      else
         return (battr.Size() < dtag.battr.Size());
   }

   std::string print() const
   {
      std::string tag = "(";
      for (int k = 0 ; k < battr.Size(); k++)
      {
         tag += std::to_string(battr[k]) + ",";
      }
      tag += ")";
      
      return tag;
   }
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int refine = 0;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   // FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   L2_FECollection l2_coll(order, dim);
   H1_FECollection h1_coll(order+1, dim);
   L2_FECollection pl2_coll(order, dim);
   H1_FECollection ph1_coll(order, dim);

   FiniteElementSpace fes(&mesh, &h1_coll);
   FiniteElementSpace ufes(&mesh, &l2_coll, dim);
   FiniteElementSpace pfes(&mesh, &ph1_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(dim+2); // number of variables + 1
   block_offsets[0] = 0;
   for (int d = 1; d <= dim; d++)
      block_offsets[d] = ufes.GetNDofs();
   block_offsets[dim+1] = pfes.GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   for (int d = 1; d < block_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, block_offsets[d] - block_offsets[d-1]);
   printf("dim(q) = %d\n", block_offsets.Last());
   std::cout << "***********************************************************\n";

   printf("ufes vsize: %d\n", ufes.GetVSize());
   printf("pfes vsize: %d\n", pfes.GetVSize());

   Array<int> udofs(ufes.GetVSize());
   Array<int> pdofs(pfes.GetVSize());
   udofs = 0;
   pdofs = 0;

   int max_battr = -1;
   for (int i = 0; i < pfes.GetNBE(); i++)
      max_battr = max(max_battr, mesh.GetBdrAttribute(i));

   Array2D<int> pbattr(pfes.GetVSize(), max_battr);
   pbattr = 0;

   Array<int> vdofs;
   FaceElementTransformations *tr;
   printf("pfes nbe: %d\n", pfes.GetNBE());
   for (int i = 0; i < pfes.GetNBE(); i++)
   {
      const int bdr_attr = mesh.GetBdrAttribute(i);

      pfes.GetBdrElementVDofs(i, vdofs);
      for (int k = 0 ; k < vdofs.Size(); k++)
         printf("%d\t", vdofs[k]);

      tr = mesh.GetBdrFaceTransformations(i);
      if (tr != NULL)
      {
         pfes.GetElementVDofs(tr->Elem1No, vdofs);
         printf(" / ");
         for (int k = 0 ; k < vdofs.Size(); k++)
         {
            printf("%d\t", vdofs[k]);
            pdofs[vdofs[k]] += 1;
            pbattr(vdofs[k], bdr_attr-1) = 1;
         }
      }
      printf("\n");
   }

   for (int k = 0; k < pdofs.Size(); k++)
   {
      printf("pdofs[%d] = %d / ", k, pdofs[k]);
      int nbattr = 0;
      for (int d = 0; d < max_battr; d++)
      {
         printf("%d ", pbattr(k, d));
         nbattr += pbattr(k, d);
      }
      printf(": %d\n", nbattr);
   }

   std::map<DofTag, Array<int>> pdofbattr;
   for (int k = 0; k < pfes.GetVSize(); k++)
   {
      Array<int> battrs(0);
      for (int d = 0; d < max_battr; d++)
      {
         if (pbattr(k, d) > 0)
            battrs.Append(d+1);
      }
      battrs.Sort();

      DofTag pdtag(battrs);
      if (!pdofbattr.count(pdtag))
         pdofbattr[pdtag] = Array<int>(0);

      pdofbattr[pdtag].Append(k);
   }

   for(std::map<DofTag, Array<int>>::iterator iter = pdofbattr.begin(); iter != pdofbattr.end(); ++iter)
   {
      const DofTag *key = &(iter->first);
      const Array<int> *val = &(iter->second);

      printf("%s: ", key->print().c_str());
      for (int k = 0 ; k < val->Size(); k++)
      {
         printf("%d ", (*val)[k]);
      }
      printf("\n");
   }

   return 0;
}