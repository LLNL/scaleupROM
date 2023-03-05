#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include "multiblock_solver.hpp"
#include "input_parser.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double dbc2(const Vector &);
double dbc4(const Vector &);

int main(int argc, char *argv[])
{
   const char *input_file = "test.input";
   OptionsParser args(argc, argv);
   args.AddOption(&input_file, "-i", "--input", "Input file to use.");
   args.ParseCheck();
   config = InputParser(input_file);

   // MultiBlockSolver test(argc, argv);
   MultiBlockSolver test;

   test.InitVariables();
   test.InitVisualization();

   test.AddBCFunction(dbc2, 2);
   test.AddBCFunction(dbc4, 4);
   test.AddRHSFunction(1.0);

   test.BuildOperators();

   test.SetupBCOperators();

   test.Assemble();

   test.Solve();
   test.SaveVisualization();

   // for (int i = 0; i < test.numSub; i++) {
   //    printf("Submesh %d\n", i);
   //    for (int k = 0; k < test.meshes[i]->GetNBE(); k++) {
   //       printf("bdr element %d attribute: %d\n", k, test.meshes[i]->GetBdrAttribute(k));
   //    }

   //    // Setting a new boundary attribute does not append bdr_attributes.
   //    printf("submesh nbe: %d\n", test.meshes[i]->GetNBE());
   //    for (int k = 0; k < test.meshes[i]->bdr_attributes.Size(); k++) {
   //       printf("bdr attribute %d: %d\n", k, test.meshes[i]->bdr_attributes[k]);
   //    }

   //    int nfaces = test.meshes[i]->GetNumFaces();
   //    printf("submesh nfaces: %d\n", nfaces);
   // }

   // for (int i = 0; i < test.numSub; i++) {
   //    printf("Submesh %d\n", i);
   //    for (int ib = 0; ib < test.meshes[i]->GetNBE(); ib++) {
   //       int interface_attr = test.meshes[i]->GetBdrAttribute(ib);
   //       if (interface_attr <= test.pmesh->bdr_attributes.Max()) continue;

   //       int parent_face_i = (*test.parentFaceMap[i])[test.meshes[i]->GetBdrFace(ib)];
         
   //       for (int j = 0; j < test.numSub; j++) {
   //          if (i == j) continue;
   //          for (int jb = 0; jb < test.meshes[j]->GetNBE(); jb++) {
   //             int parent_face_j = (*test.parentFaceMap[j])[test.meshes[j]->GetBdrFace(jb)];
   //             if (parent_face_i == parent_face_j) {
   //             printf("(BE %d, face %d) - parent face %d, attr %d - Submesh %d (BE %d, face %d)\n",
   //                   ib, test.meshes[i]->GetBdrFace(ib), parent_face_i, interface_attr, j, jb, test.meshes[j]->GetBdrFace(jb));
   //             }
   //          }
   //       }
   //    }
   // }

   // for (int k = 0; k < test.interfaceInfos.Size(); k++) {
   //      printf("(Mesh %d, BE %d) - Attr %d - (Mesh %d, BE %d)\n",
   //            interface_infos[k].Mesh1, interface_infos[k].BE1, interface_infos[k].Attr,
   //            interface_infos[k].Mesh2, interface_infos[k].BE2);
   //  }
}

double dbc2(const Vector &x)
{
   return 0.1 - 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

double dbc4(const Vector &x)
{
   return -0.1 + 0.1 * (x(1) - 1.0) * (x(1) - 1.0);
}

