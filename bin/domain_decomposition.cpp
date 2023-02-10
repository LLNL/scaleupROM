#include <stdio.h>
#include "mfem.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char* argv[])
{
  // 1. Initialize MPI and HYPRE.
  Mpi::Init(argc, argv);
  Hypre::Init();

  // 2. Parse command line options.
  const char *mesh_file = "./rectangle.mesh";
  int order = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
  args.ParseCheck();

  // 3. Read the serial mesh from the given mesh file.
  Mesh serial_mesh(mesh_file);

  // 5. Define a finite element space on the mesh. Here we use H1 continuous
  //    high-order Lagrange finite elements of the given order.
  H1_FECollection fec(order, serial_mesh.Dimension());
  FiniteElementSpace serial_fespace(&serial_mesh, &fec);
  int total_num_dofs = serial_fespace.GetTrueVSize();
  if (Mpi::Root())
  {
     cout << "Number of unknowns: " << total_num_dofs << endl;
  }

  {
    cout << "Global mesh information." << endl;
    int numElem = serial_mesh.GetNE();
    printf("number of elements in mesh: %d\n", numElem);
    for (int el = 0; el < numElem; el++) {
      int attr = serial_mesh.GetAttribute(el);
      Vector center(2);
      serial_mesh.GetElementCenter(el, center);
      printf("Element %d: %d - (%f, %f)\n", el, attr, center(0), center(1));
    }

    int numBE = serial_mesh.GetNBE();
    printf("number of boundary elements in mesh: %d\n", numBE);
    int numBdr = serial_mesh.bdr_attributes.Size();
    printf("number of boundary attributes in mesh: %d\n", numBdr);
    for (int b = 0; b < numBdr; b++) {
      printf("boundary attribute %d: %d\n", b, serial_mesh.bdr_attributes[b]);
    }

    int numF = serial_mesh.GetNumFaces();
    printf("number of faces in mesh: %d\n", numF);

    GridFunction serial_x(&serial_fespace);
    serial_x = 0.0;
    printf("number of points in GridFunction: %d\n", serial_x.Size());

    for (int b = 0; b < numBE; b++) {
      int bdrAttr = serial_mesh.GetBdrElement(b)->GetAttribute();
      Array<int> vdofs;
      DofTransformation *beVdof = serial_fespace.GetBdrElementVDofs(b, vdofs);
      for (int v = 0; v < vdofs.Size(); v++) {
        printf("Bdr element %d, attribute %d, %d-th vdof index: %d\n", b, bdrAttr, v, vdofs[v]);
      }
    }

    // 6. Define the BlockStructure of the problem, i.e. define the array of
    //    offsets for each variable. The last component of the Array is the sum
    //    of the dimensions of each block.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = serial_fespace.GetVSize();
    block_offsets[2] = serial_fespace.GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(D1) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(D2) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(D1+D2) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";
  }

  // // Create submesh.
  // Array<int> domainAttr(1);
  // domainAttr[0] = 1;
  // auto submesh = SubMesh::CreateFromDomain(serial_mesh, domainAttr);
  // FiniteElementSpace fes_submesh(&submesh, &fec);
  // {
  //   cout << "Submesh information." << endl;
  //   int numElem = submesh.GetNE();
  //   printf("number of elements in submesh: %d\n", numElem);
  //   for (int el = 0; el < numElem; el++) {
  //     int attr = submesh.GetAttribute(el);
  //     Vector center(2);
  //     submesh.GetElementCenter(el, center);
  //     printf("Element %d: %d - (%f, %f)\n", el, attr, center(0), center(1));
  //   }
  //
  //   int numBE = submesh.GetNBE();
  //   printf("number of boundary elements in submesh: %d\n", numBE);
  //   int numBdr = submesh.bdr_attributes.Size();
  //   printf("number of boundary attributes in submesh: %d\n", numBdr);
  //   for (int b = 0; b < numBdr; b++) {
  //     printf("boundary attribute %d: %d\n", b, submesh.bdr_attributes[b]);
  //   }
  //
  //   int numF = submesh.GetNumFaces();
  //   printf("number of faces in submesh: %d\n", numF);
  // }

  return 0;
}
