// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include "mfem.hpp"
#include "interfaceinteg.hpp"

using namespace std;
using namespace mfem;

void AssembleInterfaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
   DenseMatrix &elmat);

int main(int argc, char* argv[])
{
  // 1. Initialize MPI and HYPRE.
  Mpi::Init(argc, argv);
  Hypre::Init();

  const Element::Type el_type = Element::QUADRILATERAL;
  int n_el = 1;
  double a_ = 1.0;
  int order = 1;
  int dim = 1;

  // 2. Parse command line options.
  OptionsParser args(argc, argv);
  args.AddOption(&n_el, "-n", "--num-elem", "Number of element in one direction.");
  args.AddOption(&a_, "-a", "--length", "Length of the domain.");
  args.AddOption(&order, "-o", "--order", "Order of finite element space.");
  args.AddOption(&dim, "-d", "--dim", "Dimension of the mesh.");
  args.ParseCheck();

  // 3. Read the serial mesh from the given mesh file.
  int numSub = 2;
  Mesh unit_mesh;
  switch (dim) {
    case 1:

    unit_mesh = Mesh::MakeCartesian1D(n_el, a_);
    break;

    case 2:

    unit_mesh = Mesh::MakeCartesian2D(n_el, n_el, el_type, false, a_, a_, false);
    break;

    default:

    printf("Dimension %d is not supported!\n", dim);
    exit(1);
  }

  std::string mesh_file = "dode.mesh";
  ofstream ofs(mesh_file.c_str());
  ofs.precision(8);
  unit_mesh.Print(ofs);
  ofs.close();

  Array<Mesh *> meshes(numSub);
  for (int n = 0; n < numSub; n++) {
    meshes[n] = new Mesh(unit_mesh);
  }

  // 5. Define a finite element space on the mesh. Here we use H1 continuous
  //    high-order Lagrange finite elements of the given order.
  H1_FECollection fec(order, unit_mesh.Dimension());
  Array<FiniteElementSpace *> fespaces(numSub);
  for (int m = 0; m < numSub; m++) {
    fespaces[m] = new FiniteElementSpace(meshes[m], &fec);
  }
  int total_num_dofs = fespaces[0]->GetTrueVSize();
  cout << "Number of unknowns: " << total_num_dofs << endl;

  // Turn of boundary handling for now.
  Array<int> ess_attr(2);
  ess_attr = 0;
  Array<int> ess_tdof_list;
  for (int m = 0; m < numSub; m++) {
    fespaces[m]->GetEssentialTrueDofs(ess_attr, ess_tdof_list);
  }

  Array<BilinearForm *> as(numSub);
  for (int m = 0; m < numSub; m++) {
    as[m] = new BilinearForm(fespaces[m]);
  }

  double sigma = -1.0;
  double kappa = 1.5;
  ConstantCoefficient one(1.0), minusOne(-1.0);
  Array<int> bdr_marker(meshes[0]->bdr_attributes.Max());
  bdr_marker = 0;
  bdr_marker[1] = 1;
  as[0]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa), bdr_marker);
  // as[0]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));

  //NOTE: This only produces zero matrix.
  // as[0]->AddBoundaryIntegrator(new MixedScalarDerivativeIntegrator(minusOne));
  //NOTE: This is non-zero, but taking volume integral in each element.
  // as[0]->AddDomainIntegrator(new MixedScalarDerivativeIntegrator(minusOne));

  //NOTE: DG integrator are implemented only face integrators.
  // as[0]->AddDomainIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
  //NOTE: AddBoundaryIntegrator is for continuous FEM and DG integrators do not have them (or unimpelmented.)
  // as[0]->AddBoundaryIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
  //NOTE: Likewise, continuous integrators do not implement AddBdrFaceIntegrator. e.g.,
  // as[0]->AddBdrFaceIntegrator(new MixedScalarDerivativeIntegrator(minusOne));

  as[0]->Assemble();

  SparseMatrix A;
  as[0]->FormSystemMatrix(ess_tdof_list, A);

  DenseMatrix Ad;
  A.ToDenseMatrix(Ad);
  printf("Using MFEM DGDiffusionIntegrator.\n");
  printf("A (%d x %d)\n", Ad.Width(), Ad.Height());

  for (int h = 0; h < Ad.Height(); h++) {
    for (int w = 0; w < Ad.Width(); w++) {
      printf("%2.3f\t", Ad(h, w));
    }
    printf("\n");
  }

  // Building FaceInfo
  bool print_face_info = true;
  if (print_face_info) {
    for (int m = 0; m < 1; m++){
      printf("Mesh %d\n", m);
      for (int i = 0; i < meshes[m]->GetNBE(); i++) {
        int fn = meshes[m]->GetBdrFace(i);
        printf("Boundary Element %d : Face %d\n", i, fn);
      }
      // for (int i = 0; i < meshes[m]->GetNBE(); i++) {
      //   // FaceInfo face_info;
      //   // int Elem1No, Elem2No, Elem1Inf, Elem2Inf;
      //   // int NCFace; // -1 if this is a regular conforming/boundary face;
      //   // face_info.NCFace = -1;
      //
      //   int elem_id, face_info;
      //   meshes[m]->GetBdrElementAdjacentElement(i, elem_id, face_info);
      //   printf("Boundary element %d - Adjacent element %d\n", i, elem_id);
      //   printf("Face index: %d, face orientation : %d\n", face_info / 64, face_info % 64);
      //
      //   int fn = meshes[m]->GetBdrFace(i);
      //   int face_inf[2];
      //   meshes[m]->GetFaceInfos(fn, &face_inf[0], &face_inf[1]);
      //   printf("From faces_info - Face index: %d, face orientation : %d\n", face_inf[0] / 64, face_inf[0] % 64);
      //
      // }
      //
      // int nfaces = meshes[m]->GetNumFaces();
      // for (int i = 0; i < nfaces; i++)
      // {
      //    int face_inf[2], face_idx[2];
      //    meshes[m]->GetFaceInfos(i, &face_inf[0], &face_inf[1]);
      //    meshes[m]->GetFaceElements(i, &face_idx[0], &face_idx[1]);
      //
      //    Mesh::FaceInformation face_info = meshes[m]->GetFaceInformation(i);
      //    for (int j = 0; j < 2; j++) {
      //      printf("Face %d Element %d information\n", i, j);
      //      printf("Index: %d =? %d\n", face_info.element[j].index, face_idx[j]);
      //      printf("Local Face ID: %d =? %d\n", face_info.element[j].local_face_id, face_inf[j] / 64);
      //      printf("Orientation: %d =? %d\n", face_info.element[j].orientation, face_inf[j] % 64);
      //    }
      // }
    }
  }

  // From BilinearForm boundary face integrator assemble part.
  {
      // using InterfaceDGDiffusionIntegrator.
      printf("Using InterfaceDGDiffusionIntegrator.\n");
      InterfaceDGDiffusionIntegrator interface_integ(one, sigma, kappa);

      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs, vdofs2;
      tr1 = meshes[0]->GetBdrFaceTransformations(1);
      tr2 = meshes[1]->GetBdrFaceTransformations(0);
      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fespaces[0]->GetElementVDofs(tr1->Elem1No, vdofs);
         // Both domains will have the adjacent element as Elem1.
         fe1 = fespaces[0]->GetFE(tr1->Elem1No);
         fe2 = fespaces[1]->GetFE(tr2->Elem1No);

         interface_integ.AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);
         printf("Interface elemmat.\n");
         for (int I = 0; I < 2; I++)
          {
            for (int J = 0; J < 2; J++)
            {
              printf("elemmat(%d,%d)\n", I, J);
              DenseMatrix *elemmat = elemmats(I,J);
              for (int i = 0; i < elemmat->NumRows(); i++) {
                for (int j = 0; j < elemmat->NumCols(); j++) {
                  printf("%.3f\t", (*elemmat)(i,j));
                }
                printf("\n");
              }
            } // for (int J = 0; J < 2; J++)
          } // for (int I = 0; I < 2; I++)
         // for (int k = 0; k < boundary_face_integs.Size(); k++)
         // {
         //    if (boundary_face_integs_marker[k] &&
         //        (*boundary_face_integs_marker[k])[bdr_attr-1] == 0)
         //    { continue; }
         //
         //    boundary_face_integs[k] -> AssembleFaceMatrix (*fe1, *fe2, *tr,
         //                                                   elemmat);
         //    mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
         // }
      }
  }

  return 0;
}

void AssembleInterfaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans1, FaceElementTransformations &Trans2,
   DenseMatrix &elmat)
{
  Coefficient *Q = new ConstantCoefficient(1.0);
  MatrixCoefficient *MQ = NULL;
  double sigma = -1.0;
  double kappa = 1.5;
  const IntegrationRule *IntRule = NULL;
  bool boundary = false;

  // these are not thread-safe!
  Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
  DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

  Vector nor2;

  {
     int dim, ndof1, ndof2, ndofs;
     bool kappa_is_nonzero = (kappa != 0.);
     double w, wq = 0.0;

     dim = el1.GetDim();

     nor.SetSize(dim);
     nor2.SetSize(dim);
     nh.SetSize(dim);
     ni.SetSize(dim);
     adjJ.SetSize(dim);
     if (MQ)
     {
        mq.SetSize(dim);
     }

     ndof1 = el1.GetDof();
     shape1.SetSize(ndof1);
     dshape1.SetSize(ndof1, dim);
     dshape1dn.SetSize(ndof1);

     // TODO: different boolean handling is needed (Elem2No will be zero for both Trans1 and Trans2).
     if (boundary) {
     // if (Trans.Elem2No >= 0)
       ndof2 = 0;
     } else {
       ndof2 = el2.GetDof();
       shape2.SetSize(ndof2);
       dshape2.SetSize(ndof2, dim);
       dshape2dn.SetSize(ndof2);
     }

     ndofs = ndof1 + ndof2;
     elmat.SetSize(ndofs);
     elmat = 0.0;
     if (kappa_is_nonzero)
     {
        jmat.SetSize(ndofs);
        jmat = 0.;
     }

     const IntegrationRule *ir = IntRule;
     if (ir == NULL)
     {
        // a simple choice for the integration order; is this OK?
        int order;
        if (ndof2)
        {
           order = 2*max(el1.GetOrder(), el2.GetOrder());
        }
        else
        {
           order = 2*el1.GetOrder();
        }
        ir = &IntRules.Get(Trans1.GetGeometryType(), order);

        assert(Trans1.GetGeometryType() == Trans2.GetGeometryType());
     }

     // assemble: < {(Q \nabla u).n},[v] >      --> elmat
     //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
     for (int p = 0; p < ir->GetNPoints(); p++)
     {
        const IntegrationPoint &ip = ir->IntPoint(p);

        // Set the integration point in the face and the neighboring elements
        Trans1.SetAllIntPoints(&ip);
        Trans2.SetAllIntPoints(&ip);

        // Access the neighboring elements' integration points
        // Note: eip1 and eip2 come from Element1 of Trans1 and Trans2 respectively.
        const IntegrationPoint &eip1 = Trans1.GetElement1IntPoint();
        const IntegrationPoint &eip2 = Trans2.GetElement1IntPoint();

        // computing outward normal vectors.
        if (dim == 1)
        {
           nor(0) = 2*eip1.x - 1.0;
           nor2(0) = 2*eip2.x - 1.0;
           printf("nor1: %f, nor2: %f\n", nor(0), nor2(0));
        }
        else
        {
           CalcOrtho(Trans1.Jacobian(), nor);
           CalcOrtho(Trans2.Jacobian(), nor2);
           printf("nor1: ");
           for (int d = 0; d < dim; d++) {
             printf("%f\t", nor(d));
           }
           printf("\n");

           printf("nor2: ");
           for (int d = 0; d < dim; d++) {
             printf("%f\t", nor2(d));
           }
           printf("\n");
        }

        el1.CalcShape(eip1, shape1);
        el1.CalcDShape(eip1, dshape1);
        w = ip.weight/Trans1.Elem1->Weight();
        if (ndof2)
        {
           w /= 2;
        }
        if (!MQ)
        {
           if (Q)
           {
              w *= Q->Eval(*Trans1.Elem1, eip1);
           }
           ni.Set(w, nor);
        }
        else
        {
           nh.Set(w, nor);
           MQ->Eval(mq, *Trans1.Elem1, eip1);
           mq.MultTranspose(nh, ni);
        }
        CalcAdjugate(Trans1.Elem1->Jacobian(), adjJ);
        adjJ.Mult(ni, nh);
        if (kappa_is_nonzero)
        {
           wq = ni * nor;
        }
        // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
        // independent of Loc1 and always gives the size of element 1 in
        // direction perpendicular to the face. Indeed, for linear transformation
        //     |nor|=measure(face)/measure(ref. face),
        //   det(J1)=measure(element)/measure(ref. element),
        // and the ratios measure(ref. element)/measure(ref. face) are
        // compatible for all element/face pairs.
        // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
        // for any tetrahedron vol(tet)=(1/3)*height*area(base).
        // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

        dshape1.Mult(nh, dshape1dn);
        for (int i = 0; i < ndof1; i++)
           for (int j = 0; j < ndof1; j++)
           {
              elmat(i, j) += shape1(i) * dshape1dn(j);
           }

        if (ndof2)
        {
           el2.CalcShape(eip2, shape2);
           el2.CalcDShape(eip2, dshape2);
           // Trans2 is also boundary face and only has Elem1.
           w = ip.weight/2/Trans2.Elem1->Weight();
           if (!MQ)
           {
              if (Q)
              {
                 w *= Q->Eval(*Trans2.Elem1, eip2);
              }
              ni.Set(w, nor);
           }
           else
           {
              nh.Set(w, nor);
              MQ->Eval(mq, *Trans2.Elem1, eip2);
              mq.MultTranspose(nh, ni);
           }
           CalcAdjugate(Trans2.Elem1->Jacobian(), adjJ);
           adjJ.Mult(ni, nh);
           if (kappa_is_nonzero)
           {
              wq += ni * nor;
           }

           dshape2.Mult(nh, dshape2dn);

           for (int i = 0; i < ndof1; i++)
              for (int j = 0; j < ndof2; j++)
              {
                 elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
              }

           for (int i = 0; i < ndof2; i++)
              for (int j = 0; j < ndof1; j++)
              {
                 elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
              }

           for (int i = 0; i < ndof2; i++)
              for (int j = 0; j < ndof2; j++)
              {
                 elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
              }
        }

        if (kappa_is_nonzero)
        {
           // only assemble the lower triangular part of jmat
           wq *= kappa;
           for (int i = 0; i < ndof1; i++)
           {
              const double wsi = wq*shape1(i);
              for (int j = 0; j <= i; j++)
              {
                 jmat(i, j) += wsi * shape1(j);
              }
           }
           if (ndof2)
           {
              for (int i = 0; i < ndof2; i++)
              {
                 const int i2 = ndof1 + i;
                 const double wsi = wq*shape2(i);
                 for (int j = 0; j < ndof1; j++)
                 {
                    jmat(i2, j) -= wsi * shape1(j);
                 }
                 for (int j = 0; j <= i; j++)
                 {
                    jmat(i2, ndof1 + j) += wsi * shape2(j);
                 }
              }
           }
        }
     }

     // elmat := -elmat + sigma*elmat^t + jmat
     if (kappa_is_nonzero)
     {
        for (int i = 0; i < ndofs; i++)
        {
           for (int j = 0; j < i; j++)
           {
              double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
              elmat(i,j) = sigma*aji - aij + mij;
              elmat(j,i) = sigma*aij - aji + mij;
           }
           elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
        }
     }
     else
     {
        for (int i = 0; i < ndofs; i++)
        {
           for (int j = 0; j < i; j++)
           {
              double aij = elmat(i,j), aji = elmat(j,i);
              elmat(i,j) = sigma*aji - aij;
              elmat(j,i) = sigma*aij - aji;
           }
           elmat(i,i) *= (sigma - 1.);
        }
     }
  }
}
