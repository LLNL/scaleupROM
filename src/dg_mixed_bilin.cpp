// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "dg_mixed_bilin.hpp"

using namespace std;

namespace mfem
{

/*
   MixedBilinearFormDGExtension
*/

MixedBilinearFormDGExtension::MixedBilinearFormDGExtension(FiniteElementSpace *tr_fes,
                                                            FiniteElementSpace *te_fes,
                                                            MixedBilinearFormDGExtension *mbf)
   : MixedBilinearForm(tr_fes, te_fes, mbf)
{
   interior_face_integs = mbf->interior_face_integs;
   boundary_face_integs = mbf->boundary_face_integs;
   boundary_face_integs_marker = mbf->boundary_face_integs_marker;
};

void MixedBilinearFormDGExtension::AddInteriorFaceIntegrator(MixedBilinearFormFaceIntegrator * bfi)
{
   interior_face_integs.Append(bfi);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi)
{
   boundary_face_integs.Append(bfi);
   // NULL marker means apply everywhere
   boundary_face_integs_marker.Append(NULL);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi,
                                                         Array<int> &bdr_marker)
{
   boundary_face_integs.Append(bfi);
   boundary_face_integs_marker.Append(&bdr_marker);
}

MixedBilinearFormDGExtension::~MixedBilinearFormDGExtension()
{
   if (!extern_bfs)
   {
      int i;
      for (i = 0; i < interior_face_integs.Size(); i++) { delete interior_face_integs[i]; }
      for (i = 0; i < boundary_face_integs.Size(); i++)
      { delete boundary_face_integs[i]; }
   }
}

void MixedBilinearFormDGExtension::Assemble(int skip_zeros)
{
   MixedBilinearForm::Assemble(skip_zeros);
   if (ext)
      return;

   Mesh *mesh = test_fes -> GetMesh();

   assert(mat != NULL);

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Array<int> trial_vdofs2, test_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         // ftr = mesh->GetFaceElementTransformations(i);
         // trial_fes->GetFaceVDofs(i, trial_vdofs);
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            trial_fes->GetElementVDofs(tr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(tr->Elem1No, test_vdofs);
            
            trial_fes->GetElementVDofs(tr->Elem2No, trial_vdofs2);
            test_fes->GetElementVDofs(tr->Elem2No, test_vdofs2);
            trial_vdofs.Append(trial_vdofs2);
            test_vdofs.Append(test_vdofs2);

            trial_fe1 = trial_fes->GetFE(tr->Elem1No);
            test_fe1 = test_fes->GetFE(tr->Elem1No);
            trial_fe2 = trial_fes->GetFE(tr->Elem2No);
            test_fe2 = test_fes->GetFE(tr->Elem2No);

            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *trial_fe2, *test_fe1,
                                                            *test_fe2, *tr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Array<int> trial_vdofs2, test_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary trace face"
                     "integrator #" << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < trial_fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr)
         {
            trial_fes->GetElementVDofs(tr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(tr->Elem1No, test_vdofs);
            trial_fe1 = trial_fes->GetFE(tr->Elem1No);
            test_fe1 = test_fes->GetFE(tr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            trial_fe2 = trial_fe1;
            test_fe2 = test_fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0)
               { continue; }

               boundary_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *trial_fe2,
                                                            *test_fe1, *test_fe2,
                                                            *tr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }
}

/*
   DGNormalFluxIntegrator
*/

void DGNormalFluxIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                                const FiniteElement &trial_fe2,
                                                const FiniteElement &test_fe1,
                                                const FiniteElement &test_fe2,
                                                FaceElementTransformations &Trans,
                                                DenseMatrix &elmat)
{
   dim = trial_fe1.GetDim();
   trial_dof1 = trial_fe1.GetDof();
   trial_vdof1 = dim * trial_dof1;
   test_dof1 = test_fe1.GetDof();

   nor.SetSize(dim);
   wnor.SetSize(dim);

   // vshape1.SetSize(trial_dof1, dim);
   // vshape1_n.SetSize(trial_dof1);
   trshape1.SetSize(trial_dof1);
   shape1.SetSize(test_dof1);

   if (Trans.Elem2No >= 0)
   {
      trial_dof2 = trial_fe2.GetDof();
      trial_vdof2 = dim * trial_dof2;
      test_dof2 = test_fe2.GetDof();

      // vshape2.SetSize(trial_dof2, dim);
      // vshape2_n.SetSize(trial_dof2);
      trshape2.SetSize(trial_dof2);
      shape2.SetSize(test_dof2);
   }
   else
   {
      trial_dof2 = 0;
      test_dof2 = 0;
   }

   elmat.SetSize((test_dof1 + test_dof2), (trial_vdof1 + trial_vdof2));
   elmat = 0.0;

   // TODO: need to revisit this part for proper convergence rate.
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }  // if (ir == NULL)

   for (p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      // trial_fe1.CalcVShape(eip1, vshape1);
      trial_fe1.CalcShape(eip1, trshape1);
      test_fe1.CalcShape(eip1, shape1);
      // vshape1.Mult(nor, vshape1_n);

      w = ip.weight;
      if (trial_dof2)
         w *= 0.5;

      wnor.Set(w, nor);
      
      for (jm = 0, j = 0; jm < dim; jm++)
      {
         wn = wnor(jm);
         for (jdof = 0; jdof < trial_dof1; jdof++, j++)
            for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
               elmat(i, j) += wn * shape1(idof) * trshape1(jdof);
      }

      if (trial_dof2)
      {
         // trial_fe2.CalcVShape(eip2, vshape2);
         trial_fe2.CalcShape(eip2, trshape2);
         test_fe2.CalcShape(eip2, shape2);
         // vshape2.Mult(nor, vshape2_n);

         for (jm = 0, j = 0; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof1; jdof++, j++)
               for (idof = 0, i = test_dof1; idof < test_dof2; idof++, i++)
                  elmat(i, j) += wn * shape2(idof) * trshape1(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof1; j++)
         //    {
         //       elmat(test_dof1+i, j) += w * shape2(i) * vshape1_n(j);
         //    }

         for (jm = 0, j = trial_vdof1; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = test_dof1; idof < test_dof2; idof++, i++)
                  elmat(i, j) -= wn * shape2(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(test_dof1+i, trial_dof1+j) -= w * shape2(i) * vshape2_n(j);
         //    }

         for (jm = 0, j = trial_vdof1; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
                  elmat(i, j) -= wn * shape1(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof1; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(i, trial_dof1+j) -= w * shape1(i) * vshape2_n(j);
         //    }
      }  // if (trial_dof2)
   }  // for (p = 0; p < ir->GetNPoints(); p++)
}

}
