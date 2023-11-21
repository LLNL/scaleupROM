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

#include "rom_nonlinearform.hpp"
#include "linalg_utils.hpp"

using namespace std;

namespace mfem
{

ROMNonlinearForm::ROMNonlinearForm(const int num_basis, FiniteElementSpace *f)
   : NonlinearForm(f)
{
   height = width = num_basis;
}

ROMNonlinearForm::~ROMNonlinearForm()
{
   delete Grad;

   for (int i = 0; i <  dnfi.Size(); i++)
   {
      delete dnfi[i];
      delete dnfi_sample[i];
   }

   for (int i = 0; i <  fnfi.Size(); i++)
   {
      delete fnfi[i];
      delete fnfi_sample[i];
   }

   for (int i = 0; i < bfnfi.Size(); i++)
   {
      delete bfnfi[i];
      delete bfnfi_sample[i];
   }
}

void ROMNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   assert(x.Size() == width);
   assert(y.Size() == height);

   assert(basis);
   assert(basis->NumCols() == height);

   // TODO(kevin): will need to consider how we should address this case.
   //              do we really need prolongation when we lift up from ROM?
   //              we might need a similar operation on the ROM basis DenseMatrix.
   // const Vector &px = Prolongate(x);
   // if (P) { aux2.SetSize(P->Height()); }

   // // If we are in parallel, ParNonLinearForm::Mult uses the aux2 vector. In
   // // serial, place the result directly in y (when there is no P).
   // Vector &py = P ? aux2 : y;

   // if (ext)
   // {
   //    mfem_error("ROMNonlinearForm::Mult - external operator is not implemented!\n");
   //    // ext->Mult(px, py);
   //    // if (Serial())
   //    // {
   //    //    if (cP) { cP->MultTranspose(py, y); }
   //    //    const int N = ess_tdof_list.Size();
   //    //    const auto tdof = ess_tdof_list.Read();
   //    //    auto Y = y.ReadWrite();
   //    //    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Y[tdof[i]] = 0.0; });
   //    // }
   //    // // In parallel, the result is in 'py' which is an alias for 'aux2'.
   //    // return;
   // }

   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   // TODO(kevin): not exactly sure what doftrans impacts.
   DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();

   // py = 0.0;
   y = 0.0;

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = dnfi_sample[k];
         assert(sample_info);

         int prev_el = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int el = sample->el;
            if (el != prev_el)
            {
               fe = fes->GetFE(el);
               doftrans = fes->GetElementVDofs(el, vdofs);
               T = fes->GetElementTransformation(el);
               MultSubMatrix(*basis, vdofs, x, el_x);
               if (doftrans) { doftrans->InvTransformPrimal(el_x); }

               prev_el = el;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            dnfi[k]->AssembleQuadratureVector(*fe, *T, ip, sample->qw, el_x, el_y);
            if (doftrans) { doftrans->TransformDual(el_y); }

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
         }  // for (int i = 0; i < el_samples->Size(); i++)
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   if (fnfi.Size())
   {
      FaceElementTransformations *tr = NULL;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int k = 0; k < fnfi.Size(); k++)
      {
         const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = fnfi_sample[k];
         assert(sample_info);

         int prev_face = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int face = sample->face;
            if (face != prev_face)
            {
               tr = mesh->GetInteriorFaceTransformations(face);
               if (tr == NULL)
               {
                  // for EQP, this indicates an ill sampling.
                  // for other hyper reductions, we can simply continue the loop.
                  mfem_error("InteriorFaceTransformation of the sampled face is NULL,\n"
                             "   indicating that an empty quadrature point is sampled.\n");
               }  // if (tr == NULL)

               fe1 = fes->GetFE(tr->Elem1No);
               fe2 = fes->GetFE(tr->Elem2No);

               fes->GetElementVDofs(tr->Elem1No, vdofs);
               fes->GetElementVDofs(tr->Elem2No, vdofs2);
               vdofs.Append (vdofs2);

               MultSubMatrix(*basis, vdofs, x, el_x);

               prev_face = face;
            }  // if (face != prev_face)

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            fnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->qw, el_x, el_y);

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < fnfi.Size(); k++)
   }  // if (fnfi.Size())

   if (bfnfi.Size())
   {
      FaceElementTransformations *tr = NULL;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }  // for (int k = 0; k < bfnfi.Size(); k++)

      for (int k = 0; k < bfnfi.Size(); k++)
      {
         const IntegrationRule *ir = bfnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = bfnfi_sample[k];
         assert(sample_info);

         int prev_be = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int be = sample->be;
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0)
            { 
               // for EQP, this indicates an ill sampling.
               // for other hyper reductions, we can simply continue the loop.
               mfem_error("The sampled boundary element does not have a boundary condition,\n"
                           "   indicating that an empty quadrature point is sampled.\n");
               // continue;
            }

            if (be != prev_be)
            {
               tr = mesh->GetBdrFaceTransformations (be);
               if (tr == NULL)
               {
                  // for EQP, this indicates an ill sampling.
                  // for other hyper reductions, we can simply continue the loop.
                  mfem_error("BdrFaceTransformation of the sampled face is NULL,\n"
                             "   indicating that an empty quadrature point is sampled.\n");
               }

               fes->GetElementVDofs(tr->Elem1No, vdofs);

               fe1 = fes->GetFE(tr->Elem1No);
               // The fe2 object is really a dummy and not used on the boundaries,
               // but we can't dereference a NULL pointer, and we don't want to
               // actually make a fake element.
               fe2 = fe1;

               MultSubMatrix(*basis, vdofs, x, el_x);

               prev_be = be;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            bfnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->qw, el_x, el_y);

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < bfnfi.Size(); k++)
   }  // if (bfnfi.Size())

   // TODO(kevin): will need to consider how we should address this case.
   //              do we really need prolongation when we lift up from ROM?
   //              we might need a similar operation on the ROM basis DenseMatrix.
   // if (Serial())
   // {
   //    if (cP) { cP->MultTranspose(py, y); }

   //    for (int i = 0; i < ess_tdof_list.Size(); i++)
   //    {
   //       y(ess_tdof_list[i]) = 0.0;
   //    }
   //    // y(ess_tdof_list[i]) = x(ess_tdof_list[i]);
   // }
   // // In parallel, the result is in 'py' which is an alias for 'aux2'.
}

Operator& ROMNonlinearForm::GetGradient(const Vector &x) const
{
   assert(x.Size() == width);
   // if (ext)
   // {
   //    hGrad.Clear();
   //    Operator &grad = ext->GetGradient(Prolongate(x));
   //    Operator *Gop;
   //    grad.FormSystemOperator(ess_tdof_list, Gop);
   //    hGrad.Reset(Gop);
   //    // In both serial and parallel, when using extension, we return the final
   //    // global true-dof gradient with imposed b.c.
   //    return *hGrad;
   // }

   const int skip_zeros = 0;
   Array<int> vdofs, rom_vdofs;
   Vector el_x;
   DenseMatrix elmat, quadmat;
   const FiniteElement *fe;
   ElementTransformation *T;
   DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();

   // rom vdofs simply cover the number of entire basis.
   rom_vdofs.SetSize(width);
   for (int b = 0; b < width; b++) rom_vdofs[b] = b;

   // TODO(kevin): will need to consider how we should address this case.
   //              do we really need prolongation when we lift up from ROM?
   //              we might need a similar operation on the ROM basis DenseMatrix.
   // const Vector &px = Prolongate(x);

   if (Grad == NULL)
   {
      // Grad = new SparseMatrix(fes->GetVSize());
      // Grad = new SparseMatrix(width);
      Grad = new DenseMatrix(height, width);
   }
   else
   {
      *Grad = 0.0;
   }

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = dnfi_sample[k];
         assert(sample_info);

         int prev_el = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int el = sample->el;
            if (el != prev_el)
            {
               fe = fes->GetFE(el);
               doftrans = fes->GetElementVDofs(el, vdofs);
               T = fes->GetElementTransformation(el);
               MultSubMatrix(*basis, vdofs, x, el_x);
               if (doftrans) { doftrans->InvTransformPrimal(el_x); }

               prev_el = el;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            dnfi[k]->AssembleQuadratureGrad(*fe, *T, ip, sample->qw, el_x, elmat);
            if (doftrans) { doftrans->TransformDual(elmat); }

            AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
            // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
         }  // for (int i = 0; i < el_samples->Size(); i++)
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   if (fnfi.Size())
   {
      FaceElementTransformations *tr = NULL;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int k = 0; k < fnfi.Size(); k++)
      {
         const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = fnfi_sample[k];
         assert(sample_info);

         int prev_face = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int face = sample->face;
            if (face != prev_face)
            {
               tr = mesh->GetInteriorFaceTransformations(face);
               if (tr == NULL)
               {
                  // for EQP, this indicates an ill sampling.
                  // for other hyper reductions, we can simply continue the loop.
                  mfem_error("InteriorFaceTransformation of the sampled face is NULL,\n"
                             "   indicating that an empty quadrature point is sampled.\n");
               }  // if (tr == NULL)

               fe1 = fes->GetFE(tr->Elem1No);
               fe2 = fes->GetFE(tr->Elem2No);

               fes->GetElementVDofs(tr->Elem1No, vdofs);
               fes->GetElementVDofs(tr->Elem2No, vdofs2);
               vdofs.Append (vdofs2);

               MultSubMatrix(*basis, vdofs, x, el_x);

               prev_face = face;
            }  // if (face != prev_face)

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            fnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->qw, el_x, elmat);
            AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
            // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < fnfi.Size(); k++)
   }  // if (fnfi.Size())

   if (bfnfi.Size())
   {
      FaceElementTransformations *tr = NULL;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }  // for (int k = 0; k < bfnfi.Size(); k++)

      for (int k = 0; k < bfnfi.Size(); k++)
      {
         const IntegrationRule *ir = bfnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = bfnfi_sample[k];
         assert(sample_info);

         int prev_be = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int be = sample->be;
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0)
            { 
               // for EQP, this indicates an ill sampling.
               // for other hyper reductions, we can simply continue the loop.
               mfem_error("The sampled boundary element does not have a boundary condition,\n"
                           "   indicating that an empty quadrature point is sampled.\n");
               // continue;
            }

            if (be != prev_be)
            {
               tr = mesh->GetBdrFaceTransformations (be);
               if (tr == NULL)
               {
                  // for EQP, this indicates an ill sampling.
                  // for other hyper reductions, we can simply continue the loop.
                  mfem_error("BdrFaceTransformation of the sampled face is NULL,\n"
                             "   indicating that an empty quadrature point is sampled.\n");
               }

               fes->GetElementVDofs(tr->Elem1No, vdofs);

               fe1 = fes->GetFE(tr->Elem1No);
               // The fe2 object is really a dummy and not used on the boundaries,
               // but we can't dereference a NULL pointer, and we don't want to
               // actually make a fake element.
               fe2 = fe1;

               MultSubMatrix(*basis, vdofs, x, el_x);

               prev_be = be;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            bfnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->qw, el_x, elmat);
            AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
            // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < bfnfi.Size(); k++)
   }  // if (bfnfi.Size())

   // if (!Grad->Finalized())
   // {
   //    Grad->Finalize(skip_zeros);
   // }

   DenseMatrix *mGrad = Grad;
   
   // TODO(kevin): will need to consider how we should address this case.
   //              do we really need prolongation when we lift up from ROM?
   //              we might need a similar operation on the ROM basis DenseMatrix.
   // if (Serial())
   // {
   //    if (cP)
   //    {
   //       delete cGrad;
   //       cGrad = RAP(*cP, *Grad, *cP);
   //       mGrad = cGrad;
   //    }
   //    for (int i = 0; i < ess_tdof_list.Size(); i++)
   //    {
   //       mGrad->EliminateRowCol(ess_tdof_list[i]);
   //    }
   // }

   return *mGrad;
}

void ROMNonlinearForm::PrecomputeCoefficients()
{
   assert(basis);
   assert(basis->NumCols() == height);

   Mesh *mesh = fes->GetMesh();

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         Array<SampleInfo> *sample_info = dnfi_sample[k];
         assert(sample_info);

         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
            dnfi[k]->AppendPrecomputeCoefficients(fes, *basis, *sample);
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   if (fnfi.Size())
   {
      for (int k = 0; k < fnfi.Size(); k++)
      {
         Array<SampleInfo> *sample_info = fnfi_sample[k];
         assert(sample_info);

         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
            fnfi[k]->AppendPrecomputeCoefficients(fes, *basis, *sample);
      }  // for (int k = 0; k < fnfi.Size(); k++)
   }  // if (fnfi.Size())

   if (bfnfi.Size())
   {
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfnfi.Size(); k++)
      {
         if (bfnfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }  // for (int k = 0; k < bfnfi.Size(); k++)

      for (int k = 0; k < bfnfi.Size(); k++)
      {
         Array<SampleInfo> *sample_info = bfnfi_sample[k];
         assert(sample_info);

         int prev_be = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int be = sample->be;
            const int bdr_attr = mesh->GetBdrAttribute(i);
            if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0)
            { 
               // for EQP, this indicates an ill sampling.
               // for other hyper reductions, we can simply continue the loop.
               mfem_error("The sampled boundary element does not have a boundary condition,\n"
                           "   indicating that an empty quadrature point is sampled.\n");
               // continue;
            }

            bfnfi[k]->AppendPrecomputeCoefficients(fes, *basis, *sample);
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < bfnfi.Size(); k++)
   }  // if (bfnfi.Size())
}

}
