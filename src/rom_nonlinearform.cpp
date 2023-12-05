// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "rom_nonlinearform.hpp"
#include "linalg_utils.hpp"

using namespace std;

namespace mfem
{

ROMNonlinearForm::ROMNonlinearForm(const int num_basis, FiniteElementSpace *f)
   : NonlinearForm(f)
{
   height = width = num_basis;

   for (int k = 0; k < Nt; k++) jac_timers[k] = new StopWatch;
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

   printf("%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
   "init", "sample-load", "elem-load", "assem-grad", "others", "sum", "total");
   double sum = 0.0;
   for (int k = 0; k < Nt-1; k++)
   {
      printf("%.3E\t", jac_timers[k]->RealTime());
      sum += jac_timers[k]->RealTime();
   }
   printf("%.3E\t", sum);
   printf("%.3E\n", jac_timers[Nt-1]->RealTime());

   for (int k = 0; k < Nt; k++) delete jac_timers[k];
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
            T = fes->GetElementTransformation(el);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (dnfi[k]->precomputable))
               dnfi[k]->AddAssembleVector_Fast(i, sample->qw, *T, ip, x, y);
            else
            {
               if (el != prev_el)
               {
                  fe = fes->GetFE(el);
                  doftrans = fes->GetElementVDofs(el, vdofs);
                  MultSubMatrix(*basis, vdofs, x, el_x);
                  if (doftrans) { doftrans->InvTransformPrimal(el_x); }

                  prev_el = el;
               }

               dnfi[k]->AssembleQuadratureVector(*fe, *T, ip, sample->qw, el_x, el_y);
               if (doftrans) { doftrans->TransformDual(el_y); }

               AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
            }  // not (precompute && (dnfi[k]->precomputable))
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
            tr = mesh->GetInteriorFaceTransformations(face);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (fnfi[k]->precomputable))
               fnfi[k]->AddAssembleVector_Fast(i, sample->qw, *tr, ip, x, y);
            else
            {
               if (face != prev_face)
               {
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

               fnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->qw, el_x, el_y);

               AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
            }  // not (precompute && (dnfi[k]->precomputable))
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

            tr = mesh->GetBdrFaceTransformations (be);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (bfnfi[k]->precomputable))
               bfnfi[k]->AddAssembleVector_Fast(i, sample->qw, *tr, ip, x, y);
            else
            {
               if (be != prev_be)
               {
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

               bfnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->qw, el_x, el_y);

               AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);
            }  // not (precompute && (dnfi[k]->precomputable))
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
   jac_timers[5]->Start();

   jac_timers[0]->Start();
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
   jac_timers[0]->Stop();

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         jac_timers[1]->Start();
         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         Array<SampleInfo> *sample_info = dnfi_sample[k];
         assert(sample_info);

         int prev_el = -1;
         SampleInfo *sample = sample_info->GetData();
         jac_timers[1]->Stop();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            jac_timers[2]->Start();
            int el = sample->el;
            T = fes->GetElementTransformation(el);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);
            jac_timers[2]->Stop();

            jac_timers[3]->Start();
            if (precompute && (dnfi[k]->precomputable))
               dnfi[k]->AddAssembleGrad_Fast(i, sample->qw, *T, ip, x, *Grad);
            else
            {
               if (el != prev_el)
               {
                  fe = fes->GetFE(el);
                  doftrans = fes->GetElementVDofs(el, vdofs);
                  MultSubMatrix(*basis, vdofs, x, el_x);
                  if (doftrans) { doftrans->InvTransformPrimal(el_x); }

                  prev_el = el;
               }

               dnfi[k]->AssembleQuadratureGrad(*fe, *T, ip, sample->qw, el_x, elmat);
               if (doftrans) { doftrans->TransformDual(elmat); }

               AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
               // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
            }  // not (precompute && (dnfi[k]->precomputable))
            jac_timers[3]->Stop();
         }  // for (int i = 0; i < el_samples->Size(); i++)
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   jac_timers[4]->Start();
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
            tr = mesh->GetInteriorFaceTransformations(face);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (fnfi[k]->precomputable))
               fnfi[k]->AddAssembleGrad_Fast(i, sample->qw, *tr, ip, x, *Grad);
            else
            {
               if (face != prev_face)
               {
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

               fnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->qw, el_x, elmat);
               AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
               // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
            }  // not (precompute && (dnfi[k]->precomputable))
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

            tr = mesh->GetBdrFaceTransformations (be);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (bfnfi[k]->precomputable))
               bfnfi[k]->AddAssembleGrad_Fast(i, sample->qw, *tr, ip, x, *Grad);
            else
            {
               if (be != prev_be)
               {
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

               bfnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->qw, el_x, elmat);
               AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
               // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);
            }  // not (precompute && (dnfi[k]->precomputable))
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int k = 0; k < bfnfi.Size(); k++)
   }  // if (bfnfi.Size())

   // if (!Grad->Finalized())
   // {
   //    Grad->Finalize(skip_zeros);
   // }

   DenseMatrix *mGrad = Grad;
   
   jac_timers[4]->Stop();
   jac_timers[5]->Stop();
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
