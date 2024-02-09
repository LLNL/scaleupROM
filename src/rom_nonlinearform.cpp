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
   delete basis;

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

void ROMNonlinearForm::SetBasis(DenseMatrix &basis_, const int offset)
{
   assert(basis_.NumCols() == height);
   assert(basis_.NumRows() >= fes->GetTrueVSize() + offset);
   if (basis) delete basis;

   const int nrow = fes->GetTrueVSize();
   basis = new DenseMatrix(nrow, height);
   basis_.GetSubMatrix(offset, offset+nrow, 0, height, *basis);
}

void ROMNonlinearForm::TrainEQP(const CAROM::Matrix &snapshots, const double eqp_tol)
{
   // NOTE(kevin): these will be resized within the routines as needed.
   // just initializing with distribute option.
   CAROM::Matrix Gt(1,1, true);
   CAROM::Vector rhs_Gw(1, false);

   Array<int> el, qp;
   Array<double> qw;

   for (int k = 0; k < dnfi.Size(); k++)
   {
      SetupEQPSystemForDomainIntegrator(snapshots, dnfi[k], Gt, rhs_Gw);
      TrainEQPForIntegrator(snapshots, dnfi[k], Gt, rhs_Gw, eqp_tol, el, qp, qw);
      UpdateDomainIntegratorSampling(k, el, qp, qw);
   }

   for (int k = 0; k < fnfi.Size(); k++)
   {
      SetupEQPSystemForInteriorFaceIntegrator(snapshots, fnfi[k], Gt, rhs_Gw);
      TrainEQPForIntegrator(snapshots, fnfi[k], Gt, rhs_Gw, eqp_tol, el, qp, qw);
      UpdateDomainIntegratorSampling(k, el, qp, qw);
   }

   for (int k = 0; k < bfnfi.Size(); k++)
   {
      SetupEQPSystemForBdrFaceIntegrator(snapshots, bfnfi[k], Gt, rhs_Gw);
      TrainEQPForIntegrator(snapshots, bfnfi[k], Gt, rhs_Gw, eqp_tol, el, qp, qw);
      UpdateDomainIntegratorSampling(k, el, qp, qw);
   }
}

void ROMNonlinearForm::TrainEQPForIntegrator(
   const CAROM::Matrix &snapshots, HyperReductionIntegrator *nlfi,
   const CAROM::Matrix &Gt, const CAROM::Vector &rhs_Gw, const double eqp_tol,
   Array<int> &sample_el, Array<int> &sample_qp, Array<double> &sample_qw)
{
   const IntegrationRule *ir = nlfi->GetIntegrationRule();

   // TODO(kevin): extension for mixed mesh elements.
   const int vdim = fes->GetVDim();
   const int nqe = ir->GetNPoints();

   //    void SolveNNLS(const int rank, const double nnls_tol, const int maxNNLSnnz,
   // CAROM::Vector const& w, CAROM::Matrix & Gt,
   // CAROM::Vector & sol)
   double nnls_tol = 1.0e-11;
   int maxNNLSnnz = 0;
   CAROM::Vector eqpSol(Gt.numRows(), true);
   int nnz = 0;
   {
      CAROM::NNLSSolver nnls(nnls_tol, 0, maxNNLSnnz, 2);

      CAROM::Vector rhs_ub(rhs_Gw);
      CAROM::Vector rhs_lb(rhs_Gw);

      double delta;
      for (int i = 0; i < rhs_ub.dim(); ++i)
      {
         delta = eqp_tol * abs(rhs_Gw(i));
         rhs_lb(i) -= delta;
         rhs_ub(i) += delta;
      }

      /*
         NOTE(kevin): turn off the normalization now.
         The termination criterion of solve_parallel_with_scalapack
         is currently unnecessarily complicated and redundant.
      */
      // nnls.normalize_constraints(Gt, rhs_lb, rhs_ub);

      /*
         The optimization will continue until
            max_i || rhs_Gw(i) - eqp_Gw(i) || / || rhs_Gw(i) || < eqp_tol
      */
      nnls.solve_parallel_with_scalapack(Gt, rhs_lb, rhs_ub, eqpSol);

      nnz = 0;
      for (int i = 0; i < eqpSol.dim(); ++i)
      {
         if (eqpSol(i) != 0.0)
            nnz++;
      }

      // TODO(kevin): parallel case.
      // std::cout << rank << ": Number of nonzeros in NNLS solution: " << nnz
      //       << ", out of " << eqpSol.dim() << std::endl;

      // MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // if (rank == 0)
      std::cout << "Global number of nonzeros in NNLS solution: " << nnz << std::endl;

      // Check residual of NNLS solution
      CAROM::Vector res(Gt.numColumns(), false);
      Gt.transposeMult(eqpSol, res);

      const double normGsol = res.norm();
      const double normRHS = rhs_Gw.norm();

      res -= rhs_Gw;
      const double relNorm = res.norm() / std::max(normGsol, normRHS);
      // cout << rank << ": relative residual norm for NNLS solution of Gs = Gw: " <<
      //       relNorm << endl;
      std::cout << "Relative residual norm for NNLS solution of Gs = Gw: " <<
                  relNorm << std::endl;
   }

   sample_el.SetSize(0);
   sample_qp.SetSize(0);
   sample_qw.SetSize(0);
   for (int i = 0; i < eqpSol.dim(); ++i)
   {
      if (eqpSol(i) > 1.0e-12)
      {
         const int e = i / nqe;  // Element index
         sample_el.Append(i / nqe);
         sample_qp.Append(i % nqe);
         sample_qw.Append(eqpSol(i));
      }
   }
   printf("Size of sampled qp: %d\n", sample_el.Size());
   if (nnz != sample_el.Size())
      printf("Sample quadrature points with weight < 1.0e-12 are neglected.\n");
}

void ROMNonlinearForm::SetupEQPSystemForDomainIntegrator(
   const CAROM::Matrix &snapshots, HyperReductionIntegrator *nlfi, 
   CAROM::Matrix &Gt, CAROM::Vector &rhs_Gw)
{
   assert(basis);
   assert(snapshots.numRows() >= fes->GetTrueVSize());
   if (snapshots.numRows() > fes->GetTrueVSize())
      mfem_warning("ROMNonlinearForm::SetupEQPSystemForDomainIntegrator- snapshot vector has a larger dimension than finite element space vector dimension. Neglecting the rest of snapshot.\n");

   const IntegrationRule *ir = nlfi->GetIntegrationRule();

   // TODO(kevin): extension for mixed mesh elements.
   const int vdim = fes->GetVDim();
   const int nqe = ir->GetNPoints();
   const int NB = basis->NumCols();
   const int nsnap = snapshots.numColumns();
   const int ne = fes->GetNE();
   const int NQ = ne * nqe;

   // Compute G of size (NB * nsnap) x NQ, but only store its transpose Gt.
   Gt.setSize(NQ, NB * nsnap);
   assert(Gt.distributed());
   // For 0 <= j < NB, 0 <= i < nsnap, 0 <= e < ne, 0 <= m < nqe,
   // G(j + (i*NB), (e*nqe) + m)
   // is the coefficient of v_j^T M(p_i) V v_i at point m of element e,
   // with respect to the integration rule weight at that point,
   // where the "exact" quadrature solution is ir0->GetWeights().

   Vector v_i(fes->GetTrueVSize());
   Vector r(nqe);

   Array<int> vdofs;
   Vector el_x, el_tr;
   DenseMatrix el_quad;
   const FiniteElement *fe;
   ElementTransformation *T;
   DofTransformation *doftrans;

   /* fill out quadrature evaluation of all snapshot-basis weak forms */
   for (int i = 0; i < nsnap; ++i)
   {
      // NOTE(kevin): have to copy the vector since libROM matrix is row-major.
      for (int k = 0; k < fes->GetTrueVSize(); ++k)
         v_i[k] = snapshots(k, i);

      for (int e = 0; e < ne; ++e)
      {
         fe = fes->GetFE(e);
         doftrans = fes->GetElementVDofs(e, vdofs);
         T = fes->GetElementTransformation(e);
         v_i.GetSubVector(vdofs, el_x);

         const int nd = fe->GetDof();
         el_quad.SetSize(nd * vdim, nqe);
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            Vector EQ(el_quad.GetColumn(i), nd * vdim);

            const IntegrationPoint &ip = ir->IntPoint(i);
            nlfi->AssembleQuadratureVector(*fe, *T, ip, 1.0, el_x, EQ);
         }
         // nlfi->AssembleElementQuadrature(*fe, *T, el_x, el_quad);

         for (int j = 0; j < NB; ++j)
         {
            Vector v_j(basis->GetColumn(j), fes->GetVSize());
            v_j.GetSubVector(vdofs, el_tr);

            el_quad.MultTranspose(el_tr, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (e * nqe), j + (i * NB)) = r[m];
         }  // for (int j = 0; j < NB; ++j)
      }  // for (int e = 0; e < ne; ++e)

      // if (precondition)
      // {
         // // Preconditioning is done by (V^T M(p_i) V)^{-1} (of size NB x NB).
         // PreconditionNNLS(fespace_R, new VectorFEMassIntegrator(a_coeff), BR, i, Gt);
      // }
   }  // for (int i = 0; i < nsnap; ++i)

   /* Fill out FOM quadrature weights */
   Array<double> const& w_el = ir->GetWeights();
   CAROM::Vector w(ne * nqe, true);
   for (int i = 0; i < ne; ++i)
      for (int j = 0; j < nqe; ++j)
         w(j + (i * nqe)) = w_el[j];

   rhs_Gw.setSize(Gt.numColumns());
   assert(!rhs_Gw.distributed());
   // rhs = Gw. Note that by using Gt and multTranspose, we do parallel communication.
   Gt.transposeMult(w, rhs_Gw);

   return;
}

void ROMNonlinearForm::GetEQPForDomainIntegrator(
   const int k, Array<int> &el, Array<int> &qp, Array<double> &qw)
{
   assert((k >= 0) && (k < dnfi.Size()));
   assert(dnfi.Size() == dnfi_sample.Size());

   el.SetSize(0);
   qp.SetSize(0);
   qw.SetSize(0);

   Array<SampleInfo> *sample = dnfi_sample[k];

   for (int s = 0; s < sample->Size(); s++)
   {
      el.Append((*sample)[s].el);
      qp.Append((*sample)[s].qp);
      qw.Append((*sample)[s].qw);
   }
}

void ROMNonlinearForm::SaveEQPForDomainIntegrator(const int k, hid_t file_id, const std::string &dsetname)
{
   Array<int> el, qp;
   Array<double> qw;
   GetEQPForDomainIntegrator(k, el, qp, qw);

   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, dsetname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::WriteDataset(grp_id, "elem", el);
   hdf5_utils::WriteDataset(grp_id, "quad-pt", qp);
   hdf5_utils::WriteDataset(grp_id, "quad-wt", qw);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

}
