// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "rom_nonlinearform.hpp"
#include "linalg_utils.hpp"
#include "utils/mpi_utils.h"  // this is from libROM/utils.

using namespace std;

namespace mfem
{

ROMNonlinearForm::ROMNonlinearForm(const int num_basis, FiniteElementSpace *f, const bool reference_)
   : NonlinearForm(f), reference(reference_), timer("ROMNonlinearForm")
{
   height = width = num_basis;

   std::string nnls_str = config.GetOption<std::string>("model_reduction/eqp/criterion", "l2");
   if (nnls_str == "l2")            nnls_criterion = CAROM::NNLS_termination::L2;
   else if (nnls_str == "linf")     nnls_criterion = CAROM::NNLS_termination::LINF;
   else
      mfem_error("ROMNonlinearForm: unknown NNLS criterion!\n");
}

ROMNonlinearForm::~ROMNonlinearForm()
{
   delete Grad;
   delete basis;

   DeletePointers(dnfi);
   DeletePointers(fnfi);
   DeletePointers(bfnfi);
   // Decided to own it, as opposed to the parent class.
   DeletePointers(bfnfi_marker);

   /* only component ROMNonlinearForm owns EQP elements */
   if (reference)
   {
      DeletePointers(dnfi_sample);
      DeletePointers(fnfi_sample);
      DeletePointers(bfnfi_sample);
   }
}

void ROMNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   timer.Start("Mult");

   timer.Start("Mult/init");

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

   timer.Stop("Mult/init");

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         timer.Start("Mult/dnfi-init");

         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         EQPElement *eqp_elem = dnfi_sample[k];
         assert(eqp_elem);

         timer.Stop("Mult/dnfi-init");

         int prev_el = -1;
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            timer.Start("Mult/dnfi-elem-init");

            EQPSample *sample = eqp_elem->GetSample(i);
            int el = sample->info.el;
            T = fes->GetElementTransformation(el);

            timer.Stop("Mult/dnfi-elem-init");

            if (precompute)
            {
               timer.Start("Mult/dnfi-fast-eqp");
               dnfi[k]->AddAssembleVector_Fast(*sample, *T, x, y);
               timer.Stop("Mult/dnfi-fast-eqp");
               continue;
            }

            timer.Start("Mult/dnfi-slow-eqp");

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);
            if (el != prev_el)
            {
               fe = fes->GetFE(el);
               doftrans = fes->GetElementVDofs(el, vdofs);
               MultSubMatrix(*basis, vdofs, x, el_x);
               if (doftrans) { doftrans->InvTransformPrimal(el_x); }

               prev_el = el;
            }

            dnfi[k]->AssembleQuadratureVector(*fe, *T, ip, sample->info.qw, el_x, el_y);
            if (doftrans) { doftrans->TransformDual(el_y); }

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);

            timer.Stop("Mult/dnfi-slow-eqp");
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
         timer.Start("Mult/fnfi-init");

         const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         EQPElement *eqp_elem = fnfi_sample[k];
         assert(eqp_elem);

         timer.Stop("Mult/fnfi-init");

         int prev_face = -1;
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            timer.Start("Mult/fnfi-elem-init");

            EQPSample *sample = eqp_elem->GetSample(i);

            int face = sample->info.el;
            tr = mesh->GetInteriorFaceTransformations(face);

            timer.Stop("Mult/fnfi-elem-init");

            if (precompute)
            {
               timer.Start("Mult/fnfi-fast-eqp");
               fnfi[k]->AddAssembleVector_Fast(*sample, *tr, x, y);
               timer.Stop("Mult/fnfi-fast-eqp");
               continue;
            }

            timer.Start("Mult/fnfi-slow-eqp");

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);

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

            fnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->info.qw, el_x, el_y);

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);

            timer.Stop("Mult/fnfi-slow-eqp");
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
         timer.Start("Mult/bfnfi-init");

         const IntegrationRule *ir = bfnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         EQPElement *eqp_elem = bfnfi_sample[k];
         assert(eqp_elem);

         timer.Stop("Mult/bfnfi-init");

         int prev_be = -1;
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            timer.Start("Mult/bfnfi-elem-init");

            EQPSample *sample = eqp_elem->GetSample(i);
            int be = sample->info.el;
            const int bdr_attr = mesh->GetBdrAttribute(be);
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

            timer.Stop("Mult/bfnfi-elem-init");

            if (precompute)
            {
               timer.Start("Mult/bfnfi-fast-eqp");
               bfnfi[k]->AddAssembleVector_Fast(*sample, *tr, x, y);
               timer.Stop("Mult/bfnfi-fast-eqp");
               continue;
            }

            timer.Start("Mult/bfnfi-slow-eqp");

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);

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

            bfnfi[k]->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, sample->info.qw, el_x, el_y);

            AddMultTransposeSubMatrix(*basis, vdofs, el_y, y);

            timer.Stop("Mult/bfnfi-slow-eqp");
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
   timer.Stop("Mult");
}

Operator& ROMNonlinearForm::GetGradient(const Vector &x) const
{
   timer.Start("GetGradient");

   timer.Start("Grad/init");
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
   timer.Stop("Grad/init");

   if (dnfi.Size())
   {
      for (int k = 0; k < dnfi.Size(); k++)
      {
         timer.Start("Grad/sample-load");
         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         EQPElement *eqp_elem = dnfi_sample[k];
         assert(eqp_elem);

         int prev_el = -1;
         timer.Stop("Grad/sample-load");
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            timer.Start("Grad/elem-load");
            EQPSample *sample = eqp_elem->GetSample(i);
            int el = sample->info.el;
            T = fes->GetElementTransformation(el);
            timer.Stop("Grad/elem-load");

            timer.Start("Grad/assem-grad");
            if (precompute)
            {
               dnfi[k]->AddAssembleGrad_Fast(*sample, *T, x, *Grad);
               timer.Stop("Grad/assem-grad");
               continue;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);
            if (el != prev_el)
            {
               fe = fes->GetFE(el);
               doftrans = fes->GetElementVDofs(el, vdofs);
               MultSubMatrix(*basis, vdofs, x, el_x);
               if (doftrans) { doftrans->InvTransformPrimal(el_x); }

               prev_el = el;
            }

            dnfi[k]->AssembleQuadratureGrad(*fe, *T, ip, sample->info.qw, el_x, elmat);
            if (doftrans) { doftrans->TransformDual(elmat); }

            AddSubMatrixRtAP(*basis, vdofs, elmat, *basis, vdofs, *Grad);
            // Grad->AddSubMatrix(rom_vdofs, rom_vdofs, quadmat, skip_zeros);

            timer.Stop("Grad/assem-grad");
         }  // for (int i = 0; i < el_samples->Size(); i++)
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   timer.Start("Grad/other-integ");
   if (fnfi.Size())
   {
      FaceElementTransformations *tr = NULL;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int k = 0; k < fnfi.Size(); k++)
      {
         const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
         assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

         EQPElement *eqp_elem = fnfi_sample[k];
         assert(eqp_elem);

         int prev_face = -1;
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            EQPSample *sample = eqp_elem->GetSample(i);
            int face = sample->info.el;
            tr = mesh->GetInteriorFaceTransformations(face);

            if (precompute)
            {
               fnfi[k]->AddAssembleGrad_Fast(*sample, *tr, x, *Grad);
               continue;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);

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

            fnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->info.qw, el_x, elmat);
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

         EQPElement *eqp_elem = bfnfi_sample[k];
         assert(eqp_elem);

         int prev_be = -1;
         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            EQPSample *sample = eqp_elem->GetSample(i);
            int be = sample->info.el;
            const int bdr_attr = mesh->GetBdrAttribute(be);
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

            if (precompute)
            {
               bfnfi[k]->AddAssembleGrad_Fast(*sample, *tr, x, *Grad);
               continue;
            }

            const IntegrationPoint &ip = ir->IntPoint(sample->info.qp);

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

            bfnfi[k]->AssembleQuadratureGrad(*fe1, *fe2, *tr, ip, sample->info.qw, el_x, elmat);
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
   
   timer.Stop("Grad/other-integ");
   timer.Stop("GetGradient");
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
         const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
         EQPElement *eqp_elem = dnfi_sample[k];
         assert(ir);
         assert(eqp_elem);

         for (int i = 0; i < eqp_elem->Size(); i++)
            PrecomputeDomainEQPSample(*ir, *basis, *eqp_elem->GetSample(i));
      }  // for (int k = 0; k < dnfi.Size(); k++)
   }  // if (dnfi.Size())

   if (fnfi.Size())
   {
      for (int k = 0; k < fnfi.Size(); k++)
      {
         const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
         EQPElement *eqp_elem = fnfi_sample[k];
         assert(ir);
         assert(eqp_elem);

         for (int i = 0; i < eqp_elem->Size(); i++)
            PrecomputeInteriorFaceEQPSample(*ir, *basis, *eqp_elem->GetSample(i));
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
         const IntegrationRule *ir = bfnfi[k]->GetIntegrationRule();
         EQPElement *eqp_elem = bfnfi_sample[k];
         assert(ir);
         assert(eqp_elem);

         for (int i = 0; i < eqp_elem->Size(); i++)
         {
            EQPSample *sample = eqp_elem->GetSample(i);
            int be = sample->info.el;

            const int bdr_attr = mesh->GetBdrAttribute(be);
            if (bfnfi_marker[k] &&
                   (*bfnfi_marker[k])[bdr_attr-1] == 0)
            { 
               // for EQP, this indicates an ill sampling.
               // for other hyper reductions, we can simply continue the loop.
               mfem_error("The sampled boundary element does not have a boundary condition,\n"
                           "   indicating that an empty quadrature point is sampled.\n");
               // continue;
            }

            PrecomputeBdrFaceEQPSample(*ir, *basis, *sample);
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
   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider local matrix construction from local snapshot/basis matrix.
   */
   assert(snapshots.distributed());
   CAROM::Matrix snapshots_work(snapshots);
   snapshots_work.gather();
   const int nrow_global = snapshots_work.numRows();
   assert(basis);
   assert(nrow_global >= fes->GetTrueVSize());
   if (nrow_global > fes->GetTrueVSize())
      mfem_warning("ROMNonlinearForm::TrainEQP- snapshot vector has a larger dimension than finite element space vector dimension. Neglecting the rest of snapshot.\n");

   // NOTE(kevin): these will be resized within the routines as needed.
   // just initializing with distribute option.
   CAROM::Matrix Gt(1,1, true);
   CAROM::Vector rhs_Gw(1, false);

   Array<SampleInfo> samples;
   Array<int> fidxs;
   Mesh *mesh = fes->GetMesh();

   for (int k = 0; k < dnfi.Size(); k++)
   {
      SetupEQPSystemForDomainIntegrator(snapshots_work, dnfi[k], Gt, rhs_Gw);
      TrainEQPForIntegrator(dnfi[k], Gt, rhs_Gw, eqp_tol, samples);
      UpdateDomainIntegratorSampling(k, samples);
   }

   for (int k = 0; k < fnfi.Size(); k++)
   {
      SetupEQPSystemForInteriorFaceIntegrator(snapshots_work, fnfi[k], Gt, rhs_Gw, fidxs);
      TrainEQPForIntegrator(fnfi[k], Gt, rhs_Gw, eqp_tol, samples);
      for (int s = 0; s < samples.Size(); s++)
         samples[s].el = fidxs[samples[s].el];
      UpdateInteriorFaceIntegratorSampling(k, samples);
   }

   // Which boundary attributes need to be processed?
   Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                              mesh->bdr_attributes.Max() : 0);

   for (int k = 0; k < bfnfi.Size(); k++)
   {
      // Determine the boundary attributes to process for k-th boundary face integrator.
      bdr_attr_marker = 0;
      if (bfnfi_marker[k] == NULL)
         bdr_attr_marker = 1;
      else
      {
         Array<int> &bdr_marker = *bfnfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      SetupEQPSystemForBdrFaceIntegrator(snapshots_work, bfnfi[k], bdr_attr_marker, Gt, rhs_Gw, fidxs);
      TrainEQPForIntegrator(bfnfi[k], Gt, rhs_Gw, eqp_tol, samples);
      for (int s = 0; s < samples.Size(); s++)
         samples[s].el = fidxs[samples[s].el];
      UpdateBdrFaceIntegratorSampling(k, samples);
   }
}

void ROMNonlinearForm::TrainEQPForIntegrator(
   HyperReductionIntegrator *nlfi, const CAROM::Matrix &Gt, const CAROM::Vector &rhs_Gw,
   const double eqp_tol, Array<SampleInfo> &samples)
{
   const IntegrationRule *ir = nlfi->GetIntegrationRule();

   // TODO(kevin): extension for mixed mesh elements.
   const int vdim = fes->GetVDim();
   const int nqe = ir->GetNPoints();

   //    void SolveNNLS(const int rank, const double nnls_tol, const int maxNNLSnnz,
   // CAROM::Vector const& w, CAROM::Matrix & Gt,
   // CAROM::Vector & sol)
   const double normRHS = rhs_Gw.norm();
   double nnls_tol = 1.0e-11;
   int maxNNLSnnz = 0;
   CAROM::Vector eqpSol(Gt.numRows(), true);
   int nnz = 0;
   {
      CAROM::NNLSSolver nnls(nnls_tol, 0, maxNNLSnnz, 2, 1.0e-4, 1.0e-14,
                             100000, 100000, nnls_criterion);

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

      res -= rhs_Gw;
      const double relNorm = res.norm() / std::max(normGsol, normRHS);
      // cout << rank << ": relative residual norm for NNLS solution of Gs = Gw: " <<
      //       relNorm << endl;
      std::cout << "Relative residual norm for NNLS solution of Gs = Gw: " <<
                  relNorm << std::endl;
   }

   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will treat EQ points/weights locally per each process.
   */
   std::vector<int> eqp_sol_offsets, eqp_sol_cnts;
   int eqp_sol_dim = CAROM::get_global_offsets(eqpSol.dim(), eqp_sol_offsets, MPI_COMM_WORLD);
   for (int k = 0; k < eqp_sol_offsets.size() - 1; k++)
      eqp_sol_cnts.push_back(eqp_sol_offsets[k + 1] - eqp_sol_offsets[k]);
   CAROM::Vector eqpSol_global(eqp_sol_dim, false);
   MPI_Allgatherv(eqpSol.getData(), eqpSol.dim(), MPI_DOUBLE, eqpSol_global.getData(),
                  eqp_sol_cnts.data(), eqp_sol_offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

   samples.SetSize(0);
   for (int i = 0; i < eqpSol_global.dim(); ++i)
   {
      if (eqpSol_global(i) > 1.0e-12)
      {
         const int e = i / nqe;  // Element index
         samples.Append({.el = i / nqe, .qp = i % nqe, .qw = eqpSol_global(i)});
      }
   }
   printf("Size of sampled qp: %d/%d\n", samples.Size(), eqpSol_global.dim());
   if (nnz != samples.Size())
      printf("Sample quadrature points with weight < 1.0e-12 are neglected.\n");
}

void ROMNonlinearForm::SetupEQPSystemForDomainIntegrator(
   const CAROM::Matrix &snapshots, HyperReductionIntegrator *nlfi, 
   CAROM::Matrix &Gt, CAROM::Vector &rhs_Gw)
{
   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider local matrix construction from local snapshot/basis matrix.
      Also, while snapshot/basis are distributed according to vdofs,
      EQP system will be distributed according to elements.
   */
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   assert(!snapshots.distributed());
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

   const int ne_global = fes->GetNE();
   const int ne = CAROM::split_dimension(ne_global, MPI_COMM_WORLD);
   std::vector<int> elem_offsets;
   int dummy = CAROM::get_global_offsets(ne, elem_offsets, MPI_COMM_WORLD);
   assert(dummy == ne_global);

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

      for (int e = elem_offsets[rank], eidx = 0; e < elem_offsets[rank+1]; e++, eidx++)
      {
         fe = fes->GetFE(e);
         doftrans = fes->GetElementVDofs(e, vdofs);
         T = fes->GetElementTransformation(e);
         v_i.GetSubVector(vdofs, el_x);

         if (doftrans) { doftrans->InvTransformPrimal(el_x); }

         const int nd = fe->GetDof();
         el_quad.SetSize(nd * vdim, nqe);
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            Vector EQ(el_quad.GetColumn(i), nd * vdim);

            const IntegrationPoint &ip = ir->IntPoint(i);
            nlfi->AssembleQuadratureVector(*fe, *T, ip, 1.0, el_x, EQ);
            if (doftrans) { doftrans->TransformDual(EQ); }
         }
         // nlfi->AssembleElementQuadrature(*fe, *T, el_x, el_quad);

         for (int j = 0; j < NB; ++j)
         {
            Vector v_j(basis->GetColumn(j), fes->GetVSize());
            v_j.GetSubVector(vdofs, el_tr);

            el_quad.MultTranspose(el_tr, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (eidx * nqe), j + (i * NB)) = r[m];
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

void ROMNonlinearForm::SetupEQPSystemForInteriorFaceIntegrator(
   const CAROM::Matrix &snapshots, HyperReductionIntegrator *nlfi, 
   CAROM::Matrix &Gt, CAROM::Vector &rhs_Gw, Array<int> &fidxs)
{
   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider local matrix construction from local snapshot/basis matrix.
      Also, while snapshot/basis are distributed according to vdofs,
      EQP system will be distributed according to elements.
   */
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   assert(!snapshots.distributed());
   assert(basis);
   assert(snapshots.numRows() >= fes->GetTrueVSize());
   if (snapshots.numRows() > fes->GetTrueVSize())
      mfem_warning("ROMNonlinearForm::SetupEQPSystemForDomainIntegrator- snapshot vector has a larger dimension than finite element space vector dimension. Neglecting the rest of snapshot.\n");

   const IntegrationRule *ir = nlfi->GetIntegrationRule();
   Mesh *mesh = fes->GetMesh();

   // TODO(kevin): extension for mixed mesh elements.
   const int vdim = fes->GetVDim();
   const int nqe = ir->GetNPoints();
   const int NB = basis->NumCols();
   const int nsnap = snapshots.numColumns();

   /* get faces with non-trivial interfaces. */
   fidxs.SetSize(0);
   FaceElementTransformations *tr;
   for (int f = 0; f < mesh->GetNumFaces(); f++)
   {
      tr = mesh->GetInteriorFaceTransformations(f);
      if (tr != NULL) fidxs.Append(f);
   }

   const int ne_global = fidxs.Size();
   const int ne = CAROM::split_dimension(ne_global, MPI_COMM_WORLD);
   std::vector<int> elem_offsets;
   int dummy = CAROM::get_global_offsets(ne, elem_offsets, MPI_COMM_WORLD);
   assert(dummy == ne_global);

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

   Array<int> vdofs, vdofs2;
   Vector el_x, el_tr;
   DenseMatrix el_quad;
   
   const FiniteElement *fe1, *fe2;

   /* fill out quadrature evaluation of all snapshot-basis weak forms */
   for (int i = 0; i < nsnap; ++i)
   {
      // NOTE(kevin): have to copy the vector since libROM matrix is row-major.
      for (int k = 0; k < fes->GetTrueVSize(); ++k)
         v_i[k] = snapshots(k, i);

      for (int e = elem_offsets[rank], eidx = 0; e < elem_offsets[rank+1]; e++, eidx++)
      {
         tr = mesh->GetInteriorFaceTransformations(fidxs[e]);
         assert(tr != NULL);

         fes->GetElementVDofs(tr->Elem1No, vdofs);
         fes->GetElementVDofs(tr->Elem2No, vdofs2);
         vdofs.Append (vdofs2);

         v_i.GetSubVector(vdofs, el_x);

         fe1 = fes->GetFE(tr->Elem1No);
         fe2 = fes->GetFE(tr->Elem2No);

         const int nd = fe1->GetDof() + fe2->GetDof();
         el_quad.SetSize(nd * vdim, nqe);
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            Vector EQ(el_quad.GetColumn(i), nd * vdim);

            const IntegrationPoint &ip = ir->IntPoint(i);
            nlfi->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, 1.0, el_x, EQ);
         }

         for (int j = 0; j < NB; ++j)
         {
            Vector v_j(basis->GetColumn(j), fes->GetVSize());
            v_j.GetSubVector(vdofs, el_tr);

            el_quad.MultTranspose(el_tr, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (eidx * nqe), j + (i * NB)) = r[m];
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

void ROMNonlinearForm::SetupEQPSystemForBdrFaceIntegrator(
   const CAROM::Matrix &snapshots, HyperReductionIntegrator *nlfi, const Array<int> &bdr_attr_marker,
   CAROM::Matrix &Gt, CAROM::Vector &rhs_Gw, Array<int> &bidxs)
{
   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider local matrix construction from local snapshot/basis matrix.
      Also, while snapshot/basis are distributed according to vdofs,
      EQP system will be distributed according to elements.
   */
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   assert(!snapshots.distributed());
   assert(basis);
   assert(snapshots.numRows() >= fes->GetTrueVSize());
   if (snapshots.numRows() > fes->GetTrueVSize())
      mfem_warning("ROMNonlinearForm::SetupEQPSystemForDomainIntegrator- snapshot vector has a larger dimension than finite element space vector dimension. Neglecting the rest of snapshot.\n");

   const IntegrationRule *ir = nlfi->GetIntegrationRule();
   Mesh *mesh = fes->GetMesh();

   // TODO(kevin): extension for mixed mesh elements.
   const int vdim = fes->GetVDim();
   const int nqe = ir->GetNPoints();
   const int NB = basis->NumCols();
   const int nsnap = snapshots.numColumns();

   /* get BEs with non-trivial boundary face. */
   bidxs.SetSize(0);
   FaceElementTransformations *tr;
   for (int b = 0; b < fes->GetNBE(); b++)
   {
      const int bdr_attr = mesh->GetBdrAttribute(b);
      if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

      tr = mesh->GetBdrFaceTransformations (b);
      if (tr != NULL) bidxs.Append(b);
   }

   const int ne_global = bidxs.Size();
   const int ne = CAROM::split_dimension(ne_global, MPI_COMM_WORLD);
   std::vector<int> elem_offsets;
   int dummy = CAROM::get_global_offsets(ne, elem_offsets, MPI_COMM_WORLD);
   assert(dummy == ne_global);

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

   Array<int> vdofs, vdofs2;
   Vector el_x, el_tr;
   DenseMatrix el_quad;
   
   const FiniteElement *fe1, *fe2;

   /* fill out quadrature evaluation of all snapshot-basis weak forms */
   for (int i = 0; i < nsnap; ++i)
   {
      // NOTE(kevin): have to copy the vector since libROM matrix is row-major.
      for (int k = 0; k < fes->GetTrueVSize(); ++k)
         v_i[k] = snapshots(k, i);

      for (int e = elem_offsets[rank], eidx = 0; e < elem_offsets[rank+1]; e++, eidx++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(bidxs[e]);
         tr = mesh->GetBdrFaceTransformations(bidxs[e]);
         assert(tr != NULL);
         assert(bdr_attr_marker[bdr_attr-1] != 0);

         fes->GetElementVDofs(tr->Elem1No, vdofs);

         v_i.GetSubVector(vdofs, el_x);

         fe1 = fes->GetFE(tr->Elem1No);
         // The fe2 object is really a dummy and not used on the boundaries,
         // but we can't dereference a NULL pointer, and we don't want to
         // actually make a fake element.
         fe2 = fe1;

         const int nd = fe1->GetDof();
         el_quad.SetSize(nd * vdim, nqe);
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            Vector EQ(el_quad.GetColumn(i), nd * vdim);

            const IntegrationPoint &ip = ir->IntPoint(i);
            nlfi->AssembleQuadratureVector(*fe1, *fe2, *tr, ip, 1.0, el_x, EQ);
         }

         for (int j = 0; j < NB; ++j)
         {
            Vector v_j(basis->GetColumn(j), fes->GetVSize());
            v_j.GetSubVector(vdofs, el_tr);

            el_quad.MultTranspose(el_tr, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (eidx * nqe), j + (i * NB)) = r[m];
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

EQPElement* ROMNonlinearForm::GetEQPForIntegrator(const IntegratorType type, const int k)
{
   EQPElement *sample = NULL;
   switch (type)
   {
      case IntegratorType::DOMAIN:
      {
         assert((k >= 0) && (k < dnfi.Size()));
         assert(dnfi.Size() == dnfi_sample.Size());
         sample = dnfi_sample[k];
      }
      break;
      case IntegratorType::INTERIORFACE:
      {
         assert((k >= 0) && (k < fnfi.Size()));
         assert(fnfi.Size() == fnfi_sample.Size());
         sample = fnfi_sample[k];
      }
      break;
      case IntegratorType::BDRFACE:
      {
         assert((k >= 0) && (k < bfnfi.Size()));
         assert(bfnfi.Size() == bfnfi_sample.Size());
         sample = bfnfi_sample[k];
      }
      break;
      default:
         mfem_error("Unknown Integrator type!\n");
   }

   return sample;
}

void ROMNonlinearForm::SaveEQPForIntegrator(
   const IntegratorType type, const int k, hid_t file_id, const std::string &dsetname)
{
   switch (type)
   {
      case IntegratorType::DOMAIN:
      {
         assert((k >= 0) && (k < dnfi.Size()));
         assert(dnfi.Size() == dnfi_sample.Size());
         dnfi_sample[k]->Save(file_id, dsetname, type);
      }
      break;
      case IntegratorType::INTERIORFACE:
      {
         assert((k >= 0) && (k < fnfi.Size()));
         assert(fnfi.Size() == fnfi_sample.Size());
         fnfi_sample[k]->Save(file_id, dsetname, type);
      }
      break;
      case IntegratorType::BDRFACE:
      {
         assert((k >= 0) && (k < bfnfi.Size()));
         assert(bfnfi.Size() == bfnfi_sample.Size());
         bfnfi_sample[k]->Save(file_id, dsetname, type);
      }
      break;
      default:
         mfem_error("Unknown Integrator type!\n");
   }

   return;
}

void ROMNonlinearForm::LoadEQPForIntegrator(
   const IntegratorType type, const int k, hid_t file_id, const std::string &dsetname)
{
   if (!reference)
      mfem_error("ROMNonlinearForm::LoadEQPForIntegrator is allowed only for reference!\n");

   switch (type)
   {
      case IntegratorType::DOMAIN:
      {
         assert((k >= 0) && (k < dnfi.Size()));
         assert(dnfi.Size() == dnfi_sample.Size());
         if (dnfi_sample[k])
            delete dnfi_sample[k];

         dnfi_sample[k] = new EQPElement;
         dnfi_sample[k]->Load(file_id, dsetname, type);
      }
      break;
      case IntegratorType::INTERIORFACE:
      {
         assert((k >= 0) && (k < fnfi.Size()));
         assert(fnfi.Size() == fnfi_sample.Size());
         if (fnfi_sample[k])
            delete fnfi_sample[k];

         fnfi_sample[k] = new EQPElement;
         fnfi_sample[k]->Load(file_id, dsetname, type);
      }
      break;
      case IntegratorType::BDRFACE:
      {
         assert((k >= 0) && (k < bfnfi.Size()));
         assert(bfnfi.Size() == bfnfi_sample.Size());
         if (bfnfi_sample[k])
            delete bfnfi_sample[k];
            
         bfnfi_sample[k] = new EQPElement;
         bfnfi_sample[k]->Load(file_id, dsetname, type);
      }
      break;
      default:
         mfem_error("Unknown Integrator type!\n");
   }

   return;
}

void ROMNonlinearForm::PrecomputeDomainEQPSample(
   const IntegrationRule &ir, const DenseMatrix &basis, EQPSample &eqp_sample)
{
   const int nbasis = basis.NumCols();
   const int el = eqp_sample.info.el;

   const FiniteElement *fe = fes->GetFE(el);
   Array<int> vdofs;
   // TODO(kevin): not exactly sure what doftrans impacts.
   DofTransformation *doftrans = fes->GetElementVDofs(el, vdofs);
   ElementTransformation *T = fes->GetElementTransformation(el);
   const IntegrationPoint &ip = ir.IntPoint(eqp_sample.info.qp);

   const int nd = fe->GetDof();
   int dim = fe->GetDim();

   Vector shape(nd);
   DenseMatrix dshape(nd, dim), EF;

   T->SetIntPoint(&ip);
   fe->CalcShape(ip, shape);
   fe->CalcPhysDShape(*T, dshape);

   Vector basis_i;
   DenseMatrix *vec1s = new DenseMatrix(dim, nbasis);
   eqp_sample.dshape1.SetSize(0);
   for (int i = 0; i < nbasis; i++)
   {
      GetBasisElement(basis, i, vdofs, basis_i, doftrans);
      EF.UseExternalData(basis_i.GetData(), nd, dim);

      Vector vec1;
      vec1s->GetColumnReference(i, vec1);
      EF.MultTranspose(shape, vec1);

      DenseMatrix *gradEF1 = new DenseMatrix(dim);
      MultAtB(EF, dshape, *gradEF1);
      eqp_sample.dshape1.Append(gradEF1);
   }
   eqp_sample.shape1 = vec1s;
}

void ROMNonlinearForm::PrecomputeFaceEQPSample(
   const IntegrationRule &ir, const DenseMatrix &basis,
   FaceElementTransformations *T, EQPSample &eqp_sample)
{
   const int nbasis = basis.NumCols();

   const bool el2 = (T->Elem2No >= 0);

   const FiniteElement *fe1 = fes->GetFE(T->Elem1No);
   const FiniteElement *fe2 = (el2) ? fes->GetFE(T->Elem2No) : fe1;

   Array<int> vdofs, vdofs2;
   fes->GetElementVDofs(T->Elem1No, vdofs);
   if (el2)
   {
      fes->GetElementVDofs(T->Elem2No, vdofs2);
      vdofs.Append(vdofs2);
   }

   int dim = fe1->GetDim();
   int ndofs1 = fe1->GetDof();
   int ndofs2 = (el2) ? fe2->GetDof() : 0;

   const IntegrationPoint &ip = ir.IntPoint(eqp_sample.info.qp);

   Vector shape1, shape2;

   shape1.SetSize(ndofs1);
   if (el2) shape2.SetSize(ndofs2);

   T->SetAllIntPoints(&ip);

   const IntegrationPoint &eip1 = T->GetElement1IntPoint();
   fe1->CalcShape(eip1, shape1);
   if (el2)
   {
      const IntegrationPoint &eip2 = T->GetElement2IntPoint();
      fe2->CalcShape(eip2, shape2);
   }

   Vector basis_i;
   DenseMatrix *vec1s, *vec2s;
   vec1s = new DenseMatrix(dim, nbasis);
   if (el2) vec2s = new DenseMatrix(dim, nbasis);

   Vector vec1, vec2;
   DenseMatrix elv1, elv2;
   for (int i = 0; i < nbasis; i++)
   {
      GetBasisElement(basis, i, vdofs, basis_i);
      elv1.UseExternalData(basis_i.GetData(), ndofs1, dim);
      if (el2) elv2.UseExternalData(basis_i.GetData() + ndofs1 * dim, ndofs2, dim);

      vec1s->GetColumnReference(i, vec1);
      elv1.MultTranspose(shape1, vec1);

      if (el2)
      {
         vec2s->GetColumnReference(i, vec2);
         elv2.MultTranspose(shape2, vec2);
      }
   }

   eqp_sample.shape1 = vec1s;
   if (el2) eqp_sample.shape2 = vec2s;

   // TODO(kevin): compute dshape1 and dshape2 as well.
}

void ROMNonlinearForm::PrecomputeInteriorFaceEQPSample(
   const IntegrationRule &ir, const DenseMatrix &basis, EQPSample &eqp_sample)
{
   const int face = eqp_sample.info.el;
   FaceElementTransformations *T = fes->GetMesh()->GetInteriorFaceTransformations(face);
   assert(T != NULL);
   assert(T->Elem2No >= 0);

   PrecomputeFaceEQPSample(ir, basis, T, eqp_sample);
}

void ROMNonlinearForm::PrecomputeBdrFaceEQPSample(
   const IntegrationRule &ir, const DenseMatrix &basis, EQPSample &eqp_sample)
{
   const int be = eqp_sample.info.el;
   FaceElementTransformations *T = fes->GetMesh()->GetBdrFaceTransformations(be);
   assert(T != NULL);
   assert(T->Elem2No < 0);

   PrecomputeFaceEQPSample(ir, basis, T, eqp_sample);
}

void ROMNonlinearForm::SaveDomainEQPCoords(const int k, hid_t file_id, const std::string &dsetname)
{
   if (!dnfi.Size())
      return;

   assert((k >= 0) && (k < dnfi.Size()));

   const IntegrationRule *ir = dnfi[k]->GetIntegrationRule();
   EQPElement *eqp_elem = dnfi_sample[k];
   assert(ir);
   assert(eqp_elem);

   DenseMatrix eqp_coords(eqp_elem->Size(), 3);

   EQPSample *eqp_sample;
   for (int i = 0; i < eqp_elem->Size(); i++)
   {
      eqp_sample = eqp_elem->GetSample(i);
      const int el = eqp_sample->info.el;
      const FiniteElement *fe = fes->GetFE(el);
      ElementTransformation *T = fes->GetElementTransformation(el);
      const IntegrationPoint &ip = ir->IntPoint(eqp_sample->info.qp);

      double x[3];
      Vector transip(x, 3);
      T->Transform(ip, transip);

      eqp_coords.SetRow(i, transip);
   }  // for (int i = 0; i < eqp_elem->Size(); i++)

   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, dsetname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   hdf5_utils::WriteDataset(grp_id, "coords", eqp_coords);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

}
