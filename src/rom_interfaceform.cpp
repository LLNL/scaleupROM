// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "rom_interfaceform.hpp"
#include "etc.hpp"

using namespace std;

namespace mfem
{

ROMInterfaceForm::ROMInterfaceForm(
   Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_, TopologyHandler *topol_)
   : InterfaceForm(meshes_, fes_, topol_), numPorts(topol_->GetNumPorts())
{
   basis.SetSize(numSub);
   basis = NULL;
   basis_dof_offsets.SetSize(numSub);

   // block_offsets should be updated according to the number of basis vectors.
   block_offsets = -1;
}

ROMInterfaceForm::~ROMInterfaceForm()
{
   DeletePointers(fnfi_sample);
}

void ROMInterfaceForm::SetBasisAtSubdomain(const int m, DenseMatrix &basis_, const int offset)
{
   // assert(basis_.NumCols() == height);
   assert(basis_.NumRows() >= fes[m]->GetTrueVSize() + offset);
   assert(basis.Size() == numSub);

   basis[m] = &basis_;
   basis_dof_offsets[m] = offset;
}

void ROMInterfaceForm::UpdateBlockOffsets()
{
   assert(basis.Size() == numSub);
   for (int m = 0; m < numSub; m++)
      assert(basis[m]);

   block_offsets.SetSize(numSub + 1);
   block_offsets = 0;
   for (int i = 1; i < numSub + 1; i++)
      block_offsets[i] = basis[i-1]->NumCols();
   block_offsets.PartialSum();
}

void ROMInterfaceForm::InterfaceAddMult(const Vector &x, Vector &y) const
{     
   assert(block_offsets.Min() >= 0);
   assert(x.Size() == block_offsets.Last());
   assert(y.Size() == block_offsets.Last());

   x_tmp.Update(const_cast<Vector&>(x), block_offsets);
   y_tmp.Update(y, block_offsets);

   /* port-related infos */
   Array<int> midx(2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *fes1, *fes2;
   DenseMatrix *basis1, *basis2;
   int basis1_offset, basis2_offset;
   Vector x1, x2, y1, y2;
   Array<InterfaceInfo> *interface_infos = NULL;
   Array<SampleInfo> *sample_info = NULL;

   /* interface-related infos */
   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   Vector el_x1, el_x2, el_y1, el_y2;

   for (int k = 0; k < fnfi.Size(); k++)
   {
      assert(fnfi[k]);

      const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
      assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

      for (int p = 0; p < numPorts; p++, sample_info++)
      {
         const PortInfo *pInfo = topol_handler->GetPortInfo(p);

         midx[0] = pInfo->Mesh1;
         midx[1] = pInfo->Mesh2;

         mesh1 = meshes[midx[0]];
         mesh2 = meshes[midx[1]];

         fes1 = fes[midx[0]];
         fes2 = fes[midx[1]];

         basis1 = basis[midx[0]];
         basis2 = basis[midx[1]];

         basis1_offset = basis_dof_offsets[midx[0]];
         basis2_offset = basis_dof_offsets[midx[1]];

         x_tmp.GetBlockView(midx[0], x1);
         x_tmp.GetBlockView(midx[1], x2);
         y_tmp.GetBlockView(midx[0], y1);
         y_tmp.GetBlockView(midx[1], y2);

         interface_infos = topol_handler->GetInterfaceInfos(p);
         assert(interface_infos);

         sample_info = fnfi_sample[p + k * numPorts];
         assert(sample_info != NULL);

         int prev_itf = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int itf = sample->itf;
            InterfaceInfo *if_info = &((*interface_infos)[itf]);
            topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (fnfi[k]->precomputable))
               mfem_error("ROMInterfaceForm- precompute mode is not implemented!\n");
            else
            {
               if (itf != prev_itf)
               {
                  if ((tr1 == NULL) || (tr2 == NULL))
                     mfem_error("InterfaceTransformation of the sampled face is NULL,\n"
                              "   indicating that an empty quadrature point is sampled.\n");

                  fes1->GetElementVDofs(tr1->Elem1No, vdofs1);
                  fes2->GetElementVDofs(tr2->Elem1No, vdofs2);

                  if (basis1_offset > 0)
                     for (int v = 0; v < vdofs1.Size(); v++)
                        vdofs1[v] += basis1_offset;
                  if (basis2_offset > 0)
                     for (int v = 0; v < vdofs2.Size(); v++)
                        vdofs2[v] += basis2_offset;

                  // Both domains will have the adjacent element as Elem1.
                  fe1 = fes1->GetFE(tr1->Elem1No);
                  fe2 = fes2->GetFE(tr2->Elem1No);

                  MultSubMatrix(*basis1, vdofs1, x1, el_x1);
                  MultSubMatrix(*basis2, vdofs2, x2, el_x2);

                  prev_itf = itf;
               }  // if (itf != prev_itf)

               fnfi[k]->AssembleQuadratureVector(
                  *fe1, *fe2, *tr1, *tr2, ip, sample->qw, el_x1, el_x2, el_y1, el_y2);

               AddMultTransposeSubMatrix(*basis1, vdofs1, el_y1, y1);
               AddMultTransposeSubMatrix(*basis2, vdofs2, el_y2, y2);
            }  // if not (precompute && (fnfi[k]->precomputable))
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int p = 0; p < numPorts; p++)
   }  // for (int k = 0; k < fnfi.Size(); k++)

   for (int i=0; i < y_tmp.NumBlocks(); ++i)
      y_tmp.GetBlock(i).SyncAliasMemory(y);
}

void ROMInterfaceForm::InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const
{
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         assert(mats(i, j));

   x_tmp.Update(const_cast<Vector&>(x), block_offsets);

   /* port-related infos */
   Array<int> midx(2);
   Array2D<SparseMatrix *> mats_p(2,2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *fes1, *fes2;
   DenseMatrix *basis1, *basis2;
   int basis1_offset, basis2_offset;
   Vector x1, x2;
   Array<InterfaceInfo> *interface_infos = NULL;
   Array<SampleInfo> *sample_info = NULL;

   /* interface-related infos */
   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   Vector el_x1, el_x2;
   Array2D<DenseMatrix *> quadmats(2, 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         quadmats(i, j) = new DenseMatrix;

   for (int k = 0; k < fnfi.Size(); k++)
   {
      assert(fnfi[k]);

      const IntegrationRule *ir = fnfi[k]->GetIntegrationRule();
      assert(ir); // we enforce all integrators to set the IntegrationRule a priori.

      for (int p = 0; p < numPorts; p++, sample_info++)
      {
         const PortInfo *pInfo = topol_handler->GetPortInfo(p);

         midx[0] = pInfo->Mesh1;
         midx[1] = pInfo->Mesh2;

         mesh1 = meshes[midx[0]];
         mesh2 = meshes[midx[1]];

         fes1 = fes[midx[0]];
         fes2 = fes[midx[1]];

         basis1 = basis[midx[0]];
         basis2 = basis[midx[1]];

         basis1_offset = basis_dof_offsets[midx[0]];
         basis2_offset = basis_dof_offsets[midx[1]];

         x_tmp.GetBlockView(midx[0], x1);
         x_tmp.GetBlockView(midx[1], x2);

         for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
               mats_p(i, j) = mats(midx[i], midx[j]);

         interface_infos = topol_handler->GetInterfaceInfos(p);
         assert(interface_infos);

         sample_info = fnfi_sample[p + k * numPorts];
         assert(sample_info != NULL);

         int prev_itf = -1;
         SampleInfo *sample = sample_info->GetData();
         for (int i = 0; i < sample_info->Size(); i++, sample++)
         {
            int itf = sample->itf;
            InterfaceInfo *if_info = &((*interface_infos)[itf]);
            topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);
            const IntegrationPoint &ip = ir->IntPoint(sample->qp);

            if (precompute && (fnfi[k]->precomputable))
               mfem_error("ROMInterfaceForm- precompute mode is not implemented!\n");
            else
            {
               if (itf != prev_itf)
               {
                  if ((tr1 == NULL) || (tr2 == NULL))
                     mfem_error("InterfaceTransformation of the sampled face is NULL,\n"
                              "   indicating that an empty quadrature point is sampled.\n");

                  fes1->GetElementVDofs(tr1->Elem1No, vdofs1);
                  fes2->GetElementVDofs(tr2->Elem1No, vdofs2);

                  if (basis1_offset > 0)
                     for (int v = 0; v < vdofs1.Size(); v++)
                        vdofs1[v] += basis1_offset;
                  if (basis2_offset > 0)
                     for (int v = 0; v < vdofs2.Size(); v++)
                        vdofs2[v] += basis2_offset;

                  // Both domains will have the adjacent element as Elem1.
                  fe1 = fes1->GetFE(tr1->Elem1No);
                  fe2 = fes2->GetFE(tr2->Elem1No);

                  MultSubMatrix(*basis1, vdofs1, x1, el_x1);
                  MultSubMatrix(*basis2, vdofs2, x2, el_x2);

                  prev_itf = itf;
               }  // if (itf != prev_itf)

               fnfi[k]->AssembleQuadratureGrad(
                  *fe1, *fe2, *tr1, *tr2, ip, sample->qw, el_x1, el_x2, quadmats);

               AddSubMatrixRtAP(*basis1, vdofs1, *quadmats(0, 0), *basis1, vdofs1, *mats_p(0, 0));
               AddSubMatrixRtAP(*basis1, vdofs1, *quadmats(0, 1), *basis2, vdofs2, *mats_p(0, 1));
               AddSubMatrixRtAP(*basis2, vdofs2, *quadmats(1, 0), *basis1, vdofs1, *mats_p(1, 0));
               AddSubMatrixRtAP(*basis2, vdofs2, *quadmats(1, 1), *basis2, vdofs2, *mats_p(1, 1));
            }  // if not (precompute && (fnfi[k]->precomputable))
         }  // for (int i = 0; i < sample_info->Size(); i++, sample++)
      }  // for (int p = 0; p < numPorts; p++)
   }  // for (int k = 0; k < fnfi.Size(); k++)

   DeletePointers(quadmats);
}

}
