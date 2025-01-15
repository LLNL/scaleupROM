// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "interface_form.hpp"
#include "etc.hpp"

using namespace std;

namespace mfem
{

InterfaceForm::InterfaceForm(
   Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_, TopologyHandler *topol_)
   : meshes(meshes_), fes(fes_), topol_handler(topol_), numSub(meshes_.Size()), timer()
{
   assert(fes_.Size() == numSub);

   block_offsets.SetSize(numSub + 1);
   block_offsets = 0;
   for (int i = 1; i < numSub + 1; i++)
      block_offsets[i] = fes[i-1]->GetTrueVSize();
   block_offsets.PartialSum();
}

InterfaceForm::~InterfaceForm()
{
   DeletePointers(fnfi);

   timer.Print("InterfaceForm::InterfaceAddMult");
}

void InterfaceForm::AssembleInterfaceMatrices(Array2D<SparseMatrix *> &mats) const
{
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++) assert(mats(i, j));

   const PortInfo *pInfo;
   Array<int> midx(2);
   Array2D<SparseMatrix *> mats_p(2,2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *fes1, *fes2;
   Array<InterfaceInfo>* interface_infos;

   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      pInfo = topol_handler->GetPortInfo(p);

      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;
      
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) mats_p(i, j) = mats(midx[i], midx[j]);

      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      fes1 = fes[midx[0]];
      fes2 = fes[midx[1]];

      interface_infos = topol_handler->GetInterfaceInfos(p);
      AssembleInterfaceMatrix(mesh1, mesh2, fes1, fes2, interface_infos, mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void InterfaceForm::InterfaceAddMult(const Vector &x, Vector &y) const
{
   timer.Start("Total");
   timer.Start("init");

   x_tmp.Update(const_cast<Vector&>(x), block_offsets);
   y_tmp.Update(y, block_offsets);

   Array<int> midx(2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *fes1, *fes2;

   timer.Stop("init");

   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      timer.Start("port-init");

      const PortInfo *pInfo = topol_handler->GetPortInfo(p);

      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;

      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      fes1 = fes[midx[0]];
      fes2 = fes[midx[1]];

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);

      timer.Stop("port-init");

      AssembleInterfaceVector(mesh1, mesh2, fes1, fes2, interface_infos,
                              x_tmp.GetBlock(midx[0]), x_tmp.GetBlock(midx[1]),
                              y_tmp.GetBlock(midx[0]), y_tmp.GetBlock(midx[1]));
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)

   timer.Start("final");

   for (int i=0; i < y_tmp.NumBlocks(); ++i)
      y_tmp.GetBlock(i).SyncAliasMemory(y);

   timer.Stop("final");
   timer.Stop("Total");
}

void InterfaceForm::InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const
{
   assert(mats.NumRows() == numSub);
   assert(mats.NumCols() == numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         assert(mats(i, j));

   x_tmp.Update(const_cast<Vector&>(x), block_offsets);

   Array<int> midx(2);
   Array2D<SparseMatrix *> mats_p(2,2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *fes1, *fes2;

   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);
      
      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            mats_p(i, j) = mats(midx[i], midx[j]);

      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      fes1 = fes[midx[0]];
      fes2 = fes[midx[1]];

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);
      AssembleInterfaceGrad(mesh1, mesh2, fes1, fes2, interface_infos,
                            x_tmp.GetBlock(midx[0]), x_tmp.GetBlock(midx[1]), mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void InterfaceForm::AssembleInterfaceMatrix(
   Mesh *mesh1, Mesh *mesh2, FiniteElementSpace *fes1, FiniteElementSpace *fes2,
   Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats) const
{
   Array2D<DenseMatrix*> elemmats(2, 2);
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) elemmats(i, j) = new DenseMatrix;
   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;
   Array<Array<int> *> vdofs(2);
   vdofs[0] = new Array<int>;
   vdofs[1] = new Array<int>;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
         fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         for (int itg = 0; itg < fnfi.Size(); itg++)
         {
            assert(fnfi[itg]);
            fnfi[itg]->AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);

            for (int i = 0; i < 2; i++) {
               for (int j = 0; j < 2; j++) {
                  mats(i, j)->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
               }
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)

   DeletePointers(elemmats);
   DeletePointers(vdofs);
}

void InterfaceForm::AssembleInterfaceMatrixAtPort(
   const int p, Array<FiniteElementSpace *> &fes_comp, Array2D<SparseMatrix *> &mats_p) const
{
   const int num_ref_ports = topol_handler->GetNumRefPorts();

   assert(topol_handler->GetType() == TopologyHandlerMode::COMPONENT);
   assert((p >= 0) && (p < num_ref_ports));
   DeletePointers(mats_p);
   mats_p.SetSize(2, 2);

   int c1, c2;
   topol_handler->GetComponentPair(p, c1, c2);

   // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
   // Need to use two copied instances.
   Mesh *comp1 = topol_handler->GetComponentMesh(c1);
   Mesh *comp2;
   if (c1 == c2)
      comp2 = new Mesh(*comp1);
   else
      comp2 = topol_handler->GetComponentMesh(c2);

   Array<int> c_idx(2);
   c_idx[0] = c1;
   c_idx[1] = c2;

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         mats_p(i, j) = new SparseMatrix(fes_comp[c_idx[i]]->GetTrueVSize(), fes_comp[c_idx[j]]->GetTrueVSize());

   Array<InterfaceInfo> *if_infos = topol_handler->GetRefInterfaceInfos(p);

   // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
   // Need to use two copied instances.
   AssembleInterfaceMatrix(comp1, comp2, fes_comp[c1], fes_comp[c2], if_infos, mats_p);

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) mats_p(i, j)->Finalize();

   if (c1 == c2)
      delete comp2;
}

void InterfaceForm::AssembleInterfaceVector(Mesh *mesh1, Mesh *mesh2,
   FiniteElementSpace *fes1, FiniteElementSpace *fes2, Array<InterfaceInfo> *interface_infos,
   const Vector &x1, const Vector &x2, Vector &y1, Vector &y2) const
{
   assert(x1.Size() == fes1->GetTrueVSize());
   assert(x2.Size() == fes2->GetTrueVSize());
   assert(y1.Size() == x1.Size());
   assert(y2.Size() == x2.Size());

   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   Vector el_x1, el_x2, el_y1, el_y2;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      timer.Start("topol");

      InterfaceInfo *if_info = &((*interface_infos)[bn]);

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      timer.Stop("topol");

      timer.Start("assemble");

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, vdofs1);
         fes2->GetElementVDofs(tr2->Elem1No, vdofs2);

         x1.GetSubVector(vdofs1, el_x1);
         x2.GetSubVector(vdofs2, el_x2);

         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         for (int itg = 0; itg < fnfi.Size(); itg++)
         {
            assert(fnfi[itg]);

            timer.Start("assemble-itf-vec");
            fnfi[itg]->AssembleInterfaceVector(*fe1, *fe2, *tr1, *tr2, el_x1, el_x2, el_y1, el_y2);
            timer.Stop("assemble-itf-vec");

            y1.AddElementVector(vdofs1, el_y1);
            y2.AddElementVector(vdofs2, el_y2);
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))

      timer.Stop("assemble");
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}

void InterfaceForm::AssembleInterfaceGrad(Mesh *mesh1, Mesh *mesh2,
   FiniteElementSpace *fes1, FiniteElementSpace *fes2, Array<InterfaceInfo> *interface_infos,
   const Vector &x1, const Vector &x2, Array2D<SparseMatrix*> &mats) const
{
   assert(x1.Size() == fes1->GetTrueVSize());
   assert(x2.Size() == fes2->GetTrueVSize());

   Array2D<DenseMatrix*> elemmats(2, 2);
   for (int i = 0; i < elemmats.NumRows(); i++)
      for (int j = 0; j < elemmats.NumCols(); j++) elemmats(i, j) = new DenseMatrix; 
   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   Vector el_x1, el_x2, el_y1, el_y2;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, vdofs1);
         fes2->GetElementVDofs(tr2->Elem1No, vdofs2);

         x1.GetSubVector(vdofs1, el_x1);
         x2.GetSubVector(vdofs2, el_x2);

         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         for (int itg = 0; itg < fnfi.Size(); itg++)
         {
            assert(fnfi[itg]);

            fnfi[itg]->AssembleInterfaceGrad(*fe1, *fe2, *tr1, *tr2, el_x1, el_x2, elemmats);

            mats(0, 0)->AddSubMatrix(vdofs1, vdofs1, *elemmats(0, 0), skip_zeros);
            mats(0, 1)->AddSubMatrix(vdofs1, vdofs2, *elemmats(0, 1), skip_zeros);
            mats(1, 0)->AddSubMatrix(vdofs2, vdofs1, *elemmats(1, 0), skip_zeros);
            mats(1, 1)->AddSubMatrix(vdofs2, vdofs2, *elemmats(1, 1), skip_zeros);
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)

   DeletePointers(elemmats);
}

/*
   MixedInterfaceForm
*/

MixedInterfaceForm::MixedInterfaceForm(
   Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &trial_fes_, 
   Array<FiniteElementSpace *> &test_fes_, TopologyHandler *topol_)
   : meshes(meshes_), trial_fes(trial_fes_), test_fes(test_fes_), topol_handler(topol_), numSub(meshes_.Size())
{
   assert(trial_fes_.Size() == numSub);
   assert(test_fes_.Size() == numSub);

   trial_block_offsets.SetSize(numSub + 1);
   test_block_offsets.SetSize(numSub + 1);
   trial_block_offsets = 0;
   test_block_offsets = 0;
   for (int i = 1; i < numSub + 1; i++)
   {
      trial_block_offsets[i] = trial_fes[i-1]->GetTrueVSize();
      test_block_offsets[i] = test_fes[i-1]->GetTrueVSize();
   }
   trial_block_offsets.PartialSum();
   test_block_offsets.PartialSum();
}

MixedInterfaceForm::~MixedInterfaceForm()
{
   DeletePointers(fnfi);
}

void MixedInterfaceForm::AssembleInterfaceMatrices(Array2D<SparseMatrix *> &mats) const
{
   const PortInfo *pInfo;
   Array<int> midx(2);
   Array2D<SparseMatrix *> b_mats_p(2,2);
   Mesh *mesh1, *mesh2;
   FiniteElementSpace *trial_fes1, *trial_fes2, *test_fes1, *test_fes2;
   Array<InterfaceInfo>* interface_infos;

   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      pInfo = topol_handler->GetPortInfo(p);

      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;
      
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            b_mats_p(i, j) = mats(midx[i], midx[j]);

      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      trial_fes1 = trial_fes[midx[0]];
      trial_fes2 = trial_fes[midx[1]];
      test_fes1 = test_fes[midx[0]];
      test_fes2 = test_fes[midx[1]];

      interface_infos = topol_handler->GetInterfaceInfos(p);

      AssembleInterfaceMatrix(
         mesh1, mesh2, trial_fes1, trial_fes2, test_fes1, test_fes2, interface_infos, b_mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void MixedInterfaceForm::AssembleInterfaceMatrix(
   Mesh *mesh1, Mesh *mesh2, FiniteElementSpace *trial_fes1, FiniteElementSpace *trial_fes2,
   FiniteElementSpace *test_fes1, FiniteElementSpace *test_fes2, Array<InterfaceInfo> *interface_infos,
   Array2D<SparseMatrix*> &mats) const
{
   Array2D<DenseMatrix*> elemmats(2, 2);
   for (int i = 0; i < elemmats.NumRows(); i++)
      for (int j = 0; j < elemmats.NumCols(); j++) elemmats(i, j) = new DenseMatrix; 

   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;
   Array<Array<int> *> test_vdofs(2), trial_vdofs(2);
   trial_vdofs[0] = new Array<int>;
   trial_vdofs[1] = new Array<int>;
   test_vdofs[0] = new Array<int>;
   test_vdofs[1] = new Array<int>;

   InterfaceInfo *if_info;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      if_info = &((*interface_infos)[bn]);
      
      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         trial_fes1->GetElementVDofs(tr1->Elem1No, *trial_vdofs[0]);
         trial_fes2->GetElementVDofs(tr2->Elem1No, *trial_vdofs[1]);
         test_fes1->GetElementVDofs(tr1->Elem1No, *test_vdofs[0]);
         test_fes2->GetElementVDofs(tr2->Elem1No, *test_vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         trial_fe1 = trial_fes1->GetFE(tr1->Elem1No);
         trial_fe2 = trial_fes2->GetFE(tr2->Elem1No);
         test_fe1 = test_fes1->GetFE(tr1->Elem1No);
         test_fe2 = test_fes2->GetFE(tr2->Elem1No);

         for (int itg = 0; itg < fnfi.Size(); itg++)
         {
            assert(fnfi[itg]);

            fnfi[itg]->AssembleInterfaceMatrix(
               *trial_fe1, *trial_fe2, *test_fe1, *test_fe2, *tr1, *tr2, elemmats);

            for (int i = 0; i < 2; i++) {
               for (int j = 0; j < 2; j++) {
                  mats(i, j)->AddSubMatrix(*test_vdofs[i], *trial_vdofs[j], *elemmats(i,j), skip_zeros);
               }
            }
         }  // for (int itg = 0; itg < fnfi.Size(); itg++)
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)

   DeletePointers(test_vdofs);
   DeletePointers(trial_vdofs);
   DeletePointers(elemmats);
}

void MixedInterfaceForm::AssembleInterfaceMatrixAtPort(
   const int p, Array<FiniteElementSpace *> &trial_fes_comp, 
   Array<FiniteElementSpace *> &test_fes_comp, Array2D<SparseMatrix *> &mats_p) const
{
   const int num_ref_ports = topol_handler->GetNumRefPorts();

   assert(topol_handler->GetType() == TopologyHandlerMode::COMPONENT);
   assert((p >= 0) && (p < num_ref_ports));
   DeletePointers(mats_p);
   mats_p.SetSize(2, 2);

   const int num_comp = topol_handler->GetNumComponents();
   assert(trial_fes_comp.Size() == num_comp);
   assert(test_fes_comp.Size() == num_comp);

   int c1, c2;
   topol_handler->GetComponentPair(p, c1, c2);
   Mesh *comp1 = topol_handler->GetComponentMesh(c1);
   Mesh *comp2 = topol_handler->GetComponentMesh(c2);

   // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
   // Need to use two copied instances.
   Mesh mesh1(*comp1);
   Mesh mesh2(*comp2);

   Array<int> c_idx(2);
   c_idx[0] = c1; c_idx[1] = c2;

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         mats_p(i, j) = new SparseMatrix(test_fes_comp[c_idx[i]]->GetTrueVSize(), trial_fes_comp[c_idx[j]]->GetTrueVSize());

   Array<InterfaceInfo>* const if_infos = topol_handler->GetRefInterfaceInfos(p);

   // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
   // Need to use two copied instances.
   AssembleInterfaceMatrix(
      &mesh1, &mesh2, trial_fes_comp[c_idx[0]], trial_fes_comp[c_idx[1]],
      test_fes_comp[c_idx[0]], test_fes_comp[c_idx[1]], if_infos, mats_p);

   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
         mats_p(i, j)->Finalize();
}

}
