// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "rom_interfaceform.hpp"
#include "etc.hpp"
#include "utils/mpi_utils.h"  // this is from libROM/utils.
#include "linalg/NNLS.h"

using namespace std;

namespace mfem
{

ROMInterfaceForm::ROMInterfaceForm(
   Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_, Array<FiniteElementSpace *> &comp_fes_, TopologyHandler *topol_)
   : InterfaceForm(meshes_, fes_, topol_), numPorts(topol_->GetNumPorts()), numRefPorts(topol_->GetNumRefPorts()),
     num_comp(topol_->GetNumComponents()), comp_fes(comp_fes_)
{
   comp_basis.SetSize(num_comp);
   comp_basis = NULL;
   comp_basis_dof_offsets.SetSize(num_comp);

   basis.SetSize(numSub);
   basis = NULL;
   basis_dof_offsets.SetSize(numSub);

   // block_offsets should be updated according to the number of basis vectors.
   block_offsets = -1;
}

ROMInterfaceForm::~ROMInterfaceForm()
{
   DeletePointers(fnfi_ref_sample);
}

void ROMInterfaceForm::SetBasisAtComponent(const int c, DenseMatrix &basis_, const int offset)
{
   // assert(basis_.NumCols() == height);
   assert(basis_.NumRows() >= comp_fes[c]->GetTrueVSize() + offset);
   assert(basis.Size() == num_comp);

   comp_basis[c] = &basis_;
   comp_basis_dof_offsets[c] = offset;

   for (int m = 0; m < numSub; m++)
   {
      if (topol_handler->GetMeshType(m) != c)
         continue;
      
      basis[m] = &basis_;
      basis_dof_offsets[m] = offset;
   }
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
         if (!sample_info) continue;

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
         if (!sample_info) continue;

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

void ROMInterfaceForm::TrainEQPForRefPort(const int p,
   const CAROM::Matrix &snapshot1, const CAROM::Matrix &snapshot2,
   const Array2D<int> &snap_pair_idx, const double eqp_tol)
{
   int c1, c2, a1, a2;
   // TODO(kevin): at least component topology handler maintain attrs the same for both reference and subdomain.
   // Need to check submesh topology handler.
   topol_handler->GetRefPortInfo(p, c1, c2, a1, a2);
   Array<InterfaceInfo> *itf_info = topol_handler->GetRefInterfaceInfos(p);

   FiniteElementSpace *fes1 = comp_fes[c1];
   FiniteElementSpace *fes2 = comp_fes[c2];
   assert(fes1 && fes2);

   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider local matrix construction from local snapshot/basis matrix.
   */
   assert(snapshot1.distributed());
   assert(snapshot2.distributed());
   CAROM::Matrix snapshot1_work(snapshot1), snapshot2_work(snapshot2);
   snapshot1_work.gather();
   snapshot2_work.gather();
   assert(snapshot1_work.numRows() >= fes1->GetTrueVSize());
   assert(snapshot2_work.numRows() >= fes2->GetTrueVSize());

   DenseMatrix *basis1 = comp_basis[c1];
   DenseMatrix *basis2 = comp_basis[c2];
   const int basis1_offset = comp_basis_dof_offsets[c1];
   const int basis2_offset = comp_basis_dof_offsets[c2];
   assert(basis1 && basis2);

   // NOTE(kevin): these will be resized within the routines as needed.
   // just initializing with distribute option.
   CAROM::Matrix Gt(1,1, true);
   CAROM::Vector rhs_Gw(1, false);

   Array<int> el, qp;
   Array<double> qw;
   Array<int> fidxs;

   for (int it = 0; it < fnfi.Size(); it++)
   {
      const IntegrationRule *ir = fnfi[it]->GetIntegrationRule();
      assert(ir);
      const int nqe = ir->GetNPoints();

      SetupEQPSystem(snapshot1_work, snapshot2_work, snap_pair_idx,
                     *basis1, *basis2, basis1_offset, basis2_offset,
                     fes1, fes2, itf_info, fnfi[it], Gt, rhs_Gw);
      TrainEQPForIntegrator(nqe, Gt, rhs_Gw, eqp_tol, el, qp, qw);
      UpdateInterFaceIntegratorSampling(it, p, el, qp, qw);
   }
}

void ROMInterfaceForm::SetupEQPSystem(
   const CAROM::Matrix &snapshot1, const CAROM::Matrix &snapshot2,
   const Array2D<int> &snap_pair_idx,
   DenseMatrix &basis1, DenseMatrix &basis2,
   const int &basis1_offset, const int &basis2_offset,
   FiniteElementSpace *fes1, FiniteElementSpace *fes2,
   Array<InterfaceInfo>* const itf_infos,
   InterfaceNonlinearFormIntegrator* const nlfi,
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

   assert(!snapshot1.distributed());
   assert(!snapshot2.distributed());
   assert(fes1 && fes2);
   assert(snapshot1.numRows() >= fes1->GetTrueVSize());
   assert(snapshot2.numRows() >= fes2->GetTrueVSize());
   if ((snapshot1.numRows() > fes1->GetTrueVSize()) || (snapshot2.numRows() > fes2->GetTrueVSize()))
      mfem_warning("ROMNonlinearForm::SetupEQPSystem- "
         "snapshot vector has a larger dimension than finite element space vector dimension. "
         "Neglecting the rest of snapshot.\n");

   const IntegrationRule *ir = nlfi->GetIntegrationRule();
   Mesh *mesh1 = fes1->GetMesh();
   Mesh *mesh2 = fes2->GetMesh();

   const int vdim1 = fes1->GetVDim();
   const int vdim2 = fes2->GetVDim();
   const int nqe = ir->GetNPoints();
   const int NB1 = basis1.NumCols();
   const int NB2 = basis2.NumCols();
   const int NB = NB1 + NB2;

   /*
      We take the snapshot pairs given by snap1/2_idx.
      These two arrays should have the same size.
   */
   const int nsnap = snap_pair_idx.NumRows();
   assert(snap_pair_idx.NumCols() == 2);
   const int nsnap1 = snapshot1.numColumns();
   const int nsnap2 = snapshot2.numColumns();
   // assert((snap1_idx.Min() >= 0) && (snap1_idx.Max() < nsnap1));
   // assert((snap2_idx.Min() >= 0) && (snap2_idx.Max() < nsnap2));

   const int ne_global = itf_infos->Size();
   const int ne = CAROM::split_dimension(ne_global, MPI_COMM_WORLD);
   std::vector<int> elem_offsets;
   int dummy = CAROM::get_global_offsets(ne, elem_offsets, MPI_COMM_WORLD);
   assert(dummy == ne_global);

   const int NQ = ne * nqe;

   /*
      Compute G of size (NB * nsnap) x NQ, but only store its transpose Gt.
         For 0 <= j < NB, 0 <= i < nsnap, 0 <= e < ne, 0 <= m < nqe,
         G(j + (i*NB), (e*nqe) + m)
         is the coefficient of v_j^T M(p_i) V v_i at point m of element e,
         with respect to the integration rule weight at that point,
         where the "exact" quadrature solution is ir0->GetWeights().
   */
   Gt.setSize(NQ, NB * nsnap);
   assert(Gt.distributed());
   
   Vector v1_i(fes1->GetTrueVSize()), v2_i(fes2->GetTrueVSize());
   Vector r(nqe);

   Array<int> vdofs1, vdofs2;
   Vector el_x1, el_x2, el_tr1, el_tr2;
   DenseMatrix el_quad1, el_quad2;

   FaceElementTransformations *tr1, *tr2;
   const FiniteElement *fe1, *fe2;

   /* fill out quadrature evaluation of all snapshot-basis weak forms */
   for (int i = 0; i < nsnap; ++i)
   {
      // NOTE(kevin): have to copy the vector since libROM matrix is row-major.
      for (int k = 0; k < fes1->GetTrueVSize(); ++k)
         v1_i[k] = snapshot1(k, snap_pair_idx(i, 0));
      for (int k = 0; k < fes2->GetTrueVSize(); ++k)
         v2_i[k] = snapshot2(k, snap_pair_idx(i, 1));

      for (int e = elem_offsets[rank], eidx = 0; e < elem_offsets[rank+1]; e++, eidx++)
      {
         InterfaceInfo *if_info = &((*itf_infos)[e]);
         topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);
         assert((tr1 != NULL) && (tr2 != NULL));

         fes1->GetElementVDofs(tr1->Elem1No, vdofs1);
         fes2->GetElementVDofs(tr2->Elem1No, vdofs2);

         v1_i.GetSubVector(vdofs1, el_x1);
         v2_i.GetSubVector(vdofs2, el_x2);

         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         const int nd1 = fe1->GetDof();
         const int nd2 = fe2->GetDof();
         el_quad1.SetSize(nd1 * vdim1, nqe);
         el_quad2.SetSize(nd2 * vdim2, nqe);

         for (int p = 0; p < ir->GetNPoints(); p++)
         {
            Vector EQ1(el_quad1.GetColumn(p), nd1 * vdim1);
            Vector EQ2(el_quad2.GetColumn(p), nd2 * vdim2);

            const IntegrationPoint &ip = ir->IntPoint(p);
            nlfi->AssembleQuadratureVector(
               *fe1, *fe2, *tr1, *tr2, ip, 1.0, el_x1, el_x2, EQ1, EQ2);
         }

         /* two bases are independent, thus stored independently. */
         for (int j = 0; j < NB1; ++j)
         {
            Vector v1_j(basis1.GetColumn(j) + basis1_offset, fes1->GetVSize());
            v1_j.GetSubVector(vdofs1, el_tr1);

            el_quad1.MultTranspose(el_tr1, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (eidx * nqe), j + (i * NB)) = r[m];
         }  // for (int j = 0; j < NB1; ++j)
         for (int j = 0; j < NB2; ++j)
         {
            Vector v2_j(basis2.GetColumn(j) + basis2_offset, fes2->GetVSize());
            v2_j.GetSubVector(vdofs2, el_tr2);

            el_quad2.MultTranspose(el_tr2, r);

            for (int m = 0; m < nqe; ++m)
               Gt(m + (eidx * nqe), j + NB1 + (i * NB)) = r[m];
         }  // for (int j = 0; j < NB2; ++j)
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

void ROMInterfaceForm::TrainEQPForIntegrator(
   const int nqe, const CAROM::Matrix &Gt, const CAROM::Vector &rhs_Gw,
   const double eqp_tol, Array<int> &sample_el, Array<int> &sample_qp, Array<double> &sample_qw)
{
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

   sample_el.SetSize(0);
   sample_qp.SetSize(0);
   sample_qw.SetSize(0);
   for (int i = 0; i < eqpSol_global.dim(); ++i)
   {
      if (eqpSol_global(i) > 1.0e-12)
      {
         const int e = i / nqe;  // Element index
         sample_el.Append(i / nqe);
         sample_qp.Append(i % nqe);
         sample_qw.Append(eqpSol_global(i));
      }
   }
   printf("Size of sampled qp: %d\n", sample_el.Size());
   if (nnz != sample_el.Size())
      printf("Sample quadrature points with weight < 1.0e-12 are neglected.\n");
}

void ROMInterfaceForm::SaveEQPForIntegrator(const int k, hid_t file_id, const std::string &dsetname)
{
   assert(file_id >= 0);
   hid_t grp_id;
   herr_t errf;

   grp_id = H5Gcreate(file_id, dsetname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   Array<SampleInfo> *ref_sample = NULL;
   int c1, c2, a1, a2;
   std::string port_dset;
   for (int p = 0; p < numRefPorts; p++)
   {
      topol_handler->GetRefPortInfo(p, c1, c2, a1, a2);
      port_dset = topol_handler->GetComponentName(c1) + ":" + topol_handler->GetComponentName(c2);
      port_dset += "-" + std::to_string(a1) + ":" + std::to_string(a2);
      ref_sample = fnfi_ref_sample[p + k * numRefPorts];

      hdf5_utils::WriteDataset(grp_id, port_dset, IntegratorType::INTERFACE, *ref_sample);
   }

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

}
