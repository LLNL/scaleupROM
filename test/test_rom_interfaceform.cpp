// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "interfaceinteg.hpp"
#include "rom_interfaceform.hpp"
#include "etc.hpp"
#include "component_topology_handler.hpp"

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-13;
static const double grad_thre = 1.0e-7;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(ROMInterfaceForm, InterfaceAddMult)
{
   config = InputParser("inputs/dd_mms.yml");
   const int order = UniformRandom(1, 3);

   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler();
   assert(submesh->GetNumSubdomains() > 1);

   Array<Mesh *> meshes;
   TopologyData topol_data;
   submesh->ExportInfo(meshes, topol_data);
   const int dim = topol_data.dim;
   const int numSub = topol_data.numSub;

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));

   Array<FiniteElementSpace *> fes(numSub);
   for (int m = 0; m < numSub; m++)
      fes[m] = new FiniteElementSpace(meshes[m], dg_coll, dim);

   /* Fictitious bases */
   const int num_basis = 10;
   Array<DenseMatrix *> basis(numSub);
   for (int m = 0; m < numSub; m++)
   {
      const int ndofs = fes[m]->GetTrueVSize();
      basis[m] = new DenseMatrix(ndofs, num_basis);
      for (int i = 0; i < ndofs; i++)
         for (int j = 0; j < num_basis; j++)
            (*basis[m])(i, j) = UniformRandom();
   }

   Array<int> block_offsets(numSub+1), rom_block_offsets(numSub+1);
   block_offsets[0] = 0;
   for (int m = 0; m < numSub; m++)
      block_offsets[m+1] = fes[m]->GetTrueVSize();
   block_offsets.PartialSum();

   rom_block_offsets = num_basis;
   rom_block_offsets[0] = 0;
   rom_block_offsets.PartialSum();

   const IntegrationRule *ir = NULL;
   for (int be = 0; be < fes[0]->GetNBE(); be++)
   {
      FaceElementTransformations *tr = meshes[0]->GetBdrFaceTransformations(be);
      if (tr != NULL)
      {
         ir = &IntRules.Get(tr->GetGeometryType(),
                            (int)(ceil(1.5 * (2 * fes[0]->GetMaxElementOrder() - 1))));
         break;
      }
   }
   assert(ir);

   ConstantCoefficient pi(3.141592);
   auto *integ1 = new DGLaxFriedrichsFluxIntegrator(pi);
   auto *integ2 = new DGLaxFriedrichsFluxIntegrator(pi);
   integ1->SetIntRule(ir);
   integ2->SetIntRule(ir);

   InterfaceForm *nform = new InterfaceForm(meshes, fes, submesh);
   nform->AddInterfaceIntegrator(integ1);

   BlockVector rom_u(rom_block_offsets), u(block_offsets);
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   for (int m = 0; m < numSub; m++)
      basis[m]->Mult(rom_u.GetBlock(m), u.GetBlock(m));

   BlockVector rom_y(rom_block_offsets), y(block_offsets), Pty(rom_block_offsets);

   y = 0.0;
   nform->InterfaceAddMult(u, y);
   for (int m = 0; m < numSub; m++)
      basis[m]->MultTranspose(y.GetBlock(m), Pty.GetBlock(m));

   ROMInterfaceForm *rform = new ROMInterfaceForm(meshes, fes, fes, submesh);
   rform->AddInterfaceIntegrator(integ2);
   /* For submesh, mesh index is the same as component index */
   for (int m = 0; m < numSub; m++)
      rform->SetBasisAtComponent(m, *basis[m]);
   rform->UpdateBlockOffsets();

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nport = submesh->GetNumPorts();
   const int nqe = ir->GetNPoints();
   Array<double> const& w_el = ir->GetWeights();
   for (int p = 0; p < nport; p++)
   {
      Array<InterfaceInfo> *interface_infos = submesh->GetInterfaceInfos(p);
      Array<SampleInfo> samples(0);
      for (int itf = 0; itf < interface_infos->Size(); itf++)
         for (int q = 0; q < nqe; q++)
         {
            samples.Append({.el=itf, .qp=q, .qw=w_el[q]});
         }

      rform->UpdateInterFaceIntegratorSampling(0, p, samples);
   }
   
   rom_y = 0.0; 
   rform->InterfaceAddMult(rom_u, rom_y);

   for (int k = 0; k < rom_y.Size(); k++)
      EXPECT_NEAR(rom_y(k), Pty(k), threshold);
   
   delete nform;
   delete rform;
   delete dg_coll;
   DeletePointers(fes);
   DeletePointers(basis);
   delete submesh;
}

TEST(ROMInterfaceForm, InterfaceGetGradient)
{
   config = InputParser("inputs/dd_mms.yml");
   const int order = UniformRandom(1, 3);

   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler();
   assert(submesh->GetNumSubdomains() > 1);

   Array<Mesh *> meshes;
   TopologyData topol_data;
   submesh->ExportInfo(meshes, topol_data);
   const int dim = topol_data.dim;
   const int numSub = topol_data.numSub;

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));

   Array<FiniteElementSpace *> fes(numSub);
   for (int m = 0; m < numSub; m++)
      fes[m] = new FiniteElementSpace(meshes[m], dg_coll, dim);

   /* Fictitious bases */
   const int num_basis = 10;
   Array<DenseMatrix *> basis(numSub);
   for (int m = 0; m < numSub; m++)
   {
      const int ndofs = fes[m]->GetTrueVSize();
      basis[m] = new DenseMatrix(ndofs, num_basis);
      for (int i = 0; i < ndofs; i++)
         for (int j = 0; j < num_basis; j++)
            (*basis[m])(i, j) = UniformRandom();
   }

   Array<int> block_offsets(numSub+1), rom_block_offsets(numSub+1);
   block_offsets[0] = 0;
   for (int m = 0; m < numSub; m++)
      block_offsets[m+1] = fes[m]->GetTrueVSize();
   block_offsets.PartialSum();

   rom_block_offsets = num_basis;
   rom_block_offsets[0] = 0;
   rom_block_offsets.PartialSum();

   const IntegrationRule *ir = NULL;
   for (int be = 0; be < fes[0]->GetNBE(); be++)
   {
      FaceElementTransformations *tr = meshes[0]->GetBdrFaceTransformations(be);
      if (tr != NULL)
      {
         ir = &IntRules.Get(tr->GetGeometryType(),
                            (int)(ceil(1.5 * (2 * fes[0]->GetMaxElementOrder() - 1))));
         break;
      }
   }
   assert(ir);

   ConstantCoefficient pi(3.141592);
   auto *integ2 = new DGLaxFriedrichsFluxIntegrator(pi);
   integ2->SetIntRule(ir);

   BlockVector rom_u(rom_block_offsets);
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   BlockVector rom_y(rom_block_offsets);

   ROMInterfaceForm *rform = new ROMInterfaceForm(meshes, fes, fes, submesh);
   rform->AddInterfaceIntegrator(integ2);
   /* For submesh, mesh index is the same as component index */
   for (int m = 0; m < numSub; m++)
      rform->SetBasisAtComponent(m, *basis[m]);
   rform->UpdateBlockOffsets();

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nport = submesh->GetNumPorts();
   const int nqe = ir->GetNPoints();
   Array<double> const& w_el = ir->GetWeights();
   for (int p = 0; p < nport; p++)
   {
      Array<InterfaceInfo> *interface_infos = submesh->GetInterfaceInfos(p);
      Array<SampleInfo> samples(0);
      for (int itf = 0; itf < interface_infos->Size(); itf++)
         for (int q = 0; q < nqe; q++)
         {
            samples.Append({.el=itf, .qp=q, .qw=w_el[q]});
         }

      rform->UpdateInterFaceIntegratorSampling(0, p, samples);
   }
   
   rom_y = 0.0; 
   rform->InterfaceAddMult(rom_u, rom_y);

   Array2D<SparseMatrix *> jac_mats(numSub, numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         jac_mats(i, j) = new SparseMatrix(num_basis);

   rform->InterfaceGetGradient(rom_u, jac_mats);

   BlockMatrix jac(rom_block_offsets);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         jac.SetBlock(i, j, jac_mats(i, j));

   double J0 = 0.5 * (rom_y * rom_y);
   BlockVector grad(rom_block_offsets);
   jac.MultTranspose(rom_y, grad);
   double gg = sqrt(grad * grad);
   printf("J0: %.15E\n", J0);
   printf("grad: %.15E\n", gg);

   BlockVector du(grad);
   du /= gg;

   BlockVector rom_y1(rom_block_offsets);
   const int Nk = 40;
   double error1 = 1e100, error;
   printf("amp\tJ1\tdJdx\terror\n");
   for (int k = 0; k < Nk; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      BlockVector rom_u1(rom_u);
      rom_u1.Add(amp, du);

      rom_y1 = 0.0;
      rform->InterfaceAddMult(rom_u1, rom_y1);
      double J1 = 0.5 * (rom_y1 * rom_y1);
      double dJdx = (J1 - J0) / amp;
      error = abs(dJdx - gg) / gg;

      printf("%.5E\t%.5E\t%.5E\t%.5E\n", amp, J1, dJdx, error);
      if (error >= error1)
         break;
      error1 = error;
   }
   EXPECT_TRUE(min(error, error1) < grad_thre);
   
   delete rform;
   delete dg_coll;
   DeletePointers(fes);
   DeletePointers(basis);
   DeletePointers(jac_mats);
   delete submesh;
}

TEST(ROMInterfaceForm, SetupEQPSystem_for_a_port)
{
   config = InputParser("inputs/dd_mms.yml");
   const int order = UniformRandom(1, 3);

   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler();
   assert(submesh->GetNumSubdomains() > 1);

   Array<Mesh *> meshes;
   TopologyData topol_data;
   submesh->ExportInfo(meshes, topol_data);
   const int dim = topol_data.dim;
   const int numSub = topol_data.numSub;

   const int pidx = UniformRandom(0, submesh->GetNumPorts()-1);
   const PortInfo *pInfo = submesh->GetPortInfo(pidx);
   Array<InterfaceInfo>* const itf_infos = submesh->GetInterfaceInfos(pidx);

   Array<int> midx(2);
   Mesh *mesh1, *mesh2;

   midx[0] = pInfo->Mesh1;
   midx[1] = pInfo->Mesh2;

   mesh1 = meshes[midx[0]];
   mesh2 = meshes[midx[1]];
   
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   Array<FiniteElementSpace *> fes(numSub);
   for (int m = 0; m < numSub; m++)
      fes[m] = new FiniteElementSpace(meshes[m], dg_coll, dim);

   FiniteElementSpace *fes1, *fes2;
   fes1 = fes[midx[0]];
   fes2 = fes[midx[1]];

   const int ndofs1 = fes1->GetTrueVSize();
   const int ndofs2 = fes2->GetTrueVSize();

   /* number of snapshots on each domain */
   const int nsnap1 = UniformRandom(3, 5);
   const int nsnap2 = UniformRandom(3, 5);
   /* same number of basis to fully represent snapshots */
   const int num_basis1 = nsnap1;
   const int num_basis2 = nsnap2;
   const int NB = num_basis1 + num_basis2;
   /* number of basis for other domains that do not belong to the port */
   const int nb_def = min(num_basis1, num_basis2);
   /* number of snapshot pairs for port EQP training */
   const int num_snap = UniformRandom(3, 5);

   /* fictitious snapshots */
   DenseMatrix snapshots1(ndofs1, nsnap1);
   DenseMatrix snapshots2(ndofs2, nsnap2);
   for (int i = 0; i < ndofs1; i++)
      for (int j = 0; j < nsnap1; j++)
         snapshots1(i, j) = 2.0 * UniformRandom() - 1.0;
   for (int i = 0; i < ndofs2; i++)
      for (int j = 0; j < nsnap2; j++)
         snapshots2(i, j) = 2.0 * UniformRandom() - 1.0;

   /* fictitious pair indices */
   Array2D<int> snap_pair_idx(num_snap, 2);
   for (int i = 0; i < num_snap; i++)
   {
      snap_pair_idx(i, 0) = UniformRandom(0, nsnap1-1);
      snap_pair_idx(i, 1) = UniformRandom(0, nsnap2-1);
   }

   /* bases generated from fictitious snapshots */
   DenseMatrix basis1(ndofs1, num_basis1);
   DenseMatrix basis2(ndofs2, num_basis2);
   const CAROM::Matrix *carom_snapshots1, *carom_snapshots2;
   CAROM::BasisGenerator *basis1_generator, *basis2_generator;
   {
      CAROM::Options options(ndofs1, nsnap1, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis1_generator = new CAROM::BasisGenerator(options, false, "test_basis1");
      Vector snapshot(ndofs1);
      for (int s = 0; s < nsnap1; s++)
      {
         snapshots1.GetColumnReference(s, snapshot);
         basis1_generator->takeSample(snapshot.GetData());
      }
      basis1_generator->endSamples();
      carom_snapshots1 = basis1_generator->getSnapshotMatrix();

      CAROM::BasisReader basis_reader("test_basis1");
      const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis1);
      CAROM::CopyMatrix(*carom_basis, basis1);
   }
   {
      CAROM::Options options(ndofs2, nsnap2, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis2_generator = new CAROM::BasisGenerator(options, false, "test_basis2");
      Vector snapshot(ndofs2);
      for (int s = 0; s < nsnap2; s++)
      {
         snapshots2.GetColumnReference(s, snapshot);
         basis2_generator->takeSample(snapshot.GetData());
      }
      basis2_generator->endSamples();
      carom_snapshots2 = basis2_generator->getSnapshotMatrix();

      CAROM::BasisReader basis_reader("test_basis2");
      const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis2);
      CAROM::CopyMatrix(*carom_basis, basis2);
   }

   /* get integration rule for eqp */
   const IntegrationRule *ir = NULL;
   for (int be = 0; be < fes1->GetNBE(); be++)
   {
      FaceElementTransformations *tr = mesh1->GetBdrFaceTransformations(be);
      if (tr != NULL)
      {
         ir = &IntRules.Get(tr->GetGeometryType(),
                            (int)(ceil(1.5 * (2 * fes1->GetMaxElementOrder() - 1))));
         break;
      }
   }
   assert(ir);

   ConstantCoefficient pi(3.141592);
   auto *integ1 = new DGLaxFriedrichsFluxIntegrator(pi);
   integ1->SetIntRule(ir);
   auto *integ2 = new DGLaxFriedrichsFluxIntegrator(pi);
   integ2->SetIntRule(ir);

   /* FOM interface operator */
   InterfaceForm *nform(new InterfaceForm(meshes, fes, submesh));
   nform->AddInterfaceIntegrator(integ1);

   /* EQP interface operator */
   ROMInterfaceForm *rform(new ROMInterfaceForm(meshes, fes, fes, submesh));
   rform->AddInterfaceIntegrator(integ2);

   Array<DenseMatrix *> bases(numSub);
   /* For submesh, mesh index is the same as component index */
   for (int m = 0; m < bases.Size(); m++)
   {
      bases[m] = new DenseMatrix(fes[m]->GetTrueVSize(), nb_def);
      *bases[m] = 0.0;
      rform->SetBasisAtComponent(m, *bases[m]);
   }
   rform->SetBasisAtComponent(midx[0], basis1);
   rform->SetBasisAtComponent(midx[1], basis2);
   rform->UpdateBlockOffsets();
   rform->SetPrecomputeMode(false);

   CAROM::Vector rhs1(num_snap * NB, false);
   CAROM::Vector rhs2(num_snap * NB, false);
   CAROM::Matrix Gt(1, 1, true);

   /* exact right-hand side by inner product of basis and fom vectors */
   Vector rhs_vec1(ndofs1), rhs_vec2(ndofs2);
   Vector snap1(ndofs1), snap2(ndofs2);
   Vector basis_col;
   for (int s = 0; s < num_snap; s++)
   {
      snapshots1.GetColumnReference(snap_pair_idx(s, 0), snap1);
      snapshots2.GetColumnReference(snap_pair_idx(s, 1), snap2);

      rhs_vec1 = 0.0; rhs_vec2 = 0.0;
      nform->AssembleInterfaceVector(mesh1, mesh2, fes1, fes2, itf_infos, snap1, snap2, rhs_vec1, rhs_vec2);

      for (int b = 0; b < num_basis1; b++)
      {
         basis1.GetColumnReference(b, basis_col);
         rhs1(b + s * NB) = basis_col * rhs_vec1;
      }
      for (int b = 0; b < num_basis2; b++)
      {
         basis2.GetColumnReference(b, basis_col);
         rhs1(b + num_basis1 + s * NB) = basis_col * rhs_vec2;
      }
   }

   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Will have to consider parallel-compatible test.
   */
   CAROM::Matrix carom_snapshots1_work(*carom_snapshots1);
   carom_snapshots1_work.gather();
   CAROM::Matrix carom_snapshots2_work(*carom_snapshots2);
   carom_snapshots2_work.gather();

   /* equivalent operation must happen within this routine */
   rform->SetupEQPSystem(carom_snapshots1_work, carom_snapshots2_work,
                         snap_pair_idx, basis1, basis2, 0, 0,
                         fes1, fes2, itf_infos, integ2, Gt, rhs2);

   for (int k = 0; k < rhs1.dim(); k++)
      EXPECT_NEAR(rhs1(k), rhs2(k), threshold);

   double eqp_tol = 1.0e-10;
   Array<SampleInfo> samples(0);
   const int nqe = ir->GetNPoints();
   rform->TrainEQPForIntegrator(nqe, Gt, rhs2, eqp_tol, samples);
   // if (rform->PrecomputeMode()) rform->PrecomputeCoefficients();
   rform->UpdateInterFaceIntegratorSampling(0, pidx, samples);

   DenseMatrix rom_rhs1(rhs1.getData(), NB, num_snap), rom_rhs2(NB, num_snap);
   Array<int> rom_blocks = rform->GetBlockOffsets();
   BlockVector rom_sol(rom_blocks), rom_rhs2_vec(rom_blocks);
   for (int s = 0; s < num_snap; s++)
   {
      snapshots1.GetColumnReference(snap_pair_idx(s, 0), snap1);
      snapshots2.GetColumnReference(snap_pair_idx(s, 1), snap2);

      rom_sol = 0.0;
      basis1.MultTranspose(snap1, rom_sol.GetBlock(midx[0]));
      basis2.MultTranspose(snap2, rom_sol.GetBlock(midx[1]));
      rom_rhs2_vec = 0.0;
      rform->InterfaceAddMult(rom_sol, rom_rhs2_vec);

      for (int b = 0; b < num_basis1; b++)
         rom_rhs2(b, s) = rom_rhs2_vec.GetBlock(midx[0])(b);
      for (int b = 0; b < num_basis2; b++)
         rom_rhs2(b + num_basis1, s) = rom_rhs2_vec.GetBlock(midx[1])(b);
   }

   for (int i = 0; i < NB; i++)
      for (int j = 0; j < num_snap; j++)
         EXPECT_NEAR(rom_rhs1(i, j), rom_rhs2(i, j), threshold);

   delete submesh;
   delete dg_coll;
   delete nform;
   delete rform;
   DeletePointers(bases);
   delete basis1_generator;
   delete basis2_generator;
}

TEST(ROMInterfaceForm, Precompute)
{
   Mesh *pmesh = new Mesh("meshes/test.2x1.mesh");
   TopologyHandler *topol = new SubMeshTopologyHandler(pmesh);

   const int order = UniformRandom(1, 3);

   Array<Mesh *> meshes;
   TopologyData topol_data;
   topol->ExportInfo(meshes, topol_data);
   const int dim = topol_data.dim;
   const int numSub = topol_data.numSub;
   assert(numSub == 2);
   assert(topol->GetNumRefPorts() == 1);

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   Array<FiniteElementSpace *> fes(numSub);
   for (int m = 0; m < numSub; m++)
      fes[m] = new FiniteElementSpace(meshes[m], dg_coll, dim);

   const int ndofs1 = fes[0]->GetTrueVSize();
   const int ndofs2 = fes[1]->GetTrueVSize();

   /* number of snapshots on each domain */
   const int nsnap1 = UniformRandom(3, 5);
   const int nsnap2 = UniformRandom(3, 5);
   /* same number of basis to fully represent snapshots */
   const int num_basis1 = nsnap1;
   const int num_basis2 = nsnap2;
   const int rom_dim = num_basis1 + num_basis2;
   /* number of snapshot pairs for port EQP training */
   const int num_snap = UniformRandom(3, 5);

   /* fictitious snapshots */
   DenseMatrix snapshots1(ndofs1, nsnap1);
   DenseMatrix snapshots2(ndofs2, nsnap2);
   for (int i = 0; i < ndofs1; i++)
      for (int j = 0; j < nsnap1; j++)
         snapshots1(i, j) = 2.0 * UniformRandom() - 1.0;
   for (int i = 0; i < ndofs2; i++)
      for (int j = 0; j < nsnap2; j++)
         snapshots2(i, j) = 2.0 * UniformRandom() - 1.0;

   /* fictitious pair indices */
   Array2D<int> snap_pair_idx(num_snap, 2);
   for (int i = 0; i < num_snap; i++)
   {
      snap_pair_idx(i, 0) = UniformRandom(0, nsnap1-1);
      snap_pair_idx(i, 1) = UniformRandom(0, nsnap2-1);
   }

   /* bases generated from fictitious snapshots */
   DenseMatrix basis1(ndofs1, num_basis1);
   DenseMatrix basis2(ndofs2, num_basis2);
   const CAROM::Matrix *carom_snapshots1, *carom_snapshots2;
   CAROM::BasisGenerator *basis1_generator, *basis2_generator;
   {
      CAROM::Options options(ndofs1, nsnap1, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis1_generator = new CAROM::BasisGenerator(options, false, "test_basis1");
      Vector snapshot(ndofs1);
      for (int s = 0; s < nsnap1; s++)
      {
         snapshots1.GetColumnReference(s, snapshot);
         basis1_generator->takeSample(snapshot.GetData());
      }
      basis1_generator->endSamples();
      carom_snapshots1 = basis1_generator->getSnapshotMatrix();

      CAROM::BasisReader basis_reader("test_basis1");
      const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis1);
      CAROM::CopyMatrix(*carom_basis, basis1);
   }
   {
      CAROM::Options options(ndofs2, nsnap2, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis2_generator = new CAROM::BasisGenerator(options, false, "test_basis2");
      Vector snapshot(ndofs2);
      for (int s = 0; s < nsnap2; s++)
      {
         snapshots2.GetColumnReference(s, snapshot);
         basis2_generator->takeSample(snapshot.GetData());
      }
      basis2_generator->endSamples();
      carom_snapshots2 = basis2_generator->getSnapshotMatrix();

      CAROM::BasisReader basis_reader("test_basis2");
      const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis2);
      CAROM::CopyMatrix(*carom_basis, basis2);
   }

   /* get integration rule for eqp */
   const IntegrationRule *ir = NULL;
   for (int be = 0; be < fes[0]->GetNBE(); be++)
   {
      FaceElementTransformations *tr = meshes[0]->GetBdrFaceTransformations(be);
      if (tr != NULL)
      {
         ir = &IntRules.Get(tr->GetGeometryType(),
                            (int)(ceil(1.5 * (2 * fes[0]->GetMaxElementOrder() - 1))));
         break;
      }
   }
   assert(ir);

   ConstantCoefficient pi(3.141592);
   auto *integ1 = new DGLaxFriedrichsFluxIntegrator(pi);
   integ1->SetIntRule(ir);

   /* EQP interface operator */
   ROMInterfaceForm *rform(new ROMInterfaceForm(meshes, fes, fes, topol));
   rform->AddInterfaceIntegrator(integ1);

   /* For submesh, mesh index is the same as component index */
   rform->SetBasisAtComponent(0, basis1);
   rform->SetBasisAtComponent(1, basis2);
   rform->UpdateBlockOffsets();

   rform->TrainEQPForRefPort(0, *carom_snapshots1, *carom_snapshots2, snap_pair_idx, 1.0e-11);
   rform->PrecomputeCoefficients();
   rform->SetPrecomputeMode(false);

   Vector rom_u(rom_dim);
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   Vector rom_y(rom_dim), rom_yfast(rom_dim);
   rom_y = 0.0;
   rform->InterfaceAddMult(rom_u, rom_y);

   rform->SetPrecomputeMode(true);
   rom_yfast = 0.0;
   rform->InterfaceAddMult(rom_u, rom_yfast);

   /* compare precompute vs non-precompute */
   for (int k = 0; k < rom_y.Size(); k++)
      EXPECT_NEAR(rom_y(k), rom_yfast(k), threshold);

   /* compute precomputed gradient */
   Array2D<SparseMatrix *> mats(numSub, numSub);
   Array<int> rom_block(numSub);
   rom_block[0] = num_basis1;
   rom_block[1] = num_basis2;
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         mats(i, j) = new SparseMatrix(rom_block[i], rom_block[j]);

   rform->InterfaceGetGradient(rom_u, mats);
   
   rom_block.SetSize(numSub+1);
   rom_block[0] = 0;
   rom_block[1] = num_basis1;
   rom_block[2] = num_basis2;
   rom_block.PartialSum();

   BlockMatrix jac(rom_block);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         jac.SetBlock(i, j, mats(i, j));

   double J0 = 0.5 * (rom_y * rom_y);
   Vector grad(rom_dim);
   jac.MultTranspose(rom_y, grad);
   double gg = sqrt(grad * grad);
   printf("J0: %.15E\n", J0);
   printf("grad: %.15E\n", gg);

   Vector du(grad);
   du /= gg;

   Vector rom_y1(rom_dim);
   const int Nk = 40;
   double error1 = 1e100, error;
   printf("amp\tJ1\tdJdx\terror\n");
   for (int k = 0; k < Nk; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      Vector rom_u1(rom_u);
      rom_u1.Add(amp, du);

      rom_y1 = 0.0;
      rform->InterfaceAddMult(rom_u1, rom_y1);
      double J1 = 0.5 * (rom_y1 * rom_y1);
      double dJdx = (J1 - J0) / amp;
      error = abs(dJdx - gg) / gg;

      printf("%.5E\t%.5E\t%.5E\t%.5E\n", amp, J1, dJdx, error);
      if (error >= error1)
         break;
      error1 = error;
   }
   EXPECT_TRUE(min(error, error1) < grad_thre);

   DeletePointers(mats);
   delete rform;
   DeletePointers(fes);
   delete dg_coll;
   delete topol;
}

// // TODO(kevin): figure out how to set up interface for self.
// TEST(ROMInterfaceForm, Self_Interface)
// {
//    config = InputParser("inputs/test.interface.yml");
//    const int order = UniformRandom(1, 1);

//    ComponentTopologyHandler *topol = new ComponentTopologyHandler();

//    Array<Mesh *> meshes;
//    TopologyData topol_data;
//    topol->ExportInfo(meshes, topol_data);
//    const int dim = topol_data.dim;
//    const int numSub = topol_data.numSub;

//    const int pidx = UniformRandom(0, topol->GetNumPorts()-1);
//    const PortInfo *pInfo = topol->GetPortInfo(pidx);
//    Array<InterfaceInfo>* const itf_infos = topol->GetInterfaceInfos(pidx);

//    Array<int> midx(2);
//    Mesh *mesh1, *mesh2;

//    midx[0] = pInfo->Mesh1;
//    midx[1] = pInfo->Mesh2;
//    assert(midx[0] == midx[1]);

//    mesh1 = meshes[midx[0]];
//    mesh2 = meshes[midx[1]];
   
//    FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
//    Array<FiniteElementSpace *> fes(numSub);
//    for (int m = 0; m < numSub; m++)
//       fes[m] = new FiniteElementSpace(meshes[m], dg_coll, dim);

//    FiniteElementSpace *fes1, *fes2;
//    fes1 = fes[midx[0]];
//    fes2 = fes[midx[1]];

//    const int ndofs1 = fes1->GetTrueVSize();
//    const int ndofs2 = fes2->GetTrueVSize();
//    assert(ndofs1 == ndofs2);

//    /* number of snapshots on the domain */
//    /* we only have only one domain */
//    const int nsnap = UniformRandom(3, 5);
//    /* same number of basis to fully represent snapshots */
//    const int num_basis = nsnap;
//    const int NB = num_basis + num_basis;
//    const int num_snap = nsnap;

//    /* fictitious snapshots */
//    DenseMatrix snapshots(ndofs1, nsnap);
//    for (int i = 0; i < ndofs1; i++)
//       for (int j = 0; j < nsnap; j++)
//          snapshots(i, j) = 2.0 * UniformRandom() - 1.0;

//    /* fictitious pair indices */
//    Array2D<int> snap_pair_idx(num_snap, 2);
//    for (int i = 0; i < num_snap; i++)
//    {
//       snap_pair_idx(i, 0) = i;
//       snap_pair_idx(i, 1) = i;
//    }

//    /* bases generated from fictitious snapshots */
//    DenseMatrix basis(ndofs1, num_basis);
//    const CAROM::Matrix *carom_snapshots;
//    CAROM::BasisGenerator *basis_generator;
//    {
//       CAROM::Options options(ndofs1, nsnap, 1, true);
//       options.static_svd_preserve_snapshot = true;
//       basis_generator = new CAROM::BasisGenerator(options, false, "test_basis");
//       Vector snapshot(ndofs1);
//       for (int s = 0; s < nsnap; s++)
//       {
//          snapshots.GetColumnReference(s, snapshot);
//          basis_generator->takeSample(snapshot.GetData());
//       }
//       basis_generator->endSamples();
//       carom_snapshots = basis_generator->getSnapshotMatrix();

//       CAROM::BasisReader basis_reader("test_basis");
//       const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis);
//       CAROM::CopyMatrix(*carom_basis, basis);
//    }

//    /* get integration rule for eqp */
//    const IntegrationRule *ir = NULL;
//    for (int be = 0; be < fes1->GetNBE(); be++)
//    {
//       FaceElementTransformations *tr = mesh1->GetBdrFaceTransformations(be);
//       if (tr != NULL)
//       {
//          ir = &IntRules.Get(tr->GetGeometryType(),
//                             (int)(ceil(1.5 * (2 * fes1->GetMaxElementOrder() - 1))));
//          break;
//       }
//    }
//    assert(ir);

//    ConstantCoefficient pi(3.141592);
//    auto *integ1 = new DGLaxFriedrichsFluxIntegrator(pi);
//    integ1->SetIntRule(ir);
//    auto *integ2 = new DGLaxFriedrichsFluxIntegrator(pi);
//    integ2->SetIntRule(ir);

//    /* FOM interface operator */
//    InterfaceForm *nform(new InterfaceForm(meshes, fes, topol));
//    nform->AddInterfaceIntegrator(integ1);

//    /* EQP interface operator */
//    ROMInterfaceForm *rform(new ROMInterfaceForm(meshes, fes, fes, topol));
//    rform->AddInterfaceIntegrator(integ2);

//    rform->SetBasisAtComponent(midx[0], basis);
//    rform->UpdateBlockOffsets();
//    // rform->SetPrecomputeMode(false);

//    CAROM::Vector rhs1(num_snap * NB, false);
//    CAROM::Vector rhs2(num_snap * NB, false);
//    CAROM::Matrix Gt(1, 1, true);

//    /* exact right-hand side by inner product of basis and fom vectors */
//    Vector rhs_vec1(ndofs1), rhs_vec2(ndofs2);
//    Vector snap1(ndofs1), snap2(ndofs2);
//    Vector basis_col;
//    for (int s = 0; s < num_snap; s++)
//    {
//       snapshots.GetColumnReference(snap_pair_idx(s, 0), snap1);
//       snapshots.GetColumnReference(snap_pair_idx(s, 1), snap2);

//       rhs_vec1 = 0.0; rhs_vec2 = 0.0;
//       nform->AssembleInterfaceVector(mesh1, mesh2, fes1, fes2, itf_infos, snap1, snap2, rhs_vec1, rhs_vec2);

//       for (int b = 0; b < num_basis; b++)
//       {
//          basis.GetColumnReference(b, basis_col);
//          rhs1(b + s * NB) = basis_col * rhs_vec1;
//       }
//       for (int b = 0; b < num_basis; b++)
//       {
//          basis.GetColumnReference(b, basis_col);
//          rhs1(b + num_basis + s * NB) = basis_col * rhs_vec2;
//       }
//    }

//    /*
//       TODO(kevin): this is a boilerplate for parallel POD/EQP training.
//       Will have to consider parallel-compatible test.
//    */
//    CAROM::Matrix carom_snapshots1_work(*carom_snapshots);
//    carom_snapshots1_work.gather();
//    CAROM::Matrix carom_snapshots2_work(*carom_snapshots);
//    carom_snapshots2_work.gather();

//    /* equivalent operation must happen within this routine */
//    rform->SetupEQPSystem(carom_snapshots1_work, carom_snapshots2_work,
//                          snap_pair_idx, basis, basis, 0, 0,
//                          fes1, fes2, itf_infos, integ2, Gt, rhs2);

//    for (int k = 0; k < rhs1.dim(); k++)
//       EXPECT_NEAR(rhs1(k), rhs2(k), threshold);

//    double eqp_tol = 1.0e-10;
//    Array<SampleInfo> samples(0);
//    const int nqe = ir->GetNPoints();
//    rform->TrainEQPForIntegrator(nqe, Gt, rhs2, eqp_tol, samples);
//    // if (rform->PrecomputeMode()) rform->PrecomputeCoefficients();
//    rform->UpdateInterFaceIntegratorSampling(0, pidx, samples);

//    DenseMatrix rom_rhs1(rhs1.getData(), NB, num_snap), rom_rhs2(NB, num_snap);
//    Array<int> rom_blocks = rform->GetBlockOffsets();
//    BlockVector rom_sol(rom_blocks), rom_rhs2_vec(rom_blocks);
//    for (int s = 0; s < num_snap; s++)
//    {
//       snapshots.GetColumnReference(snap_pair_idx(s, 0), snap1);
//       snapshots.GetColumnReference(snap_pair_idx(s, 1), snap2);

//       rom_sol = 0.0;
//       basis.MultTranspose(snap1, rom_sol.GetBlock(midx[0]));
//       basis.MultTranspose(snap2, rom_sol.GetBlock(midx[1]));
// {
//    printf("rom_sol:\n");
//    for (int k = 0; k < rom_sol.Size(); k++)
//       printf("%.3e\t", rom_sol(k));
//    printf("\n");
// }
//       rom_rhs2_vec = 0.0;
//       rform->InterfaceAddMult(rom_sol, rom_rhs2_vec);
// {
//    printf("rom_rhs2_vec:\n");
//    for (int k = 0; k < rom_rhs2_vec.Size(); k++)
//       printf("%.3e\t", rom_rhs2_vec(k));
//    printf("\n");
// }

//       for (int b = 0; b < num_basis; b++)
//          rom_rhs2(b, s) = rom_rhs2_vec.GetBlock(midx[0])(b);
//       for (int b = 0; b < num_basis; b++)
//          rom_rhs2(b + num_basis, s) = rom_rhs2_vec.GetBlock(midx[1])(b);
//    }

//    for (int i = 0; i < NB; i++)
//       for (int j = 0; j < num_snap; j++)
//          EXPECT_NEAR(rom_rhs1(i, j), rom_rhs2(i, j), threshold);

//    delete topol;
//    delete dg_coll;
//    DeletePointers(fes);
//    delete nform;
//    delete rform;
//    delete basis_generator;
// }

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}