// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "interfaceinteg.hpp"
#include "rom_interfaceform.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-14;
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

   ROMInterfaceForm *rform = new ROMInterfaceForm(meshes, fes, submesh);
   rform->AddInterfaceIntegrator(integ2);
   for (int m = 0; m < numSub; m++)
      rform->SetBasisAtSubdomain(m, *basis[m]);
   rform->UpdateBlockOffsets();

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nport = submesh->GetNumPorts();
   const int nqe = ir->GetNPoints();
   Array<double> const& w_el = ir->GetWeights();
   for (int p = 0; p < nport; p++)
   {
      Array<InterfaceInfo> *interface_infos = submesh->GetInterfaceInfos(p);
      Array<int> sample_itf(0), sample_qp(0);
      Array<double> sample_qw(0);
      for (int itf = 0; itf < interface_infos->Size(); itf++)
         for (int q = 0; q < nqe; q++)
         {
            sample_itf.Append(itf);
            sample_qp.Append(q);
            sample_qw.Append(w_el[q]);
         }

      rform->UpdateInterFaceIntegratorSampling(0, p, sample_itf, sample_qp, sample_qw);
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

   ROMInterfaceForm *rform = new ROMInterfaceForm(meshes, fes, submesh);
   rform->AddInterfaceIntegrator(integ2);
   for (int m = 0; m < numSub; m++)
      rform->SetBasisAtSubdomain(m, *basis[m]);
   rform->UpdateBlockOffsets();

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nport = submesh->GetNumPorts();
   const int nqe = ir->GetNPoints();
   Array<double> const& w_el = ir->GetWeights();
   for (int p = 0; p < nport; p++)
   {
      Array<InterfaceInfo> *interface_infos = submesh->GetInterfaceInfos(p);
      Array<int> sample_itf(0), sample_qp(0);
      Array<double> sample_qw(0);
      for (int itf = 0; itf < interface_infos->Size(); itf++)
         for (int q = 0; q < nqe; q++)
         {
            sample_itf.Append(itf);
            sample_qp.Append(q);
            sample_qw.Append(w_el[q]);
         }

      rform->UpdateInterFaceIntegratorSampling(0, p, sample_itf, sample_qp, sample_qw);
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
   const int num_snap = UniformRandom(3, 5);
   const int num_basis = num_snap;

   // a fictitious snapshots.
   DenseMatrix snapshots1(ndofs1, num_snap);
   DenseMatrix snapshots2(ndofs2, num_snap);
   for (int j = 0; j < num_snap; j++)
   {
      for (int i = 0; i < ndofs1; i++)
         snapshots1(i, j) = 2.0 * UniformRandom() - 1.0;
      for (int i = 0; i < ndofs2; i++)
         snapshots2(i, j) = 2.0 * UniformRandom() - 1.0;
   }

   /* bases generated from fictitious snapshots */
   DenseMatrix basis1(ndofs1, num_basis);
   DenseMatrix basis2(ndofs2, num_basis);
   const CAROM::Matrix *carom_snapshots1, *carom_snapshots2;
   CAROM::BasisGenerator *basis1_generator, *basis2_generator;
   {
      CAROM::Options options(ndofs1, num_snap, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis1_generator = new CAROM::BasisGenerator(options, false, "test_basis1");
      Vector snapshot(ndofs1);
      for (int s = 0; s < num_snap; s++)
      {
         snapshots1.GetColumnReference(s, snapshot);
         basis1_generator->takeSample(snapshot.GetData());
      }
      basis1_generator->endSamples();
      carom_snapshots1 = basis1_generator->getSnapshotMatrix();
      const CAROM::Matrix *carom_basis = basis1_generator->getSpatialBasis();
      CAROM::CopyMatrix(*carom_basis, basis1);
   }
   {
      CAROM::Options options(ndofs2, num_snap, 1, true);
      options.static_svd_preserve_snapshot = true;
      basis2_generator = new CAROM::BasisGenerator(options, false, "test_basis2");
      Vector snapshot(ndofs2);
      for (int s = 0; s < num_snap; s++)
      {
         snapshots2.GetColumnReference(s, snapshot);
         basis2_generator->takeSample(snapshot.GetData());
      }
      basis2_generator->endSamples();
      carom_snapshots2 = basis2_generator->getSnapshotMatrix();
      const CAROM::Matrix *carom_basis = basis2_generator->getSpatialBasis();
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
   ROMInterfaceForm *rform(new ROMInterfaceForm(meshes, fes, submesh));
   rform->AddInterfaceIntegrator(integ2);

   Array<DenseMatrix *> bases(numSub);
   for (int m = 0; m < bases.Size(); m++)
   {
      bases[m] = new DenseMatrix(fes[m]->GetTrueVSize(), num_basis);
      *bases[m] = 0.0;
      rform->SetBasisAtSubdomain(m, *bases[m]);
   }
   rform->SetBasisAtSubdomain(midx[0], basis1);
   rform->SetBasisAtSubdomain(midx[1], basis2);
   rform->UpdateBlockOffsets();
   // rform->SetPrecomputeMode(false);

   CAROM::Vector rhs1(num_snap * num_basis * 2, false);
   CAROM::Vector rhs2(num_snap * num_basis * 2, false);
   CAROM::Matrix Gt(1, 1, true);

   /* exact right-hand side by inner product of basis and fom vectors */
   Vector rhs_vec1(ndofs1), rhs_vec2(ndofs2);
   Vector snap1(ndofs1), snap2(ndofs2);
   Vector basis_col;
   for (int s = 0; s < num_snap; s++)
   {
      snapshots1.GetColumnReference(s, snap1);
      snapshots2.GetColumnReference(s, snap2);

      rhs_vec1 = 0.0; rhs_vec2 = 0.0;
      nform->AssembleInterfaceVector(mesh1, mesh2, fes1, fes2, itf_infos, snap1, snap2, rhs_vec1, rhs_vec2);

      for (int b = 0; b < num_basis; b++)
      {
         basis1.GetColumnReference(b, basis_col);
         rhs1(b + s * num_basis * 2) = basis_col * rhs_vec1;
      }
      for (int b = 0; b < num_basis; b++)
      {
         basis2.GetColumnReference(b, basis_col);
         rhs1(b + num_basis + s * num_basis * 2) = basis_col * rhs_vec2;
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
                         basis1, basis2, 0, 0, fes1, fes2,
                         itf_infos, integ2, Gt, rhs2);

   for (int k = 0; k < rhs1.dim(); k++)
      EXPECT_NEAR(rhs1(k), rhs2(k), threshold);

   double eqp_tol = 1.0e-10;
   Array<int> sample_el, sample_qp;
   Array<double> sample_qw;
   const int nqe = ir->GetNPoints();
   rform->TrainEQPForIntegrator(nqe, Gt, rhs2, eqp_tol, sample_el, sample_qp, sample_qw);
   // if (rform->PrecomputeMode()) rform->PrecomputeCoefficients();
   rform->UpdateInterFaceIntegratorSampling(0, pidx, sample_el, sample_qp, sample_qw);

   DenseMatrix rom_rhs1(rhs1.getData(), 2 * num_basis, num_snap), rom_rhs2(2 * num_basis, num_snap);
   Array<int> rom_blocks = rform->GetBlockOffsets();
   BlockVector rom_sol(rom_blocks), rom_rhs2_vec(rom_blocks);
   for (int s = 0; s < num_snap; s++)
   {
      rom_sol = 0.0;
      snapshots1.GetColumnReference(s, snap1);
      snapshots2.GetColumnReference(s, snap2);

      basis1.MultTranspose(snap1, rom_sol.GetBlock(midx[0]));
      basis2.MultTranspose(snap2, rom_sol.GetBlock(midx[1]));
      rom_rhs2_vec = 0.0;
      rform->InterfaceAddMult(rom_sol, rom_rhs2_vec);

      for (int b = 0; b < num_basis; b++)
      {
         rom_rhs2(b, s) = rom_rhs2_vec.GetBlock(midx[0])(b);
         rom_rhs2(b + num_basis, s) = rom_rhs2_vec.GetBlock(midx[1])(b);
      }
   }

   for (int i = 0; i < num_basis; i++)
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

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}