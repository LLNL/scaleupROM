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

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}