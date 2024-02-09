// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "mms_suite.hpp"
#include "topology_handler.hpp"

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DG_BDR_NORMAL_LF_Test, Test_Quad)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;
   mms::fem::dg_bdr_normal_lf::CheckConvergence();

   return;
}

TEST(DG_BDR_NORMAL_LF_Test, Test_Tri)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;
   config.dict_["mesh"]["filename"] = "meshes/square.tri.mesh";
   mms::fem::dg_bdr_normal_lf::CheckConvergence();

   return;
}

// TODO(axel) : test for InterfaceDGElasticityIntegrator.
/* In essence, check the consistency with DGElasticityIntegrator*/
TEST(InterfaceDGElasticityIntegrator, Test_Quad)
{
   config = InputParser("inputs/dd_mms.yml");
   /* set your own parameters */
   const int order = 1;
   config.dict_["discretization"]["order"] = order;
   config.dict_["mesh"]["filename"] = "meshes/test.2x1.mesh";

   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler;
   Mesh *pmesh = submesh->GetGlobalMesh();
   const int dim = pmesh->Dimension();
   const int udim = dim;
   Array<Mesh *> meshes;
   TopologyData topol_data;
   submesh->ExportInfo(meshes, topol_data);

   FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace *pfes = new FiniteElementSpace(pmesh, fec, udim);
   Array<FiniteElementSpace *> fes(meshes.Size());
   for (int k = 0; k < meshes.Size(); k++)
      fes[k] = new FiniteElementSpace(meshes[k], fec, udim);

   /* will have to initialize this coefficient */
   PWConstCoefficient *lambda_cs, *mu_cs;
   double alpha, kappa;

   /* assemble standard DGElasticityIntegrator Assemble */
   BilinearForm pform(pfes);
   pform.AddInteriorFaceIntegrator(new DGElasticityIntegrator(*lambda_cs, *mu_cs, alpha, kappa));
   pform.Assemble();
   pform.Finalize();

   /* assemble InterfaceDGElasticityIntegrator Assemble */
   InterfaceForm a_itf(meshes, fes, submesh);
   a_itf.AddIntefaceIntegrator(new InterfaceDGElasticityIntegrator(alpha, kappa));

   Array2D<SparseMatrix *> mats;
   mats.SetSize(topol_data.numSub, topol_data.numSub);
   for (int i = 0; i < topol_data.numSub; i++)
      for (int j = 0; j < topol_data.numSub; j++)
         mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());

   a_itf.AssembleInterfaceMatrixes(mats);
   
   for (int i = 0; i < topol_data.numSub; i++)
      for (int j = 0; j < topol_data.numSub; j++)
         mats(i, j)->Finalize();

   /* 
      Assembled matrix from two integrators should have identical non-zero entries.
      (but their indices might be different)
      Right now their is only one interface with two elements.
      Easiest way might be visualizing matrices after changing them into DenseMatrix.
    */
   DenseMatrix *pmat = pform.SpMat().ToDenseMatrix();
   Array2D<DenseMatrix *> smats(topol_data.numSub, topol_data.numSub);
   for (int i = 0; i < topol_data.numSub; i++)
      for (int j = 0; j < topol_data.numSub; j++)
         smats(i, j) = mats(i, j)->ToDenseMatrix();
   // print out or compare the entries...


   /* clean up variables */
   // delete ..
   return;
}

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}