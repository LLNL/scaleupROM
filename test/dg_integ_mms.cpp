// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
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

void _PrintMatrix(string filename, DenseMatrix &mat)
{
   std::ofstream outfile(filename);

   double tol = 1e-7;
   double val = 0.0;

   int nonzeros = 0;
   for (size_t i = 0; i < mat.Height(); i++)
   {
      for (size_t j = 0; j < mat.Width(); j++)
      {
         val = mat(i, j);
         if (abs(val) < tol)
         {
            val = 0.0;
            nonzeros++;
         }

         outfile << setprecision(2) << val << " ";
      }
      outfile << endl;
   }
   outfile.close();
   cout << "done printing matrix" << endl;
   cout << "number of nonzeros:" << nonzeros << endl;
}

TEST(InterfaceDGElasticityIntegrator, Test_Quad)
{
   config = InputParser("inputs/dd_mms.yml");
   /* set your own parameters */
   const int order = 1;
   config.dict_["discretization"]["order"] = order;
   config.dict_["mesh"]["filename"] = "meshes/test.2x1.mesh";

   const int numBdr = 4; // hacky way to set the number of boundary attributes

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
   double alpha = -1.0, kappa = (order + 1) * (order + 1);

   Array<ConstantCoefficient *> lambda_c;
   Array<ConstantCoefficient *> mu_c;

   lambda_c.SetSize(meshes.Size());
   lambda_c = NULL;

   mu_c.SetSize(meshes.Size());
   mu_c = NULL;

   for (size_t i = 0; i < meshes.Size(); i++)
   {
      lambda_c[i] = new ConstantCoefficient(1.0);
      mu_c[i] = new ConstantCoefficient(1.0);
   }
   /* assemble standard DGElasticityIntegrator Assemble */
   BilinearForm pform(pfes);
   pform.AddInteriorFaceIntegrator(new DGElasticityIntegrator(*lambda_c[0], *mu_c[0], alpha, kappa));
   pform.Assemble();
   pform.Finalize();

   /* assemble InterfaceDGElasticityIntegrator Assemble */
   InterfaceForm a_itf(meshes, fes, submesh);
   a_itf.AddInterfaceIntegrator(new InterfaceDGElasticityIntegrator(lambda_c[0], mu_c[0], alpha, kappa));
   Array2D<SparseMatrix *> mats;
   mats.SetSize(topol_data.numSub, topol_data.numSub);
   for (int i = 0; i < topol_data.numSub; i++)
      for (int j = 0; j < topol_data.numSub; j++)
         mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());
   a_itf.AssembleInterfaceMatrices(mats);
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
   _PrintMatrix("mfem_mat.txt", *pmat);
   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = 8;
   offsets[2] = 16;
   BlockMatrix smats(offsets);
   for (int i = 0; i < topol_data.numSub; i++)
      for (int j = 0; j < topol_data.numSub; j++)
      {
         smats.SetBlock(i, j, mats(i, j));
      }
   std::string filename = "scaleup_mat.txt";
   _PrintMatrix(filename, *(smats.CreateMonolithic()->ToDenseMatrix()));
   // print out or compare the entries...
   return;
}

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}