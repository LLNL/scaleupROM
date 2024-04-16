// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "input_parser.hpp"
#include "topology_handler.hpp"
#include "interfaceinteg.hpp"
#include "interface_form.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

void CheckGradient(InterfaceNonlinearFormIntegrator *integ)
{   
   // 1. Parse command-line options.
   std::string mesh_file = config.GetRequiredOption<std::string>("mesh/filename");
   int order = config.GetOption<int>("discretization/order", 1);
   int num_refinement = config.GetOption<int>("mesh/uniform_refinement", 0);

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

   Array<int> block_offsets(numSub + 1);
   block_offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
      block_offsets[i+1] = fes[i]->GetTrueVSize();
   block_offsets.PartialSum();

   BlockVector x(block_offsets), xs(block_offsets);
   for (int k = 0; k < x.Size(); k++)
   {
      x(k) = UniformRandom();
      xs(k) = UniformRandom();
   }

   InterfaceForm *a_itf = new InterfaceForm(meshes, fes, submesh);
   a_itf->AddIntefaceIntegrator(integ);

   Vector Nx(x.Size());
   Nx = 0.0;
   a_itf->InterfaceAddMult(x, Nx);
   double J0 = xs * (Nx);
   printf("J0: %.5E\n", J0);

   Array2D<SparseMatrix *> mats;
   mats.SetSize(numSub, numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
         mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());

   a_itf->InterfaceGetGradient(x, mats);

   BlockMatrix globalMat(block_offsets);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
      {
         mats(i, j)->Finalize();
         globalMat.SetBlock(i, j, mats(i, j));
      }

   Vector grad(x.Size());
   globalMat.MultTranspose(xs, grad);
   double gg = grad * grad;
   printf("gg: %.5E\n", gg);

   BlockVector x0(x);

   double error1 = 1.0e10;
   printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "amp", "J0", "J1", "dJdx", "error");
   for (int k = 0; k < 40; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      double dx = amp;
      if (gg > 1.0e-14) dx /= sqrt(gg);

      x.Set(1.0, x0);
      x.Add(dx, grad);

      Nx = 0.0;
      a_itf->InterfaceAddMult(x, Nx);
      double J1 = xs * (Nx);
      double dJdx = (J1 - J0) / dx;
      double error = abs((dJdx - gg));
      if (gg > 1.0e-14) error /= abs(gg);

      printf("%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n", amp, J0, J1, dJdx, error);

      if (k > 4)
      {
         if (error > error1)
            break;
         else
            error1 = error;
      }
   }
   EXPECT_TRUE(error1 < 1.0e-6);

   delete submesh;
   delete a_itf;
   DeletePointers(fes);
   DeletePointers(mats);

   return;
}

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DGLaxFriedrichsFlux, Test_grad_interior)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;

   ConstantCoefficient pi(3.141592);
   auto *nlc_nlfi = new DGLaxFriedrichsFluxIntegrator(pi);
    
   CheckGradient(nlc_nlfi);

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