// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::nlelast;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound)
{
   SUCCEED();
}

TEST(DDSerialTest, Test_convergence_DG)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   config.dict_["domain-decomposition"]["type"] = "none";
   bool nonlinear = false;
   CheckConvergence(nonlinear);

   return;
}

TEST(DDSerialTest, Test_direct_solver_DG)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   config.dict_["domain-decomposition"]["type"] = "none";
   bool nonlinear = false;
   CheckConvergence(nonlinear);

   return;
}

TEST(DDSerialTest, CompareSolvers)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   config.dict_["domain-decomposition"]["type"] = "none";
   bool nonlinear = false;
   config.dict_["discretization"]["interface/alpha"] = 0.0;

   CompareLinMat();
   return;
}


/* TEST(DDSerialTest, Test_convergence_DG_DD)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   config.dict_["domain-decomposition"]["type"] = "interior_penalty";
   CheckConvergence();

   return;
} */

/* TEST(DDSerialTest, Test_direct_solver_DG_DD)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   config.dict_["domain-decomposition"]["type"] = "interior_penalty";
   CheckConvergence();

   return;
} */

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}