// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::poisson;

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
   CheckConvergence();

   return;
}

TEST(DDSerialTest, Test_direct_solver_DG)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tri.mesh";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["discretization"]["full-discrete-galerkin"] = true;
   CheckConvergence();

   return;
}

TEST(DDSerialTest, Test_convergence_NoDG)
{
   config = InputParser("inputs/dd_mms.yml");
   //config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tet.mesh";
   config.dict_["discretization"]["full-discrete-galerkin"] = false;
   CheckConvergence();

   return;
}

TEST(DDSerialTest, Test_direct_solver_NoDG)
{
   config = InputParser("inputs/dd_mms.yml");
   //config.dict_["mesh"]["filename"] = "../examples/linelast/meshes/beam-tet.mesh";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["discretization"]["full-discrete-galerkin"] = false;
   CheckConvergence();

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