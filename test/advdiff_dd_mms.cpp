// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::advdiff;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DDSerialTest, Test_convergence)
{
   config = InputParser("inputs/dd_mms.yml");
   CheckConvergence();

   return;
}

TEST(DDSerialTest, Test_direct_solver)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["solver"]["direct_solve"] = true;
   CheckConvergence();

   return;
}

TEST(DDSerial_component_wise_test, Test_convergence)
{
   config = InputParser("inputs/dd_mms.component.yml");
   CheckConvergence();

   return;
}

TEST(DDSerial_component_3D_hex_test, Test_convergence)
{
   config = InputParser("inputs/dd_mms.comp.3d.yml");
   CheckConvergence();

   return;
}

// TEST(DDSerial_component_3D_tet_test, Test_convergence)
// {
//    config = InputParser("inputs/dd_mms.comp.3d.yml");
//    config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "meshes/dd_mms.3d.tet.mesh";
//    CheckConvergence();

//    return;
// }

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}