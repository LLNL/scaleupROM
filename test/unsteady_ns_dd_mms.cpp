// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::unsteady_ns;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(DDSerialTest, Test_convergence)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["navier-stokes"]["operator-type"] = "lf";
   config.dict_["discretization"]["order"] = 1;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 3;
   config.dict_["navier-stokes"]["timestep_size"] = 0.01;
   config.dict_["navier-stokes"]["number_of_timesteps"] = 50;
   CheckConvergence();

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