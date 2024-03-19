// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "../mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::poisson;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(Poisson, Test_convergence)
{
   config = InputParser("test.component.yml");
   CheckConvergence();

   return;
}

TEST(AdvDiff, Test_convergence)
{
   config = InputParser("test.component.yml");
   config.dict_["adv-diff"]["peclet_number"] = 1.1;
   mms::advdiff::CheckConvergence();

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