// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "../mms_suite.hpp"

using namespace std;
using namespace mfem;
using namespace mms::stokes;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(StokesFlow, Test_convergence)
{
   config = InputParser("test.component.yml");
   config.dict_["discretization"]["order"] = 2;
   config.dict_["manufactured_solution"]["baseline_refinement"] = 0;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 3;
   config.dict_["mesh"]["component-wise"]["vertex_gap_threshold"] = 1.0e-9;
   config.dict_["solver"]["direct_solve"] = true;
   // TODO: add ROM capability for stokes solver.
   config.dict_["main"]["use_rom"] = false;
   CheckConvergence(1.0);

   return;
}

TEST(SteadyNS, Test_convergence)
{
   config = InputParser("test.component.yml");
   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["discretization"]["order"] = 2;
   config.dict_["manufactured_solution"]["baseline_refinement"] = 0;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 3;
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["mesh"]["component-wise"]["vertex_gap_threshold"] = 1.0e-9;
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["solver"]["print_level"] = 1;
   // TODO: add ROM capability for stokes solver.
   config.dict_["main"]["use_rom"] = false;
   mms::steady_ns::CheckConvergence(1.0);

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