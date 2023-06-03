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

TEST(DDSerialTest, Test_convergence)
{
   config = InputParser("test.component.yml");
   config.dict_["discretization"]["order"] = 2;
   config.dict_["manufactured_solution"]["baseline_refinement"] = 0;
   config.dict_["manufactured_solution"]["number_of_refinement"] = 2;
   config.dict_["solver"]["absolute_tolerance"] = 1.0E-12;
   config.dict_["solver"]["relative_tolerance"] = 1.0E-12;
   // TODO: add ROM capability for stokes solver.
   config.dict_["main"]["use_rom"] = false;
   CheckConvergence(1.0);

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