#include<gtest/gtest.h>
#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "sample_generator.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(SampleGeneratorTest, Test_Parsing)
{
   config = InputParser("inputs/test_param_prob.yml");

   ParameterizedProblem *test = InitParameterizedProblem();
   SampleGenerator sample_gen(MPI_COMM_WORLD, test);

   EXPECT_EQ(test->GetProblemName(), "poisson0");
   EXPECT_EQ(sample_gen.GetNumSampleParams(), 2);

   Vector *test_x = sample_gen.GetDoubleParamSpace("k");
   Vector true_x({0.0, 1.0, 2.0, 3.0, 4.0});
   EXPECT_EQ(test_x->Size(), 5);
   for (int k = 0; k < test_x->Size(); k++)
      EXPECT_TRUE(abs((*test_x)[k] - true_x[k]) < 1.0e-15);

   Vector *test_y = sample_gen.GetDoubleParamSpace("offset");
   Vector true_y({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
   EXPECT_EQ(test_y->Size(), 8);
   for (int k = 0; k < test_y->Size(); k++)
      EXPECT_TRUE(abs((*test_y)[k] - true_y[k]) < 1.0e-15);

   for (int s = 0; s < sample_gen.GetTotalSampleSize(); s++)
   {
      if (sample_gen.IsMyJob(s))
      {
         Array<int> local_idx = sample_gen.GetSampleIndex(s);
         printf("%d: (%d, %d) - rank %d\n", s, local_idx[0], local_idx[1], sample_gen.GetProcRank());
      }
   }

   return;
}

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}