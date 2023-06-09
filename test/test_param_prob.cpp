#include<gtest/gtest.h>
#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "random_sample_generator.hpp"
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
   sample_gen.SetParamSpaceSizes();

   EXPECT_EQ(test->GetProblemName(), "poisson0");
   EXPECT_EQ(sample_gen.GetNumSampleParams(), 2);

   Parameter *param = NULL;
   param = sample_gen.GetParam(0);
   EXPECT_EQ(param->GetSize(), 5);
   EXPECT_EQ(param->GetKey(), "test/k");

   Vector true_x({0.0, 1.0, 2.0, 3.0, 4.0});
   for (int s = 0; s < param->GetSize(); s++)
   {
      param->SetParam(s, config);
      double k = config.GetRequiredOption<double>(param->GetKey());
      EXPECT_TRUE(abs(k - true_x[s]) < 1.0e-15);
   }

   param = sample_gen.GetParam(1);
   EXPECT_EQ(param->GetSize(), 8);
   EXPECT_EQ(param->GetKey(), "test/offset");

   Vector true_y({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
   for (int s = 0; s < param->GetSize(); s++)
   {
      param->SetParam(s, config);
      double offset = config.GetRequiredOption<double>(param->GetKey());
      EXPECT_TRUE(abs(offset - true_y[s]) < 1.0e-15);
   }

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

TEST(RandomSampleGeneratorTest, Test_Parsing)
{
   config = InputParser("inputs/test_param_prob.yml");

   ParameterizedProblem *test = InitParameterizedProblem();
   RandomSampleGenerator sample_gen(MPI_COMM_WORLD, test);
   sample_gen.SetParamSpaceSizes();

   EXPECT_EQ(test->GetProblemName(), "poisson0");
   EXPECT_EQ(sample_gen.GetNumSampleParams(), 2);

   Parameter *param = NULL;
   param = sample_gen.GetParam(0);
   EXPECT_EQ(param->GetSize(), 7);
   EXPECT_EQ(param->GetKey(), "test/k");

   param = sample_gen.GetParam(1);
   EXPECT_EQ(param->GetSize(), 7);
   EXPECT_EQ(param->GetKey(), "test/offset");

   for (int s = 0; s < sample_gen.GetTotalSampleSize(); s++)
   {
      if (sample_gen.IsMyJob(s))
      {
         sample_gen.SetSampleParams(s);
         double k = config.GetRequiredOption<double>("test/k");
         double offset = config.GetRequiredOption<double>("test/offset");
         printf("%d: (%.4f, %.4f) - rank %d\n", s, k, offset, sample_gen.GetProcRank());
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