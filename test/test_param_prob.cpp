#include<gtest/gtest.h>
#include "mfem.hpp"
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

   SampleGenerator sample_gen(MPI_COMM_WORLD);
   sample_gen.SetParamSpaceSizes();

   EXPECT_EQ(sample_gen.GetNumSampleParams(), 3);

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

   param = sample_gen.GetParam(2);
   EXPECT_EQ(param->GetSize(), 4);
   EXPECT_EQ(param->GetKey(), "test/filename");

   std::vector<std::string> true_filenames(4);
   true_filenames[0] = "testfile.00000000.h5";
   true_filenames[1] = "testfile.00000004.h5";
   true_filenames[2] = "testfile.00000008.h5";
   true_filenames[3] = "testfile.00000012.h5";
   for (int s = 0; s < param->GetSize(); s++)
   {
      param->SetParam(s, config);
      std::string filename = config.GetRequiredOption<std::string>(param->GetKey());
      EXPECT_EQ(filename, true_filenames[s]);
   }

   for (int s = 0; s < sample_gen.GetTotalSampleSize(); s++)
   {
      if (sample_gen.IsMyJob(s))
      {
         Array<int> local_idx = sample_gen.GetSampleIndex(s);
         printf("%d: (%d, %d, %d) - rank %d\n", s,
               local_idx[0], local_idx[1], local_idx[2],
               sample_gen.GetProcRank());
      }
   }

   return;
}

TEST(RandomSampleGeneratorTest, Test_Parsing)
{
   config = InputParser("inputs/test_param_prob.yml");

   RandomSampleGenerator sample_gen(MPI_COMM_WORLD);
   sample_gen.SetParamSpaceSizes();

   EXPECT_EQ(sample_gen.GetNumSampleParams(), 3);

   Parameter *param = NULL;
   param = sample_gen.GetParam(0);
   EXPECT_EQ(param->GetSize(), 7);
   EXPECT_EQ(param->GetKey(), "test/k");

   param = sample_gen.GetParam(1);
   EXPECT_EQ(param->GetSize(), 7);
   EXPECT_EQ(param->GetKey(), "test/offset");

   param = sample_gen.GetParam(2);
   EXPECT_EQ(param->GetSize(), 7);
   EXPECT_EQ(param->GetKey(), "test/filename");

   for (int s = 0; s < sample_gen.GetTotalSampleSize(); s++)
   {
      if (sample_gen.IsMyJob(s))
      {
         sample_gen.SetSampleParams(s);
         double k = config.GetRequiredOption<double>("test/k");
         double offset = config.GetRequiredOption<double>("test/offset");
         std::string filename = config.GetRequiredOption<std::string>("test/filename");
         printf("%d: (%.4f, %.4f, %s) - rank %d\n", s, k, offset, filename.c_str(), sample_gen.GetProcRank());
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