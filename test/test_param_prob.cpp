#include<gtest/gtest.h>
#include "mfem.hpp"
#include "parameterized_problem.hpp"
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

TEST(ParameterizedProblemTest, Test_Parsing)
{
   config = InputParser("inputs/test_param_prob.yml");

   ParameterizedProblem test;

   EXPECT_EQ(test.GetProblemName(), "custom_problem");
   EXPECT_EQ(test.GetNumParams(), 2);

   Array<double> *test_x = test.GetDoubleParamSpace("x");
   Array<double> true_x({0.0, 1.0, 2.0, 3.0, 4.0});
   EXPECT_EQ(test_x->Size(), 5);
   for (int k = 0; k < test_x->Size(); k++)
      EXPECT_TRUE(abs((*test_x)[k] - true_x[k]) < 1.0e-15);

   Array<double> *test_y = test.GetDoubleParamSpace("y");
   Array<double> true_y({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
   EXPECT_EQ(test_y->Size(), 8);
   for (int k = 0; k < test_y->Size(); k++)
      EXPECT_TRUE(abs((*test_y)[k] - true_y[k]) < 1.0e-15);

   return;
}

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}