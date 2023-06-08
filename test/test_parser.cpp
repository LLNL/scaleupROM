#include<gtest/gtest.h>
#include "input_parser.hpp"
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

TEST(YAML_test, BasicNodetest)
{
   YAML::Node test = YAML::LoadFile("inputs/test.parser.yml");
   config = InputParser("inputs/test.parser.yml");

   YAML::Node zombie = test["non-existing"];
   EXPECT_TRUE(test["codename"]);
   EXPECT_TRUE(!test["non-existing"]);

   EXPECT_TRUE(config.FindNode("codename"));
   EXPECT_TRUE(!config.FindNode("non-existing"));

   std::string codename = config.GetRequiredOption<std::string>("codename");
   EXPECT_EQ(codename, "scaleupROM");

   return;
}

TEST(YAML_test, SetOptionTest)
{
   YAML::Node test = YAML::LoadFile("inputs/test.parser.yml");
   config = InputParser("inputs/test.parser.yml");

   YAML::Node zombie = test["non-existing"];
   EXPECT_TRUE(test["codename"]);
   EXPECT_TRUE(!test["non-existing"]);

   EXPECT_TRUE(config.FindNode("codename"));
   EXPECT_TRUE(!config.FindNode("non-existing"));

   std::string codename = config.GetRequiredOption<std::string>("codename");
   EXPECT_EQ(codename, "scaleupROM");

   config.SetOption<std::string>("codename", "mfem");

   std::string codename2 = config.GetRequiredOption<std::string>("codename");
   EXPECT_EQ(codename2, "mfem");

   codename = test["codename"].as<std::string>();
   EXPECT_EQ(codename, "scaleupROM");

   config.SetOptionInDict<std::string>("codename", "mfem", test);

   codename2 = test["codename"].as<std::string>();
   EXPECT_EQ(codename2, "mfem");

   int default_val = config.GetOption<int>("non-existing/test_category/test_int", -1);
   EXPECT_EQ(default_val, -1);

   config.SetOption<int>("non-existing/test_category/test_int", 3);
   int now_exists = config.GetRequiredOption<int>("non-existing/test_category/test_int");
   EXPECT_EQ(now_exists, 3);

   return;
}

// TODO: add more tests from sketches/yaml_example.cpp.

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}