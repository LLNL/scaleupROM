#include<gtest/gtest.h>
#include "hdf5_utils.hpp"
#include "random.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(ReadAttribute_string_test, Test_hdf5)
{
   std::string filename("meshes/dd_mms.h5");
   hid_t file_id;
   hid_t grp_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   std::string tmp;
   hdf5_utils::ReadAttribute(grp_id, "0", tmp);
   printf("%s\n", tmp.c_str());

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

TEST(WriteAttribute_string_test, Test_hdf5)
{
   std::string filename("test.h5");
   hid_t file_id;
   hid_t grp_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   std::string tmp = "world";
   hdf5_utils::WriteAttribute(file_id, "hello", tmp);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   std::string result;
   hdf5_utils::ReadAttribute(file_id, "hello", result);
   printf("tmp: %s\n", tmp.c_str());
   printf("result: %s\n", result.c_str());
   EXPECT_EQ(tmp, result);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

TEST(WriteDataset_test, Test_hdf5)
{
   std::string filename("test.h5");
   hid_t file_id;
   hid_t grp_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   Array<int> int_ans(5);
   for (int i = 0; i < int_ans.Size(); i++) int_ans[i] = rand();
   Array<double> double_ans(8);
   for (int d = 0; d < double_ans.Size(); d++) double_ans[d] = UniformRandom();
   Array2D<int> int2_ans(3,4);
   for (int i = 0; i < int2_ans.NumRows(); i++)
      for (int j = 0; j < int2_ans.NumCols(); j++) int2_ans(i,j) = rand();
   Array2D<double> double2_ans(5,4);
   for (int i = 0; i < double2_ans.NumRows(); i++)
      for (int j = 0; j < double2_ans.NumCols(); j++) double2_ans(i,j) = UniformRandom();
   
   hdf5_utils::WriteDataset(file_id, "int_ans", int_ans);
   hdf5_utils::WriteDataset(file_id, "double_ans", double_ans);
   hdf5_utils::WriteDataset(file_id, "int2_ans", int2_ans);
   hdf5_utils::WriteDataset(file_id, "double2_ans", double2_ans);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   Array<int> int_result;
   Array<double> double_result;
   Array2D<int> int2_result;
   Array2D<double> double2_result;
   hdf5_utils::ReadDataset(file_id, "int_ans", int_result);
   hdf5_utils::ReadDataset(file_id, "double_ans", double_result);
   hdf5_utils::ReadDataset(file_id, "int2_ans", int2_result);
   hdf5_utils::ReadDataset(file_id, "double2_ans", double2_result);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   EXPECT_EQ(int_result.Size(), int_ans.Size());
   EXPECT_EQ(double_result.Size(), double_ans.Size());
   EXPECT_EQ(int2_result.NumRows(), int2_ans.NumRows());
   EXPECT_EQ(int2_result.NumCols(), int2_ans.NumCols());
   EXPECT_EQ(double2_result.NumRows(), double2_ans.NumRows());
   EXPECT_EQ(double2_result.NumCols(), double2_ans.NumCols());

   for (int i = 0; i < int_ans.Size(); i++)
      EXPECT_EQ(int_result[i], int_ans[i]);
   for (int d = 0; d < double_ans.Size(); d++)
      EXPECT_EQ(double_result[d], double_ans[d]);
   for (int i = 0; i < int2_ans.NumRows(); i++)
      for (int j = 0; j < int2_ans.NumCols(); j++)
         EXPECT_EQ(int2_result(i,j), int2_ans(i,j));
   for (int i = 0; i < double2_ans.NumRows(); i++)
      for (int j = 0; j < double2_ans.NumCols(); j++)
         EXPECT_EQ(double2_result(i,j), double2_ans(i,j));

   return;
}

TEST(DenseMatrix_test, Test_hdf5)
{
   std::string filename("test.h5");
   hid_t file_id;
   hid_t grp_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   DenseMatrix double2_ans(5,4);
   for (int i = 0; i < double2_ans.NumRows(); i++)
      for (int j = 0; j < double2_ans.NumCols(); j++) double2_ans(i,j) = UniformRandom();
   
   hdf5_utils::WriteDataset(file_id, "double2_ans", double2_ans);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   DenseMatrix double2_result;
   hdf5_utils::ReadDataset(file_id, "double2_ans", double2_result);

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   EXPECT_EQ(double2_result.NumRows(), double2_ans.NumRows());
   EXPECT_EQ(double2_result.NumCols(), double2_ans.NumCols());

   for (int i = 0; i < double2_ans.NumRows(); i++)
      for (int j = 0; j < double2_ans.NumCols(); j++)
         EXPECT_EQ(double2_result(i,j), double2_ans(i,j));

   return;
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
