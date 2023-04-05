#include<gtest/gtest.h>
#include "hdf5_utils.hpp"
#include <fstream>
#include <iostream>

using namespace std;

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

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
