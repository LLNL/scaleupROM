// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "hdf5_utils.hpp"
#include <stdlib.h>

using namespace mfem;

namespace hdf5_utils
{

hid_t GetNativeType(hid_t type)
{
   hid_t p_type;
   H5T_class_t type_class;

   type_class = H5Tget_class(type);
   if (type_class == H5T_BITFIELD)
      p_type = H5Tcopy(type);
   else
      p_type = H5Tget_native_type(type, H5T_DIR_DEFAULT);

   return p_type;
}

void ReadAttribute(hid_t source, std::string attribute, std::string &value)
{
   herr_t status;
   hid_t attr;
   attr = H5Aopen_name(source, attribute.c_str());
   hid_t tmp = H5Aget_type(attr);

   // hid_t attrType = hdf5_utils::GetNativeType(tmp);
   // status = H5Aread(attr, attrType, &tmp_str);

   /*
      For some reason H5Aread deletes tmp_str.
      If tmp_str is not a pointer,
      it leads to double-free at the end of this fuction.
   */
   std::string *tmp_str = new std::string;
   status = H5Aread(attr, tmp, tmp_str);
   assert(status >= 0);

   H5Aclose(attr);
   value = tmp_str->c_str();
}

// This currently only reads the first item. Do not use it.
void ReadDataset(hid_t source, std::string dataset, std::vector<std::string> &value)
{
   mfem_error("hdf5_utils::ReadDataset for std::string is not fully implemented yet!\n");
   herr_t errf = 0;
   
   hid_t dset_id = H5Dopen(source, dataset.c_str(), H5P_DEFAULT);
   assert(dset_id >= 0);

   hid_t dspace_id = H5Dget_space(dset_id);
   int ndims = H5Sget_simple_extent_ndims(dspace_id);
   assert(ndims == 1);

   hsize_t dims[ndims];
   errf = H5Sget_simple_extent_dims(dspace_id, dims, NULL);
   assert(errf >= 0);

   hid_t tmp = H5Dget_type(dset_id);
   hid_t dataType = hdf5_utils::GetNativeType(tmp);

   value.resize(dims[0]);
   // H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data());
   std::string tmp_strs[dims[0]];
   // H5Dread(dset_id, tmp, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.data());
   errf = H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_strs);
   assert(errf >= 0);

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

}
