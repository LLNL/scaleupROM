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

int GetDatasetSize(hid_t &source, std::string dataset, hsize_t* &dims)
{
   herr_t errf = 0;
   
   hid_t dset_id = H5Dopen(source, dataset.c_str(), H5P_DEFAULT);
   assert(dset_id >= 0);

   hid_t dspace_id = H5Dget_space(dset_id);
   int ndims = H5Sget_simple_extent_ndims(dspace_id);
   assert(ndims > 0);

   dims = new hsize_t[ndims];
   errf = H5Sget_simple_extent_dims(dspace_id, dims, NULL);
   assert(errf >= 0);

   errf = H5Dclose(dset_id);
   assert(errf >= 0);

   return ndims;
}

void ReadAttribute(hid_t &source, std::string attribute, std::string &value)
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

void WriteAttribute(hid_t &dest, std::string attribute, const std::string &value)
{
   hid_t attr, status;
   hid_t attrType = H5Tcreate(H5T_STRING, H5T_VARIABLE);
   hid_t dataspaceId = H5Screate(H5S_SCALAR);

   attr = H5Acreate(dest, attribute.c_str(), attrType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
   assert(attr >= 0);
   status = H5Awrite(attr, attrType, &value);
   assert(status >= 0);
   H5Aclose(attr);
   H5Sclose(dataspaceId);
   H5Tclose(attrType);
}

void ReadDataset(hid_t &source, std::string dataset, DenseMatrix &value)
{
   herr_t errf = 0;
   
   hid_t dset_id = H5Dopen(source, dataset.c_str(), H5P_DEFAULT);
   assert(dset_id >= 0);

   hid_t dspace_id = H5Dget_space(dset_id);
   int ndims = H5Sget_simple_extent_ndims(dspace_id);
   assert(ndims == 2);

   hsize_t dims[ndims];
   errf = H5Sget_simple_extent_dims(dspace_id, dims, NULL);
   assert(errf >= 0);

   // hdf5 is row-major, while mfem::DenseMatrix is column major. we load the transpose.
   value.SetSize(dims[1], dims[0]);
   hid_t dataType = hdf5_utils::GetType(value(0, 0));
   H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Write());

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

void WriteDataset(hid_t &source, std::string dataset, const DenseMatrix &value)
{
   herr_t errf = 0;

   hid_t dataType = hdf5_utils::GetType(value(0,0));
   hsize_t dims[2];
   // hdf5 is row-major, while mfem::DenseMatrix is column major. we save the transpose.
   dims[0] = value.NumCols();
   dims[1] = value.NumRows();

   hid_t dspace_id = H5Screate_simple(2, dims, NULL);
   assert(dspace_id >= 0);

   hid_t dset_id = H5Dcreate2(source, dataset.c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(dset_id >= 0);
   
   errf = H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Read());
   assert(errf >= 0);

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

// This currently only reads the first item. Do not use it.
void ReadDataset(hid_t &source, std::string dataset, std::vector<std::string> &value)
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

SparseMatrix* ReadSparseMatrix(hid_t &source, std::string matrix_name)
{
   herr_t errf = 0;
   hid_t grp_id;
   grp_id = H5Gopen2(source, matrix_name.c_str(), H5P_DEFAULT);
   assert(grp_id >= 0);

   Array<int> i_read, j_read, size;
   Array<double> data_read;

   ReadDataset(grp_id, "I", i_read);
   ReadDataset(grp_id, "J", j_read);
   ReadDataset(grp_id, "size", size);
   ReadDataset(grp_id, "data", data_read);
   const int n_entry = i_read.Size();

   assert(size.Size() == 2);
   assert((size[0] > 0) && (size[1] > 0));
   assert(i_read.Size() == size[0] + 1);
   assert(j_read.Size() == i_read[size[0]]);
   assert(data_read.Size() == i_read[size[0]]);

   // Need to transfer the ownership to avoid segfault or double-free.
   int *ip, *jp;
   double *vp;
   i_read.StealData(&ip);
   j_read.StealData(&jp);
   data_read.StealData(&vp);

   SparseMatrix *mat = new SparseMatrix(ip, jp, vp, size[0], size[1]);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   return mat;
}

void WriteSparseMatrix(hid_t &source, std::string matrix_name, SparseMatrix* mat)
{
   assert(mat->Finalized());

   herr_t errf = 0;
   hid_t grp_id;
   grp_id = H5Gcreate(source, matrix_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   const int nnz = mat->NumNonZeroElems();
   const int height = mat->Height();
   int *i_idx = mat->GetI();
   int *j_idx = mat->GetJ();
   double *data = mat->GetData();

   Array<int> i_write, j_write;
   Array<double> data_write;
   i_write.MakeRef(i_idx, height+1);
   j_write.MakeRef(j_idx, i_write[height]);
   data_write.MakeRef(data, i_write[height]);

   WriteDataset(grp_id, "I", i_write);
   WriteDataset(grp_id, "J", j_write);
   WriteDataset(grp_id, "data", data_write);

   Array<int> size(2);
   size[0] = mat->NumRows();
   size[1] = mat->NumCols();
   WriteDataset(grp_id, "size", size);

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

BlockMatrix* ReadBlockMatrix(hid_t &source, std::string matrix_name,
                             const Array<int> &block_offsets)
{
   herr_t errf = 0;
   hid_t grp_id;
   grp_id = H5Gopen2(source, matrix_name.c_str(), H5P_DEFAULT);
   assert(grp_id >= 0);

   Array<int> size, row_offsets, col_offsets;
   Array2D<int> zero_blocks;
   ReadDataset(grp_id, "size", size);
   ReadDataset(grp_id, "row_offsets", row_offsets);
   ReadDataset(grp_id, "col_offsets", col_offsets);
   ReadDataset(grp_id, "zero_blocks", zero_blocks);
   assert(size.Size() == 2);
   assert(row_offsets.Size() == size[0] + 1);
   assert(col_offsets.Size() == size[1] + 1);
   assert(zero_blocks.NumRows() == size[0]);
   assert(zero_blocks.NumCols() == size[1]);

   /*
      NOTE: BlockMatrix does not own/copy offset Array in its construction.
      If we use row_offsets/col_offsets to construct the BlockMatrix,
      then we need to carry these variables for the whole lifetime of this BlockMatrix.
      We decided to rather use an input offset Array,
      and check the size is correct.
      This currently only support square block matrix.
   */
   assert(row_offsets.Size() == block_offsets.Size());
   assert(col_offsets.Size() == block_offsets.Size());
   for (int i = 0; i < block_offsets.Size(); i++)
   {
      assert(row_offsets[i] == block_offsets[i]);
      assert(col_offsets[i] == block_offsets[i]);
   }

   BlockMatrix *mat = new BlockMatrix(block_offsets);
   mat->owns_blocks = true;

   std::string block_name;
   for (int i = 0; i < size[0]; i++)
      for (int j = 0; j < size[1]; j++)
      {
         if (zero_blocks(i, j)) continue;

         block_name = "block_" + std::to_string(i) + "_" + std::to_string(j);
         mat->SetBlock(i, j, ReadSparseMatrix(grp_id, block_name));
      }
   mat->Finalize();

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   return mat;
}

void WriteBlockMatrix(hid_t &source, std::string matrix_name, BlockMatrix* mat)
{
   herr_t errf = 0;
   hid_t grp_id;
   grp_id = H5Gcreate(source, matrix_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(grp_id >= 0);

   Array<int> size(2);
   size[0] = mat->NumRowBlocks();
   size[1] = mat->NumColBlocks();
   WriteDataset(grp_id, "size", size);

   Array<int> row_offsets = mat->RowOffsets();
   Array<int> col_offsets = mat->ColOffsets();
   WriteDataset(grp_id, "row_offsets", row_offsets);
   WriteDataset(grp_id, "col_offsets", col_offsets);

   Array2D<int> zero_blocks(size[0], size[1]);
   for (int i = 0; i < size[0]; i++)
      for (int j = 0; j < size[1]; j++)
         zero_blocks(i, j) = (mat->IsZeroBlock(i, j) || (mat->GetBlock(i,j).NumNonZeroElems() == 0));
   WriteDataset(grp_id, "zero_blocks", zero_blocks);

   std::string block_name;
   for (int i = 0; i < size[0]; i++)
      for (int j = 0; j < size[1]; j++)
      {
         if (zero_blocks(i, j)) continue;

         block_name = "block_" + std::to_string(i) + "_" + std::to_string(j);
         WriteSparseMatrix(grp_id, block_name, &(mat->GetBlock(i, j)));
      }

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void ReadDataset(hid_t &source, std::string dataset, Vector &value)
{
   herr_t errf = 0;
   
   hid_t dset_id = H5Dopen(source, dataset.c_str(), H5P_DEFAULT);
   assert(dset_id >= 0);

   hid_t dspace_id = H5Dget_space(dset_id);
   int ndims = H5Sget_simple_extent_ndims(dspace_id);
   assert(ndims == 1);

   hsize_t dims[ndims];
   errf = H5Sget_simple_extent_dims(dspace_id, dims, NULL);
   assert(errf >= 0);

   value.SetSize(dims[0]);
   hid_t dataType = hdf5_utils::GetType(value[0]);
   H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Write());

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

void WriteDataset(hid_t &source, std::string dataset, const Vector &value)
{
   herr_t errf = 0;
   assert(value.Size() > 0);

   hid_t dataType = hdf5_utils::GetType(value[0]);
   hsize_t dims[1];
   dims[0] = value.Size();

   hid_t dspace_id = H5Screate_simple(1, dims, NULL);
   assert(dspace_id >= 0);

   hid_t dset_id = H5Dcreate2(source, dataset.c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(dset_id >= 0);

   errf = H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Read());
   assert(errf >= 0);

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

}
