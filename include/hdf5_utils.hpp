// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef HDF5_UTILS_HPP
#define HDF5_UTILS_HPP

#include "mfem.hpp"
#include "hdf5.h"
#include "linalg_utils.hpp"

namespace mfem
{

enum IntegratorType
{
   DOMAIN,
   INTERIORFACE,
   BDRFACE,
   INTERFACE,
   NUM_INTEG_TYPE
};

struct SampleInfo
{
   /*
      - For DomainIntegrator: element index
      - For BdrFaceIntegrator: boundary element index
      - For InteriorFaceIntegrator: face index
      - For InterfaceIntegrator: interface info index
   */
   int el;

   int qp;         // quadrature point
   double qw;      // quadrature weight
};

struct BasisTag
{
   /* component mesh name */
   std::string comp = "";
   /* variable name, if separate basis is used */
   std::string var = "";

   BasisTag() {}

   BasisTag(const std::string &comp_, const std::string &var_="")
      : comp(comp_), var(var_) {}

   const std::string print() const
   {
      std::string tag = comp;
      if (var != "")
         tag += "_" + var;
      return tag;
   }

   bool operator==(const BasisTag &tag) const
   {
      return ((comp == tag.comp) && (var == tag.var));
   }

   bool operator<(const BasisTag &tag) const
   {
      if (comp == tag.comp)
         return (var < tag.var);

      return (comp < tag.comp);
   }

   BasisTag& operator=(const BasisTag &tag)
   {
      comp = tag.comp;
      var = tag.var;
      return *this;
   }
};

}

using namespace mfem;

namespace hdf5_utils
{

inline hid_t GetType(int) { return (H5T_NATIVE_INT); }
inline hid_t GetType(double) { return (H5T_NATIVE_DOUBLE); }
inline hid_t GetType(bool) { return (H5T_NATIVE_HBOOL); }

hid_t GetNativeType(hid_t type);

int GetDatasetSize(hid_t &source, std::string dataset, hsize_t* &dims);

void ReadAttribute(hid_t &source, std::string attribute, std::string &value);
void WriteAttribute(hid_t &source, std::string attribute, const std::string &value);

void ReadAttribute(hid_t &source, std::string attribute, BasisTag &value);
void WriteAttribute(hid_t &source, std::string attribute, const BasisTag &value);

template <typename T>
void ReadAttribute(hid_t &source, std::string attribute, T &value) {
   herr_t status;
   hid_t attr;
   hid_t attrType = hdf5_utils::GetType(value);

   attr = H5Aopen_name(source, attribute.c_str());
   status = H5Aread(attr, attrType, &value);
   assert(status >= 0);
   H5Aclose(attr);
}

template <typename T>
void WriteAttribute(hid_t &dest, std::string attribute, const T &value) {
   hid_t attr, status;
   hid_t attrType = hdf5_utils::GetType(value);
   hid_t dataspaceId = H5Screate(H5S_SCALAR);

   attr = H5Acreate(dest, attribute.c_str(), attrType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
   assert(attr >= 0);
   status = H5Awrite(attr, attrType, &value);
   assert(status >= 0);
   H5Aclose(attr);
   H5Sclose(dataspaceId);
}

template <typename T>
void ReadDataset(hid_t &source, std::string dataset, Array<T> &value)
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
   if (dims[0] > 0)
   {
      hid_t dataType = hdf5_utils::GetType(value[0]);
      H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Write());
   }

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

template <typename T>
void ReadDataset(hid_t &source, std::string dataset, Array2D<T> &value)
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

   value.SetSize(dims[0], dims[1]);
   if ((dims[0] == 0) || (dims[1] == 0))
   {
      errf = H5Dclose(dset_id);
      assert(errf >= 0);

      return;
   }

   hid_t dataType = hdf5_utils::GetType(value(0, 0));
   H5Dread(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.GetRow(0));

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

// This currently only reads the first item. Do not use it.
void ReadDataset(hid_t &source, std::string dataset, std::vector<std::string> &value);

template <typename T>
void WriteDataset(hid_t &source, std::string dataset, const Array<T> &value)
{
   herr_t errf = 0;

   T tmp;
   hid_t dataType = hdf5_utils::GetType(tmp);
   hsize_t dims[1];
   dims[0] = value.Size();

   hid_t dspace_id = H5Screate_simple(1, dims, NULL);
   assert(dspace_id >= 0);

   hid_t dset_id = H5Dcreate2(source, dataset.c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(dset_id >= 0);

   if (dims[0] > 0)
   {
      errf = H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.Read());
      assert(errf >= 0);
   }

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

template <typename T>
void WriteDataset(hid_t &source, std::string dataset, const Array2D<T> &value)
{
   herr_t errf = 0;

   hid_t dataType = hdf5_utils::GetType(value(0,0));
   hsize_t dims[2];
   dims[0] = value.NumRows();
   dims[1] = value.NumCols();

   hid_t dspace_id = H5Screate_simple(2, dims, NULL);
   assert(dspace_id >= 0);

   hid_t dset_id = H5Dcreate2(source, dataset.c_str(), dataType, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   assert(dset_id >= 0);
   
   errf = H5Dwrite(dset_id, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, value.GetRow(0));
   assert(errf >= 0);

   errf = H5Dclose(dset_id);
   assert(errf >= 0);
}

SparseMatrix* ReadSparseMatrix(hid_t &source, std::string matrix_name);
void WriteSparseMatrix(hid_t &source, std::string matrix_name, SparseMatrix* mat);

BlockMatrix* ReadBlockMatrix(hid_t &source, std::string matrix_name,
                             const Array<int> &block_offsets);
void WriteBlockMatrix(hid_t &source, std::string matrix_name, BlockMatrix* mat);

void ReadDataset(hid_t &source, std::string dataset, Vector &value);
void WriteDataset(hid_t &source, std::string dataset, const Vector &value);

void ReadDataset(hid_t &source, std::string dataset, DenseMatrix &value);
void WriteDataset(hid_t &source, std::string dataset, const DenseMatrix &value);

void ReadDataset(hid_t &source, std::string dataset, DenseTensor &value);
void WriteDataset(hid_t &source, std::string dataset, const DenseTensor &value);

void ReadDataset(hid_t &source, std::string dataset, MatrixBlocks &value);
void WriteDataset(hid_t &source, std::string dataset, const MatrixBlocks &value);

void ReadDataset(hid_t &source, std::string dataset, const IntegratorType type, Array<SampleInfo> &value);
void WriteDataset(hid_t &source, std::string dataset, const IntegratorType type, const Array<SampleInfo> &value);

inline bool pathExists(hid_t id, const std::string& path)
{
  return H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0;
}

}

#endif
