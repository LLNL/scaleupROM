// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

#include "mfem.hpp"
#include "parameter.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg_utils.hpp"

using namespace mfem;

enum SampleGeneratorType
{
   BASE,
   RANDOM,
   NUM_SAMPLE_GEN_TYPE
};

class SampleGenerator
{
protected:
   int num_procs;
   int proc_rank;
   Array<int> sample_offsets;

   // input path for parameter list
   std::string param_list_str;

   std::size_t num_sampling_params;

   Array<int> sampling_sizes;
   int total_samples;
   int report_freq = -1;

   Array<Parameter *> params;

   // file path
   std::string sample_dir = ".";
   std::string sample_prefix;
   int file_offset = 0;

   // snapshot generators: purely for saving snapshots. will not execute svd.
   int max_num_snapshots = 100;
   bool save_sv = false;
   bool update_right_SV = false;
   const bool incremental = false;
   Array<CAROM::Options*> snapshot_options;
   Array<CAROM::BasisGenerator*> snapshot_generators;
   // each snapshot will be sorted out by its basis tag.
   std::vector<std::string> basis_tags;
   std::map<std::string, int> basis_tag2idx;

public:
   SampleGenerator(MPI_Comm comm);

   virtual ~SampleGenerator();

   virtual SampleGeneratorType GetType() { return BASE; }

   const int GetNumSampleParams() { return num_sampling_params; }
   const Array<int> GetSampleSizes() { return sampling_sizes; }
   const int GetTotalSampleSize() { return total_samples; }
   const int GetProcRank() { return proc_rank; }
   const int GetFileOffset() { return file_offset; }
   Parameter* GetParam(const int &k) { return params[k]; }
   const std::string GetSamplePrefix()
   { return sample_dir + "/" + sample_prefix + "_sample"; }
   const std::string GetBaseFilename(const std::string &prefix, const std::string &basis_tag)
   { return prefix + "_" + basis_tag; }

   // Generate parameter space as listed in sample_generation/problem_name.
   virtual void SetParamSpaceSizes();

   virtual void SetSampleParams(const int &index);
   virtual void SetSampleParams(const Array<int> &index)
   { SetSampleParams(GetSampleIndex(index)); }

   // Determine the given index is assigned to the current process.
   void DistributeSamples();
   virtual const int GetSampleIndex(const Array<int> &index);
   virtual const Array<int> GetSampleIndex(const int &index);
   bool IsMyJob(const Array<int> &index)
   { return IsMyJob(GetSampleIndex(index)); }
   bool IsMyJob(const int &index)
   { return ((index >= sample_offsets[proc_rank]) && (index < sample_offsets[proc_rank+1])); }

   const std::string GetSamplePath(const int& idx, const std::string &prefix = "");

   void SaveSnapshot(BlockVector *U_snapshots, std::vector<std::string> &snapshot_basis_tags);
   void AddSnapshotGenerator(const int &fom_vdofs, const std::string &prefix, const std::string &basis_tag);
   void WriteSnapshots();
   const CAROM::Matrix* LookUpSnapshot(const std::string &basis_tag);

   void ReportStatus(const int &sample_idx);

   // Perform SVD over snapshot for basis_tag. Calculate the coverage for ref_num_basis (optional).
   void FormReducedBasis(const std::string &basis_prefix,
                         const std::string &basis_tag,
                         const std::vector<std::string> &file_list,
                         const int &ref_num_basis = -1);

private:
   const int GetDimFromSnapshots(const std::string &filename);
   // Save all singular value spectrum. Calculate the coverage for ref_num_basis (optional).
   void SaveSV(CAROM::BasisGenerator *basis_generator, const std::string& prefix, const int &ref_num_basis = -1);

};

#endif
