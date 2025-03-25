// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

#include "mfem.hpp"
#include "parameter.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg_utils.hpp"
#include "rom_handler.hpp"

using namespace mfem;

enum SampleGeneratorType
{
   BASE,
   RANDOM,
   NUM_SAMPLE_GEN_TYPE
};

struct PortTag
{
   std::string Mesh1;
   std::string Mesh2;
   int Attr1;
   int Attr2;
};

bool operator<(const PortTag &tag1, const PortTag &tag2);

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
   std::vector<BasisTag> basis_tags;
   std::map<BasisTag, int> basis_tag2idx;

   /* snapshot pairs per interface port, for nonlinear interface EQP */
   std::vector<PortTag> port_tags;
   std::map<PortTag, int> port_tag2idx;
   Array<Array2D<int> *> port_colidxs;

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
   const std::string GetBaseFilename(const std::string &prefix, const BasisTag &basis_tag)
   { return prefix + "_" + basis_tag.print(); }

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

   /*
      Save each block of U_snapshots according to snapshot_basis_tags.
      Number of blocks in U_snapshots must be equal to the size of snapshot_basis_tags.
      The appended column indices of each basis tag are stored in col_idxs.
   */
   void SaveSnapshot(BlockVector *U_snapshots, std::vector<BasisTag> &snapshot_basis_tags, Array<int> &col_idxs);
   void SaveSnapshotPorts(TopologyHandler *topol_handler, const Array<int> &col_idxs);
   void AddSnapshotGenerator(const int &fom_vdofs, const std::string &prefix, const BasisTag &basis_tag);
   void WriteSnapshots();
   void WriteSnapshotPorts();
   std::shared_ptr<const CAROM::Matrix> LookUpSnapshot(const BasisTag &basis_tag);
   Array2D<int>* LookUpSnapshotPortColOffsets(const PortTag &port_tag);

   void ReportStatus(const int &sample_idx);

   /*
      Collect snapshot matrices from the file list to the specified basis tag.
   */
   void CollectSnapshotsByBasis(const std::string &basis_prefix,
                              const BasisTag &basis_tag,
                              const std::vector<std::string> &file_list);
   /*
      Collect snapshot matrices from the file list to the specified port tag file.
   */
   void CollectSnapshotsByPort(const std::string &basis_prefix,
                               const std::string &port_tag_file);
   /*
      Perform SVD over snapshot for basis_tag.
      Calculate the energy fraction for num_basis.
      CollectSnapshots must be executed before this.
   */
   void FormReducedBasis(const std::string &basis_prefix);

private:
   const int GetDimFromSnapshots(const std::string &filename);
   // Save all singular value spectrum. Calculate the coverage for ref_num_basis (optional).
   void SaveSV(CAROM::BasisGenerator *basis_generator, const std::string& prefix, const int &ref_num_basis = -1);

   const int GetSnapshotOffset(const std::string &comp);

};

#endif
