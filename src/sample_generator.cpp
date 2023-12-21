// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "sample_generator.hpp"
#include "etc.hpp"

using namespace mfem;
using namespace std;

SampleGenerator::SampleGenerator(MPI_Comm comm)
{
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &proc_rank);

   sample_dir = config.GetOption<std::string>("sample_generation/file_path/directory", ".");
   std::string problem_name = config.GetOption<std::string>("parameterized_problem/name", "sample");
   sample_prefix = config.GetOption<std::string>("sample_generation/file_path/prefix", problem_name);

   file_offset = config.GetOption<int>("sample_generation/file_path/offset", 0);

   param_list_str = "sample_generation/parameters";
   YAML::Node param_list = config.FindNode(param_list_str);
   if (!param_list) mfem_error("SampleGenerator - cannot find the parameter list!\n");

   num_sampling_params = param_list.size();
   params.SetSize(num_sampling_params);

   for (int p = 0; p < num_sampling_params; p++)
   {
      std::string param_key = config.GetRequiredOptionFromDict<std::string>("key", param_list[p]);
      std::string param_type = config.GetRequiredOptionFromDict<std::string>("type", param_list[p]);

      if (param_type == "double")         params[p] = new DoubleParam(param_key, param_list[p]);
      else if (param_type == "integer")   params[p] = new IntegerParam(param_key, param_list[p]);
      else if (param_type == "filename")  params[p] = new FilenameParam(param_key, param_list[p]);
      else mfem_error("SampleGenerator: Unknown parameter type!\n");
   }  // for (int p = 0; p < num_sampling_params; p++)

   // NOTE: option used for BasisGenerator. must be sufficient to cover at least total_samples.
   // As different geometric configurations are used as samples,
   // the number of snapshots can exceed total_samples.
   max_num_snapshots = config.GetOption<int>("sample_generation/maximum_number_of_snapshots", 100);

   // Report frequency
   report_freq = config.GetOption<int>("sample_generation/report_frequency", 1);

   // Initially no snapshot generator.
   snapshot_generators.SetSize(0);
   snapshot_options.SetSize(0);

   // BasisGenerator options.
   update_right_SV = config.GetOption<bool>("basis/svd/update_right_sv", false);
   save_sv = config.GetOption<bool>("basis/svd/save_spectrum", false);
}

SampleGenerator::~SampleGenerator()
{
   DeletePointers(params);
   DeletePointers(snapshot_generators);
   DeletePointers(snapshot_options);
}

void SampleGenerator::SetParamSpaceSizes()
{
   YAML::Node param_list = config.FindNode(param_list_str);
   if (!param_list) mfem_error("SampleGenerator - cannot find the parameter list!\n");
   assert(num_sampling_params > 0);
   assert(params.Size() == num_sampling_params);

   sampling_sizes.SetSize(num_sampling_params);
   sampling_sizes = -1;

   for (int p = 0; p < num_sampling_params; p++)
   {
      sampling_sizes[p] = config.GetRequiredOptionFromDict<int>("sample_size", param_list[p]);
      params[p]->SetSize(sampling_sizes[p]);
   }

   total_samples = 1;
   for (int p = 0; p < num_sampling_params; p++)
      total_samples *= sampling_sizes[p];

   // This does not need the actual samples. distributing only indexes.
   DistributeSamples();
}

void SampleGenerator::DistributeSamples()
{
   sample_offsets.SetSize(num_procs + 1);

   int quotient = total_samples / num_procs;
   sample_offsets = quotient;
   sample_offsets[0] = 0;

   int remainder = total_samples % num_procs;
   for (int r = 0; r < remainder; r++)
      sample_offsets[1+r] += 1;

   sample_offsets.PartialSum();

   assert(sample_offsets[0] == 0);
   assert(sample_offsets[num_procs] == total_samples);
}

const int SampleGenerator::GetSampleIndex(const Array<int> &index)
{
   assert(index.Size() == num_sampling_params);

   // compute global index, row-major.
   int global_idx = index[0];
   for (int p = 1; p < num_sampling_params; p++)
   {
      global_idx *= sampling_sizes[p];
      global_idx += index[p];
   }

   assert((global_idx >= 0) && (global_idx < total_samples));
   return global_idx;
}

const Array<int> SampleGenerator::GetSampleIndex(const int &index)
{
   Array<int> nested_idx(num_sampling_params);

   // compute nested local index, row-major.
   int tmp_idx = index;
   for (int p = num_sampling_params - 1; p >= 0; p--)
   {
      int local_idx = tmp_idx % sampling_sizes[p];
      assert(((local_idx >= 0) && (local_idx < sampling_sizes[p])));

      nested_idx[p] = local_idx;
      tmp_idx -= local_idx;
      tmp_idx /= sampling_sizes[p];
   }

   return nested_idx;
}

void SampleGenerator::SetSampleParams(const int &index)
{
   assert(params.Size() == num_sampling_params);

   const Array<int> nested_idx = GetSampleIndex(index);

   for (int p = 0; p < num_sampling_params; p++)
      params[p]->SetParam(nested_idx[p], config);
}

const std::string SampleGenerator::GetSamplePath(const int &idx, const std::string& prefix)
{
   std::string full_path = sample_dir;
   full_path += "/sample" + std::to_string(idx) + "_";
   if (prefix != "")
      full_path += prefix;
   else
      full_path += sample_prefix;

   return full_path;
}

void SampleGenerator::SaveSnapshot(BlockVector *U_snapshots, std::vector<std::string> &snapshot_basis_tags)
{
   assert(U_snapshots->NumBlocks() == snapshot_basis_tags.size());

   for (int s = 0; s < snapshot_basis_tags.size(); s++)
   {
      if (!basis_tag2idx.count(snapshot_basis_tags[s]))
      {
         const int fom_vdofs = U_snapshots->BlockSize(s);
         AddSnapshotGenerator(fom_vdofs, GetSamplePrefix(), snapshot_basis_tags[s]);
      }

      int index = basis_tag2idx[snapshot_basis_tags[s]];
      bool addSample = snapshot_generators[index]->takeSample(U_snapshots->GetBlock(s).GetData(), 0.0, 0.01);
      assert(addSample);
   }
}

void SampleGenerator::AddSnapshotGenerator(const int &fom_vdofs, const std::string &prefix, const std::string &basis_tag)
{
   const std::string filename = GetBaseFilename(prefix, basis_tag);

   snapshot_options.Append(new CAROM::Options(fom_vdofs, max_num_snapshots, 1, update_right_SV));
   snapshot_generators.Append(new CAROM::BasisGenerator(*(snapshot_options.Last()), incremental, filename));

   basis_tag2idx[basis_tag] = basis_tags.size();
   basis_tags.push_back(basis_tag);

   int size = snapshot_options.Size();
   assert(snapshot_generators.Size() == size);
   assert(basis_tag2idx.size() == size);
   assert(basis_tags.size() == size);
}

void SampleGenerator::WriteSnapshots()
{
   assert(snapshot_generators.Size() > 0);
   for (int s = 0; s < snapshot_generators.Size(); s++)
   {
      assert(snapshot_generators[s]);
      snapshot_generators[s]->writeSnapshot();
   }
}

void SampleGenerator::ReportStatus(const int &sample_idx)
{
   if (sample_idx % report_freq != 0) return;

   printf("==========  SampleGenerator Status  ==========\n");
   printf("%d-th sample is collected.\n", sample_idx);
   printf("Basis tags: %ld\n", basis_tags.size());
   printf("%20.20s\t# of snapshots\n", "Basis tags");
   for (int k = 0; k < basis_tags.size(); k++)
      printf("%20.20s\t%d\n", basis_tags[k].c_str(), snapshot_generators[k]->getNumSamples());
   printf("==============================================\n");
}

void SampleGenerator::FormReducedBasis(const std::string &basis_prefix,
                                       const std::string &basis_tag,
                                       const std::vector<std::string> &file_list,
                                       const int &num_basis)
{
   // Get dimension from the first snapshot file.
   const int fom_num_vdof = GetDimFromSnapshots(file_list[0]);

   std::string basis_name = GetBaseFilename(basis_prefix, basis_tag);

   // Append snapshot generator.
   AddSnapshotGenerator(fom_num_vdof, basis_prefix, basis_tag);
   CAROM::BasisGenerator *basis_generator = snapshot_generators.Last();

   for (int s = 0; s < file_list.size(); s++)
      basis_generator->loadSamples(file_list[s], "snapshot");

   basis_generator->endSamples(); // save the merged basis file
   SaveSV(basis_generator, basis_name, num_basis);
}

const int SampleGenerator::GetDimFromSnapshots(const std::string &filename)
{
   CAROM::BasisReader d_basis_reader(filename);
   // time is currently always 0.0
   return d_basis_reader.getDim("snapshot", 0.0);
}

void SampleGenerator::SaveSV(CAROM::BasisGenerator *basis_generator, const std::string& prefix, const int& num_basis)
{
   if (!save_sv) return;
   assert(basis_generator != NULL);

   const CAROM::Vector *rom_sv = basis_generator->getSingularValues();
   printf("Singular values: ");
   for (int d = 0; d < rom_sv->dim(); d++)
      printf("%.3E\t", rom_sv->item(d));
   printf("\n");

   double coverage = 0.0;
   double total = 0.0;

   for (int d = 0; d < rom_sv->dim(); d++)
   {
      if (d == num_basis) coverage = total;
      total += rom_sv->item(d);
   }
   if (rom_sv->dim() == num_basis) coverage = total;
   coverage /= total;
   printf("Coverage: %.7f%%\n", coverage * 100.0);

   // TODO: hdf5 format + parallel case.
   std::string filename = prefix + "_sv.txt";
   CAROM::PrintVector(*rom_sv, filename);
}