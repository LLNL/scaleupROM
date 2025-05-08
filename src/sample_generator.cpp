// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "sample_generator.hpp"
#include "hdf5_utils.hpp"
#include "etc.hpp"
#include "utils/mpi_utils.h"  // this is from libROM/utils.

using namespace mfem;
using namespace std;

bool operator<(const PortTag &tag1, const PortTag &tag2)
{
   if (tag1.Mesh1 != tag2.Mesh1)
      return (tag1.Mesh1 < tag2.Mesh1);
   
   if (tag1.Mesh2 != tag2.Mesh2)
      return (tag1.Mesh2 < tag2.Mesh2);

   if (tag1.Attr1 != tag2.Attr1)
      return (tag1.Attr1 < tag2.Attr1);

   return (tag1.Attr2 == tag2.Attr2) ? false : (tag1.Attr2 < tag2.Attr2);
}

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
   DeletePointers(port_colidxs);
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

void SampleGenerator::SaveSnapshot(BlockVector *U_snapshots, std::vector<BasisTag> &snapshot_basis_tags, Array<int> &col_idxs)
{ cout<<11<<endl;
   assert(U_snapshots->NumBlocks() == snapshot_basis_tags.size());
cout<<12<<endl;

   col_idxs.SetSize(U_snapshots->NumBlocks());
   cout<<13<<endl;

   /* add snapshots according to their tags */
   for (int s = 0; s < snapshot_basis_tags.size(); s++)
   {
cout<<s<<endl;
cout<<14<<endl;

      /* if the tag was never seen before, create a new snapshot generator */
      if (!basis_tag2idx.count(snapshot_basis_tags[s]))
      {
cout<<15<<endl;

         const int fom_vdofs = U_snapshots->BlockSize(s);
cout<<16<<endl;

         AddSnapshotGenerator(fom_vdofs, GetSamplePrefix(), snapshot_basis_tags[s]);
      }

      /* add the snapshot into the corresponding snapshot generator */
      int index = basis_tag2idx[snapshot_basis_tags[s]];
      cout<<17<<endl;

      assert(snapshot_generators != nullptr);
size_t num_generators = sizeof(snapshot_generators) / sizeof(snapshot_generators[0]);
assert(index >= 0 && index < num_generators);
assert(snapshot_generators[index] != nullptr);
assert(U_snapshots != nullptr);
assert(s >= 0 && s < U_snapshots->NumBlocks());

assert(U_snapshots != nullptr);
assert(s >= 0 && s < U_snapshots->NumBlocks());

mfem::Vector &block = U_snapshots->GetBlock(s);
assert(block.Size() > 0);  // Size should be positive

double *data = block.GetData();
assert(data != nullptr);  // Check data pointer is not null

// Parameters
double fraction = 0.001;  // e.g., use 10% of the data
int full_size = block.Size();
int partial_size = static_cast<int>(fraction * full_size);

// Clamp in case full_size is small
partial_size = std::max(1, partial_size);

// Copy the first partial_size elements into a new vector
mfem::Vector partial_vec(partial_size);
for (int i = 0; i < partial_size; ++i)
{
    partial_vec[i] = block[i];
}

// Pass the reduced vector to takeSample()
cout<<"ready to test with fraction: "<<fraction<<endl;
cout<<"length of test vector: "<<partial_size<<endl;

for (int i = 0; i < partial_size; ++i) {
   if (!std::isfinite(partial_vec[i])) {
       std::cerr << "Non-finite value at " << i << ": " << partial_vec[i] << std::endl;
   }
}

mfem::Vector test_vec(10);
for (int i = 0; i < 10; ++i) test_vec[i] = i;
cout<<"testing with good vector "<<endl;
bool goodaddSample = snapshot_generators[index]->takeSample(test_vec.GetData());

cout<<"testing with try catch "<<endl;

try {
   bool addSample = snapshot_generators[index]->takeSample(partial_vec.GetData());
} catch (const std::exception &e) {
   std::cerr << "Exception in takeSample: " << e.what() << std::endl;
}

bool testaddSample = snapshot_generators[index]->takeSample(partial_vec.GetData());

// Optional: print values for deeper debugging
std::cout << "testaddSample " << testaddSample<< std::endl;
std::cout << "Block " << s << " size: " << block.Size() << std::endl;
std::cout << "Data pointer: " << static_cast<void*>(data) << std::endl;

bool addSample = snapshot_generators[index]->takeSample(U_snapshots->GetBlock(s).GetData());
cout<<18<<endl;
      
      assert(addSample);
      cout<<19<<endl;

      /* save the column index in each snapshot matrix, for port data. */
      /* 0-based index */
      col_idxs[s] = snapshot_generators[index]->getNumSamples() - 1;
   }
}

void SampleGenerator::SaveSnapshotPorts(TopologyHandler *topol_handler, const Array<int> &col_idxs)
{
   assert(topol_handler);
   const int numSub = topol_handler->GetNumSubdomains();
   const int numPorts = topol_handler->GetNumPorts();

   assert(col_idxs.Size() % numSub == 0);
   const int num_var = col_idxs.Size() / numSub;

   /* iterate every port */
   for (int p = 0; p < numPorts; p++)
   {
      const PortInfo *port = topol_handler->GetPortInfo(p);

      /*
         We save the snapshot ports by component names.
         Variable separation does not matter at this point..
      */
      int c1 = topol_handler->GetMeshType(port->Mesh1);
      int c2 = topol_handler->GetMeshType(port->Mesh2);
      std::string mesh1, mesh2;
      mesh1 = topol_handler->GetComponentName(c1);
      mesh2 = topol_handler->GetComponentName(c2);

      /*
         note the mesh names are not necessarily the same as the subdomain names
         depending on the train mode.
      */
      PortTag tag = {.Mesh1 = mesh1, .Mesh2 = mesh2, .Attr1 = port->Attr1, .Attr2 = port->Attr2};

      /* if the port was never seen, create a new collector */
      if (!port_tag2idx.count(tag))
      {
         port_tag2idx[tag] = port_tags.size();
         port_tags.push_back(tag);
         port_colidxs.Append(new Array2D<int>);
      }
      const int index = port_tag2idx[tag];

      int col1, col2;
      /*
         column indices in snapshot matrices must be identical for all variables,
         whether variables are separated for training or not.
         We thus pick the column index of the first variable.
      */
      col1 = col_idxs[num_var * (port->Mesh1)];
      col2 = col_idxs[num_var * (port->Mesh2)];
      for (int v = 0; v < num_var; v++)
      {
         assert(col1 == col_idxs[v + num_var * (port->Mesh1)]);
         assert(col2 == col_idxs[v + num_var * (port->Mesh2)]);
      }

      int row_idx = port_colidxs[index]->NumRows();
      port_colidxs[index]->SetSize(row_idx + 1, 2);
      (*port_colidxs[index])(row_idx, 0) = col1;
      (*port_colidxs[index])(row_idx, 1) = col2;
   }
}

void SampleGenerator::AddSnapshotGenerator(const int &fom_vdofs, const std::string &prefix, const BasisTag &basis_tag)
{
   const std::string filename = GetBaseFilename(prefix, basis_tag);

   snapshot_options.Append(new CAROM::Options(fom_vdofs, max_num_snapshots, 1, update_right_SV));
   snapshot_options.Last()->static_svd_preserve_snapshot = true;
   snapshot_options.Last()->setSingularValueTol(1e-17);
   snapshot_generators.Append(new CAROM::BasisGenerator(*(snapshot_options.Last()), incremental, filename, CAROM::Database::formats::HDF5_MPIO));

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

void SampleGenerator::WriteSnapshotPorts()
{
   const std::string filename = GetSamplePrefix() + ".port.h5";

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   /* this is the path to all associated snapshot matrices */
   char sample_path[PATH_MAX];
   realpath(GetSamplePrefix().c_str(), sample_path);
   hdf5_utils::WriteAttribute(file_id, "sample_prefix", std::string(sample_path));
   
   /* write port information */
   hdf5_utils::WriteAttribute(file_id, "number_of_ports", (int) port_tags.size());
   for (int p = 0; p < port_tags.size(); p++)
   {
      hid_t grp_id;
      grp_id = H5Gcreate(file_id, std::to_string(p).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(grp_id >= 0);

      hdf5_utils::WriteAttribute(grp_id, "Mesh1", port_tags[p].Mesh1);
      hdf5_utils::WriteAttribute(grp_id, "Mesh2", port_tags[p].Mesh2);
      hdf5_utils::WriteAttribute(grp_id, "Attr1", port_tags[p].Attr1);
      hdf5_utils::WriteAttribute(grp_id, "Attr2", port_tags[p].Attr2);
      hdf5_utils::WriteDataset(grp_id, "col_idxs", *(port_colidxs[p]));

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   /* write basis tag list */
   hdf5_utils::WriteAttribute(file_id, "number_of_basistags", (int) basis_tags.size());
   for (int b = 0; b < basis_tags.size(); b++)
      hdf5_utils::WriteAttribute(file_id, std::string("basistag" + std::to_string(b)).c_str(), basis_tags[b]);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

std::shared_ptr<const CAROM::Matrix> SampleGenerator::LookUpSnapshot(const BasisTag &basis_tag)
{
   assert(snapshot_generators.Size() > 0);
   assert(snapshot_generators.Size() == basis_tags.size());

   int idx = -1;
   for (int k = 0; k < basis_tags.size(); k++)
      if (basis_tags[k] == basis_tag)
      {
         idx = k;
         break;
      }

   if (idx < 0)
   {
      printf("basis tag: %s\n", basis_tag.print().c_str());
      mfem_error("SampleGenerator::LookUpSnapshot- basis tag does not exist in snapshot list!\n");
   }

   return snapshot_generators[idx]->getSnapshotMatrix();
}

Array2D<int>* SampleGenerator::LookUpSnapshotPortColOffsets(const PortTag &port_tag)
{
   assert(port_colidxs.Size() > 0);
   assert(port_colidxs.Size() == port_tags.size());

   int idx = -1;
   if (port_tag2idx.count(port_tag))
      idx = port_tag2idx[port_tag];

   if (idx < 0)
      mfem_error("SampleGenerator::LookUpSnapshotPortColOffsets- port tag does not exist in port list!\n");

   return port_colidxs[idx];
}

void SampleGenerator::ReportStatus(const int &sample_idx)
{
   if (sample_idx % report_freq != 0) return;
   if (proc_rank != 0) return;

   printf("==========  SampleGenerator Status  ==========\n");
   printf("%d-th sample is collected.\n", sample_idx);
   printf("Basis tags: %ld\n", basis_tags.size());
   printf("%20.20s\t# of snapshots\n", "Basis tags");
   for (int k = 0; k < basis_tags.size(); k++)
      printf("%20.20s\t%d\n", basis_tags[k].print().c_str(), snapshot_generators[k]->getNumSamples());
   printf("==============================================\n");
}

void SampleGenerator::CollectSnapshotsByBasis(const std::string &basis_prefix,
                                       const BasisTag &basis_tag,
                                       const std::vector<std::string> &file_list)
{
   // Get dimension from the first snapshot file.
   const int fom_num_vdof = GetDimFromSnapshots(file_list[0]);

   std::string basis_name = GetBaseFilename(basis_prefix, basis_tag);

   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will have to consider distribution by MultiBlockSolver.
   */
   int local_num_vdofs = CAROM::split_dimension(fom_num_vdof, MPI_COMM_WORLD);

   /* if the tag was never seen before, append a new snapshot generator */
   if (!basis_tag2idx.count(basis_tag))
      AddSnapshotGenerator(local_num_vdofs, basis_prefix, basis_tag);
   int index = basis_tag2idx[basis_tag];
   CAROM::BasisGenerator *basis_generator = snapshot_generators[index];

   for (int s = 0; s < file_list.size(); s++)
      basis_generator->loadSamples(file_list[s], "snapshot", 1e9, CAROM::Database::formats::HDF5_MPIO);
}

void SampleGenerator::CollectSnapshotsByPort(
   const std::string &basis_prefix, const std::string &port_tag_file)
{
   // TODO(kevin): we need to see how this impacts the parallel I/O..

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(port_tag_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   std::string sample_path;
   hdf5_utils::ReadAttribute(file_id, "sample_prefix", sample_path);
   printf("SampleGenerator: snapshot port prefix=%s\n", sample_path.c_str());

   int num_ports = -1;
   hdf5_utils::ReadAttribute(file_id, "number_of_ports", num_ports);

   for (int p = 0; p < num_ports; p++)
   {
      hid_t grp_id;
      grp_id = H5Gopen2(file_id, std::to_string(p).c_str(), H5P_DEFAULT);
      assert(grp_id >= 0);

      PortTag port_tag;
      hdf5_utils::ReadAttribute(grp_id, "Mesh1", port_tag.Mesh1);
      hdf5_utils::ReadAttribute(grp_id, "Mesh2", port_tag.Mesh2);
      hdf5_utils::ReadAttribute(grp_id, "Attr1", port_tag.Attr1);
      hdf5_utils::ReadAttribute(grp_id, "Attr2", port_tag.Attr2);

      /* if the tag was never seen before, create a new port tag */
      if (!port_tag2idx.count(port_tag))
      {
         port_tag2idx[port_tag] = port_tags.size();
         port_tags.push_back(port_tag);
         port_colidxs.Append(new Array2D<int>);
      }
      int idx = port_tag2idx[port_tag];

      /* if other snapshots are already collected, column indices must be offseted */
      int col_offset1 = GetSnapshotOffset(port_tag.Mesh1);
      int col_offset2 = GetSnapshotOffset(port_tag.Mesh2);

      Array2D<int> tmp_colidx;
      hdf5_utils::ReadDataset(grp_id, "col_idxs", tmp_colidx);

      int row_offset = port_colidxs[idx]->NumRows();
      port_colidxs[idx]->SetSize(row_offset + tmp_colidx.NumRows(), 2);
      for (int r = 0, r0 = row_offset; r < tmp_colidx.NumRows(); r++, r0++)
      {
         (*port_colidxs[idx])(r0, 0) = tmp_colidx(r, 0) + col_offset1;
         (*port_colidxs[idx])(r0, 1) = tmp_colidx(r, 1) + col_offset2;
      }

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }  // for (int p = 0; p < num_ports; p++)

   /* read snapshots from basis tag list */
   int num_basistag = -1;
   hdf5_utils::ReadAttribute(file_id, "number_of_basistags", num_basistag);
   assert(num_basistag > 0);
   BasisTag basis_tag;
   std::vector<std::string> snapshot_file(1);
   for (int b = 0; b < num_basistag; b++)
   {
      hdf5_utils::ReadAttribute(file_id, std::string("basistag" + std::to_string(b)).c_str(), basis_tag);
      snapshot_file[0] = GetBaseFilename(sample_path, basis_tag) + "_snapshot";
      CollectSnapshotsByBasis(basis_prefix, basis_tag, snapshot_file);
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

void SampleGenerator::FormReducedBasis(const std::string &basis_prefix)
{
   assert(snapshot_generators.Size() > 0);
   assert(snapshot_generators.Size() == basis_tags.size());

   const int num_basis_default = config.GetOption<int>("basis/number_of_basis", -1);
   int num_basis;
   std::string basis_name;

   // tag-specific optional inputs.
   YAML::Node basis_list = config.FindNode("basis/tags");
   for (int k = 0; k < snapshot_generators.Size(); k++)
   {
      assert(snapshot_generators[k]);
      assert(snapshot_generators[k]->getNumSamples() > 0);
      snapshot_generators[k]->endSamples();

      // if optional inputs are specified, parse them first.
      if (basis_list)
      {
         // Find if additional inputs are specified for basis_tags[p].
         YAML::Node basis_tag_input = config.LookUpFromDict("name", basis_tags[k].print(), basis_list);
         
         // If basis_tags[p] has additional inputs, parse them.
         // parse tag-specific number of basis.
         if (basis_tag_input)
            num_basis = config.GetOptionFromDict<int>("number_of_basis", num_basis_default, basis_tag_input);
         else
            num_basis = num_basis_default;
      }
      else
         // if additional inputs are not specified, use default number of basis.
         num_basis = num_basis_default;

      assert(num_basis > 0);
      basis_name = GetBaseFilename(basis_prefix, basis_tags[k]);
      SaveSV(snapshot_generators[k], basis_name, num_basis);
   }
}

const int SampleGenerator::GetDimFromSnapshots(const std::string &filename)
{
   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Currently we do not have a way to know full vdofs or distribute them, as MultiBlockSolver is not initialized.
      We might need to initialize MultiBlockSolver for TrainROM.
   */
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   Array<int> nrows(1);
   if (rank == 0)
   {
      hid_t file_id;
      herr_t errf = 0;
      std::string filename_ext(filename + ".000000");
      printf("\nOpening file %s.. ", filename_ext.c_str());
      file_id = H5Fopen(filename_ext.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file_id >= 0);

      hdf5_utils::ReadDataset(file_id, "snapshot_matrix_num_rows", nrows);
      assert(nrows[0] > 0);

      errf = H5Fclose(file_id);
      assert(errf >= 0);
      printf("Done!\n");
   }
   MPI_Bcast(nrows.GetData(), 1, MPI_INT, 0, MPI_COMM_WORLD);
   
   return nrows[0];
   // CAROM::BasisReader d_basis_reader(filename);
   // return d_basis_reader.getDim("snapshot");
}

void SampleGenerator::SaveSV(CAROM::BasisGenerator *basis_generator, const std::string& prefix, const int& ref_num_basis)
{
   if (!save_sv) return;
   assert(basis_generator != NULL);

   std::shared_ptr<const CAROM::Vector> rom_sv = basis_generator->getSingularValues();
   printf("Singular values: ");
   for (int d = 0; d < rom_sv->dim(); d++)
      printf("%.3E\t", rom_sv->item(d));
   printf("\n");

   double coverage = 0.0;
   double total = 0.0;

   for (int d = 0; d < rom_sv->dim(); d++)
   {
      if (d == ref_num_basis) coverage = total;
      total += rom_sv->item(d);
   }
   if (rom_sv->dim() == ref_num_basis) coverage = total;
   coverage /= total;
   printf("Energy fraction with %d basis: %.7f%%\n", ref_num_basis, coverage * 100.0);

   // TODO: hdf5 format + parallel case.
   std::string filename = prefix + "_sv.txt";
   CAROM::PrintVector(*rom_sv, filename);
}

const int SampleGenerator::GetSnapshotOffset(const std::string &comp)
{
   int offset = 0, tmp = -1;

   assert(basis_tags.size() == snapshot_generators.Size());
   for (int b = 0; b < basis_tags.size(); b++)
   {
      if (basis_tags[b].comp != comp) continue;

      offset = snapshot_generators[b]->getNumSamples();

      if (tmp < 0)
         tmp = offset;

      if (offset != tmp)
         mfem_error("SampleGenerator::GetSnapshotOffset- Number of samples over variables does not match!\n");
   }

   return offset;
}
