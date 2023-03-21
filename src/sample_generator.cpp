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

#include "sample_generator.hpp"

using namespace mfem;
using namespace std;

SampleGenerator::SampleGenerator(MPI_Comm comm, ParameterizedProblem *target)
   : problem(target)
{
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &proc_rank);

   sample_dir = config.GetOption<std::string>("sample_generation/file_path/directory", ".");
   std::string problem_name = problem->GetProblemName();
   sample_prefix = config.GetOption<std::string>("sample_generation/file_path/prefix", problem_name);

   // TODO: currently combined with sample generation part.
   // TODO: Separate with sample generation.
   std::string param_list_str("sample_generation/" + problem_name);
   YAML::Node param_list = config.FindNode(param_list_str);
   MFEM_ASSERT(param_list, "ParameterizedProblem - cannot find the problem name!\n");

   num_sampling_params = param_list.size();
   double_paramspace.SetSize(num_sampling_params);
   double_paramspace = NULL;

   sampling_sizes.SetSize(num_sampling_params);
   sampling_sizes = -1;

   sample2problem.SetSize(num_sampling_params);
   sample2problem = -1;

   for (int p = 0; p < num_sampling_params; p++)
   {
      std::string param_name = config.GetRequiredOptionFromDict<std::string>("parameter_name", param_list[p]);
      sample_param_map[param_name] = p;
      sample2problem[p] = problem->GetParamIndex(param_name);

      sampling_sizes[p] = config.GetRequiredOptionFromDict<int>("sample_size", param_list[p]);

      double_paramspace[p] = new Vector(sampling_sizes[p]);
      double minval = config.GetRequiredOptionFromDict<double>("minimum", param_list[p]);
      double maxval = config.GetRequiredOptionFromDict<double>("maximum", param_list[p]);

      bool log_scale = config.GetOptionFromDict<bool>("log_scale", false, param_list[p]);
      if (log_scale)
      {
         double dp = log(maxval / minval) / (sampling_sizes[p] - 1);
         dp = exp(dp);

         (*double_paramspace[p])[0] = minval;
         for (int s = 1; s < double_paramspace[p]->Size(); s++)
            (*double_paramspace[p])[s] = (*double_paramspace[p])[s-1] * dp;
      }
      else
      {
         double dp = (sampling_sizes[p] == 1) ? 0.0 : (maxval - minval) / (sampling_sizes[p] - 1);
         for (int s = 0; s < double_paramspace[p]->Size(); s++)
            (*double_paramspace[p])[s] = minval + s * dp;
      }
   }  // for (int p = 0; p < num_sampling_params; p++)

   total_samples = 1;
   for (int p = 0; p < num_sampling_params; p++)
      total_samples *= sampling_sizes[p];

   DistributeSamples();
}

SampleGenerator::~SampleGenerator()
{
   for (int p = 0; p < double_paramspace.Size(); p++)
   {
      delete double_paramspace[p];
   }
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
   problem->local_sample_index = index;

   const Array<int> nested_idx = GetSampleIndex(index);

   Vector params(num_sampling_params);
   for (int p = 0; p < num_sampling_params; p++)
      params(p) = (*double_paramspace[p])[nested_idx[p]];
   problem->SetParams(sample2problem, params);
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