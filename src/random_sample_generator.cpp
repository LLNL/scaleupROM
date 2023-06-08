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

#include "random_sample_generator.hpp"
#include "random.hpp"
#include "time.h"

using namespace mfem;
using namespace std;

void RandomSampleGenerator::SetParamSpaceSizes()
{
   assert(num_sampling_params > 0);
   assert(params.Size() == num_sampling_params);

   total_samples = config.GetRequiredOption<int>("sample_generation/random_sample_generator/number_of_samples");
   for (int p = 0; p < params.Size(); p++)
      params[p]->SetSize(total_samples);

   // For random sample generator, all parameter have total_samples.
   sampling_sizes.SetSize(num_sampling_params);
   sampling_sizes = total_samples;

   // This does not need the actual samples. distributing only indexes.
   DistributeSamples();

   // TODO: custom random seed
   srand(time(NULL));
}

void RandomSampleGenerator::GenerateParamSpace()
{
   YAML::Node param_list = config.FindNode(param_list_str);
   if (!param_list) mfem_error("SampleGenerator - cannot find the problem name!\n");
   assert(num_sampling_params > 0);

   SetParamSpaceSizes();
   assert(sampling_sizes.Size() == num_sampling_params);

   double_paramspace.SetSize(num_sampling_params);
   double_paramspace = NULL;

   for (int p = 0; p < num_sampling_params; p++)
   {
      assert(sampling_sizes[p] == total_samples);

      double_paramspace[p] = new Vector(sampling_sizes[p]);
      double minval = config.GetRequiredOptionFromDict<double>("minimum", param_list[p]);
      double maxval = config.GetRequiredOptionFromDict<double>("maximum", param_list[p]);

      bool log_scale = config.GetOptionFromDict<bool>("log_scale", false, param_list[p]);
      if (log_scale)
      {
         double dp = (maxval / minval);
         for (int s = 0; s < double_paramspace[p]->Size(); s++)
            (*double_paramspace[p])[s] = minval * pow(dp, UniformRandom());
      }
      else
      {
         double dp = (maxval - minval);
         for (int s = 0; s < double_paramspace[p]->Size(); s++)
            (*double_paramspace[p])[s] = minval + dp * UniformRandom();
      }
   }  // for (int p = 0; p < num_sampling_params; p++)
}

// This is not needed for RandomGenerator, but kept it for compatibility.
const int RandomSampleGenerator::GetSampleIndex(const Array<int> &index)
{
   assert(index.Size() == num_sampling_params);

   // global index should be the same as all the indexes.
   int global_idx = index[0];
   for (int p = 1; p < num_sampling_params; p++)
   {
      assert(global_idx == index[p]);
   }

   assert((global_idx >= 0) && (global_idx < total_samples));
   return global_idx;
}

// This is not needed for RandomGenerator, but kept it for compatibility.
const Array<int> RandomSampleGenerator::GetSampleIndex(const int &index)
{
   assert((index >= 0) && (index < total_samples));

   Array<int> nested_idx(num_sampling_params);
   nested_idx = index;

   return nested_idx;
}

void RandomSampleGenerator::SetSampleParams(const int &index)
{
   assert(params.Size() == num_sampling_params);
   problem->local_sample_index = index;

   // Vector params(num_sampling_params);
   // for (int p = 0; p < num_sampling_params; p++)
   //    params(p) = (*double_paramspace[p])[index];
   // problem->SetParams(sample2problem, params);
   for (int p = 0; p < num_sampling_params; p++)
      params[p]->SetRandomParam(config);
}