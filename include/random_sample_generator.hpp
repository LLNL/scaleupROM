// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef RANDOM_SAMPLE_GENERATOR_HPP
#define RANDOM_SAMPLE_GENERATOR_HPP

#include "sample_generator.hpp"

using namespace mfem;

class RandomSampleGenerator : public SampleGenerator
{
public:
   RandomSampleGenerator(MPI_Comm comm) : SampleGenerator(comm) {}

   virtual ~RandomSampleGenerator() {}

   virtual SampleGeneratorType GetType() override { return RANDOM; }

   // RandomSampleGenerator has the same sampling size for all parameters, equal to total samples.
   // const Array<int> GetSampleSizes() { return sampling_sizes; }

   // Generate parameter space as listed in sample_generation/problem_name.
   virtual void SetParamSpaceSizes() override;

   virtual void SetSampleParams(const int &index);
   virtual void SetSampleParams(const Array<int> &index)
   { SetSampleParams(GetSampleIndex(index)); }

   // Determine the given index is assigned to the current process.
   virtual const int GetSampleIndex(const Array<int> &index);
   virtual const Array<int> GetSampleIndex(const int &index);
};

#endif
