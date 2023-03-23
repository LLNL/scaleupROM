// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef RANDOM_SAMPLE_GENERATOR_HPP
#define RANDOM_SAMPLE_GENERATOR_HPP

#include "sample_generator.hpp"

using namespace mfem;

class RandomSampleGenerator : public SampleGenerator
{
public:
   RandomSampleGenerator(MPI_Comm comm, ParameterizedProblem *target)
      : SampleGenerator(comm, target) {}

   virtual ~RandomSampleGenerator() {}

   // RandomSampleGenerator has the same sampling size for all parameters, equal to total samples.
   // const Array<int> GetSampleSizes() { return sampling_sizes; }

   // Generate parameter space as listed in sample_generation/problem_name.
   virtual void SetParamSpaceSizes();
   virtual void GenerateParamSpace();

   virtual void SetSampleParams(const int &index);
   virtual void SetSampleParams(const Array<int> &index)
   { SetSampleParams(GetSampleIndex(index)); }

   // Determine the given index is assigned to the current process.
   virtual const int GetSampleIndex(const Array<int> &index);
   virtual const Array<int> GetSampleIndex(const int &index);
};

#endif
