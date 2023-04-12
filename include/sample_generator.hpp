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

#ifndef SAMPLE_GENERATOR_HPP
#define SAMPLE_GENERATOR_HPP

#include "mfem.hpp"
#include "parameterized_problem.hpp"

using namespace mfem;

class SampleGenerator
{
protected:
   int num_procs;
   int proc_rank;
   Array<int> sample_offsets;

   ParameterizedProblem *problem;

   // input path for parameter list
   std::string param_list_str;

   std::size_t num_sampling_params;
   std::map<std::string, int> sample_param_map;
   Array<int> sample2problem;

   Array<int> sampling_sizes;
   int total_samples;

   // TODO: a way to incorporate all datatypes?
   // TODO: support other datatypes such as integer?
   // Array<Array<int> *> integer_paramspace;
   Array<Vector *> double_paramspace;

   // file path
   std::string sample_dir = ".";
   std::string sample_prefix;
   int file_offset = 0;

public:
   SampleGenerator(MPI_Comm comm, ParameterizedProblem *target);

   virtual ~SampleGenerator();

   const int GetNumSampleParams() { return num_sampling_params; }
   const Array<int> GetSampleSizes() { return sampling_sizes; }
   const int GetTotalSampleSize() { return total_samples; }
   const int GetProcRank() { return proc_rank; }
   const int GetFileOffset() { return file_offset; }

   // Generate parameter space as listed in sample_generation/problem_name.
   virtual void SetParamSpaceSizes();
   virtual void GenerateParamSpace();

   // These are made for tests, but are dangerous to be used elsewhere?
   // Array<int>* GetIntParamSpace(const std::string &param_name) { return integer_paramspace[param_indexes[param_name]]; }
   Vector* GetDoubleParamSpace(const std::string &param_name)
   { return double_paramspace[sample_param_map[param_name]]; }

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
};

#endif
