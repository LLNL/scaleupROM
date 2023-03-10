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

#ifndef PARAMETERIZED_PROBLEM_HPP
#define PARAMETERIZED_PROBLEM_HPP

#include "mfem.hpp"
#include "input_parser.hpp"

using namespace mfem;

namespace function_factory
{

typedef double scalar_rhs(const Vector &);

// parameter space index;
extern int index;

namespace poisson0
{
   extern double k;
   extern double offset;

   // TODO: function template?
   double rhs(const Vector &x);

   // TODO: function template?
   void SetParams(const double &k_, const double &offset_);
}

}

enum ParamType { INT, DOUBLE, NUM_TYPE };
extern std::map<std::string, ParamType> ParamTypeMap;

class ParameterizedProblem
{
protected:
   std::string problem_name;

   std::size_t param_num;
   std::map<std::string, int> param_indexes;
   Array<ParamType> param_types;

   Array<int> sampling_sizes;

   // TODO: a way to incorporate all datatypes?
   // TODO: support other datatypes such as string?
   Array<Array<int> *> integer_paramspace;
   Array<Array<double> *> double_paramspace;

   // local sample index;
   Array<int> local_sample_index;

public:
   ParameterizedProblem();

   ~ParameterizedProblem();

   const std::string GetProblemName() { return problem_name; }
   const Array<int> GetLocalSampleIndex() { return local_sample_index; }
   const int GetNumParams() { return param_num; }

   // These are made for tests, but are dangerous to be used elsewhere?
   Array<int>* GetIntParamSpace(const std::string &param_name) { return integer_paramspace[param_indexes[param_name]]; }
   Array<double>* GetDoubleParamSpace(const std::string &param_name) { return double_paramspace[param_indexes[param_name]]; }

   // virtual member functions cannot be passed down as argument.
   // Instead use pointers to static functions.
   function_factory::scalar_rhs *scalar_rhs_ptr = NULL;

   virtual void SetParams(const Array<int> &index)
   { mfem_error("ParameterizedProblem::SetParams is not implemented!\n"); }
};

class Poisson0 : public ParameterizedProblem
{
protected:
   // double k;
   // double offset;

   int k_idx = -1;
   int offset_idx = -1;

public:
   Poisson0();

   // virtual double rhs(const Vector &x);
   virtual void SetParams(const Array<int> &index);
};

#endif
