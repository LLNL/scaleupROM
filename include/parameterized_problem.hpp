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
}

}

class ParameterizedProblem
{
friend class SampleGenerator;

protected:
   std::string problem_name;

   std::size_t param_num;
   std::map<std::string, int> param_map;

   Array<double *> param_ptr; // address of parameters in function_factory.

   // local sample index;
   int local_sample_index;

public:
   ParameterizedProblem()
      : problem_name(config.GetRequiredOption<std::string>("parameterized_problem/name"))
   {};

   ~ParameterizedProblem() {};

   const std::string GetProblemName() { return problem_name; }
   const int GetNumParams() { return param_num; }
   const int GetLocalSampleIndex() { return local_sample_index; }
   const int GetParamIndex(const std::string &name)
   {
      if (!(param_map.count(name))) printf("%s\n", name.c_str());
      assert(param_map.count(name));
      return param_map[name];
   }

   // virtual member functions cannot be passed down as argument.
   // Instead use pointers to static functions.
   function_factory::scalar_rhs *scalar_rhs_ptr = NULL;

   // TODO: use variadic function? what would be the best format?
   // TODO: support other datatypes such as integer?
   virtual void SetParams(const std::string &key, const double &value);
   virtual void SetParams(const Array<int> &indexes, const Vector &values);
};

class Poisson0 : public ParameterizedProblem
{
protected:
   // int k_idx = -1;
   // int offset_idx = -1;

public:
   Poisson0();
   ~Poisson0() {};

   // virtual double rhs(const Vector &x);
};

ParameterizedProblem* InitParameterizedProblem();

#endif
