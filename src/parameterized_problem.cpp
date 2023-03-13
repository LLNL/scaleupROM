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

#include "parameterized_problem.hpp"

using namespace mfem;
using namespace std;

namespace function_factory
{

int index;  // parameter space index

namespace poisson0
{

double k, offset;

double rhs(const Vector &x)
{
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += x(d);
   return sin(poisson0::k * tmp + poisson0::offset);
}

}

}  // namespace function_factory

void ParameterizedProblem::SetParams(const std::string &key, const double &value)
{
   if (!param_map.count(key))
   {
      mfem_error("Poisson0: unknown parameter name!\n");
   }

   (*param_ptr[param_map[key]]) = value;
}

void ParameterizedProblem::SetParams(const Array<int> &indexes, const Vector &values)
{
   assert(indexes.Size() <= param_num);
   assert(indexes.Size() == values.Size());

   for (int idx = 0; idx < indexes.Size(); idx++)
      (*param_ptr[indexes[idx]]) = values(idx);
}

Poisson0::Poisson0()
   : ParameterizedProblem()
{
   param_num = 2;

   // pointer to static function.
   scalar_rhs_ptr = &(function_factory::poisson0::rhs);

   // Default values.
   function_factory::poisson0::k = 1.0;
   function_factory::poisson0::offset = 0.0;

   param_map["k"] = 0;
   param_map["offset"] = 1;

   param_ptr.SetSize(2);
   param_ptr[0] = &(function_factory::poisson0::k);
   param_ptr[1] = &(function_factory::poisson0::offset);
}

ParameterizedProblem* InitParameterizedProblem()
{
   ParameterizedProblem *problem = NULL;
   std::string problem_name = config.GetRequiredOption<std::string>("parameterized_problem/name");

   if (problem_name == "poisson0")
   {
      problem = new Poisson0();
   }
   else
   {
      mfem_error("Unknown parameterized problem name!\n");
   }

   return problem;
}