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

std::map<std::string, ParamType> ParamTypeMap = {{"integer", ParamType::INT}, {"double", ParamType::DOUBLE}};

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

ParameterizedProblem::ParameterizedProblem()
{
   problem_name = config.GetRequiredOption<std::string>("parameterized_problem/name");

   std::string param_list_str("parameterized_problem/" + problem_name);
   YAML::Node param_list = config.FindNode(param_list_str);
   MFEM_ASSERT(param_list, "ParameterizedProblem - cannot find the problem name!\n");

   param_num = param_list.size();
   integer_paramspace.SetSize(param_num);
   integer_paramspace = NULL;
   double_paramspace.SetSize(param_num);
   double_paramspace = NULL;

   sampling_sizes.SetSize(param_num);
   sampling_sizes = -1;

   param_types.SetSize(param_num);
   param_types = ParamType::NUM_TYPE;

   for (int p = 0; p < param_num; p++)
   {
      std::string param_name = config.GetRequiredOptionFromDict<std::string>("parameter_name", param_list[p]);
      param_indexes[param_name] = p;

      param_types[p] = ParamTypeMap[config.GetRequiredOptionFromDict<std::string>("type", param_list[p])];

      sampling_sizes[p] = config.GetRequiredOptionFromDict<int>("sample_size", param_list[p]);

      switch (param_types[p])
      {
         case ParamType::INT:
         {
            // integer_paramspace[p] = new Array<int>(sampling_sizes[p]);
            mfem_error("ParameterizedProblem - integer parameters are not implemented yet!\n");
            break;
         }
         case ParamType::DOUBLE:
         {
            double_paramspace[p] = new Array<double>(sampling_sizes[p]);
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
               double dp = (maxval - minval) / (sampling_sizes[p] - 1);
               for (int s = 0; s < double_paramspace[p]->Size(); s++)
                  (*double_paramspace[p])[s] = minval + s * dp;
            }
            break;
         }
         default:
            mfem_error("ParameterizedProblem - unsupported parameter type!\n");
            break;
      }
   }
}

ParameterizedProblem::~ParameterizedProblem()
{
   if (param_num > 0)
   {
      for (int p = 0; p < param_num; p++)
      {
         delete integer_paramspace[p];
         delete double_paramspace[p];
      }
   }
}

Poisson0::Poisson0()
   : ParameterizedProblem()
{
   scalar_rhs_ptr = &(function_factory::poisson0::rhs);
   
   if (param_indexes.count("k")) k_idx = param_indexes["k"];
   if (param_indexes.count("offset")) offset_idx = param_indexes["offset"];

   if (k_idx < 0) function_factory::poisson0::k = 1.0;
   if (offset_idx < 0) function_factory::poisson0::offset = 0.0;
}

void Poisson0::SetParams(const Array<int> &index)
{
   int param_space_size = index.Size();
   MFEM_ASSERT(param_space_size > 0, "Poisson0::SetParams - Invalid parameter space index!\n");
   MFEM_ASSERT(param_num >= param_space_size, "Poisson0::SetParams - parameter space is not generated!\n");

   local_sample_index = index;

   if (k_idx >= 0)
      function_factory::poisson0::k = (*double_paramspace[k_idx])[index[k_idx]];
   if (offset_idx >= 0)
      function_factory::poisson0::offset = (*double_paramspace[offset_idx])[index[offset_idx]];
}