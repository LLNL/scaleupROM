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

#include "parameter.hpp"
#include "random.hpp"

using namespace mfem;
using namespace std;

DoubleParam::DoubleParam(const std::string &input_key, YAML::Node option)
   : Parameter(input_key),
     log_scale(config.GetOptionFromDict<bool>("log_scale", false, option)),
     minval(config.GetRequiredOptionFromDict<double>("minimum", option)),
     maxval(config.GetRequiredOptionFromDict<double>("maximum", option))
{}

void DoubleParam::SetParam(const int &param_index, InputParser &parser)
{
   assert(size > 0);
   assert((param_index >= 0) && (param_index < size));
   double val = -1;

   int Np = (size == 1) ? 1 : (size - 1);
   if (log_scale)
   {
      double dp = log(maxval / minval) / static_cast<double>(Np);
      dp = exp(dp);
      val = minval * pow(dp, param_index);
   }
   else
   {
      double dp = (maxval - minval) / static_cast<double>(Np);
      val = minval + param_index * dp;
   }
printf("%s: %.3E\n", key.c_str(), val);
   parser.SetOption<double>(key, val);
}

void DoubleParam::SetRandomParam(InputParser &parser)
{
   double val = -1;

   if (log_scale)
   {
      double range = (maxval / minval);
      val = minval * pow(range, UniformRandom());
   }
   else
   {
      double range = (maxval - minval);
      val = minval + range * UniformRandom();
   }

   parser.SetOption<double>(key, val);
}
