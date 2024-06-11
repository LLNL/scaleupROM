// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "parameter.hpp"
#include "etc.hpp"
#include <memory>
#include <string>
#include <stdexcept>

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

/*
   IntegerParam
*/

IntegerParam::IntegerParam(const std::string &input_key, YAML::Node option)
   : Parameter(input_key),
     minval(config.GetRequiredOptionFromDict<int>("minimum", option)),
     maxval(config.GetRequiredOptionFromDict<int>("maximum", option))
{}

void IntegerParam::SetParam(const int &param_index, InputParser &parser)
{
   int val = GetInteger(param_index);

   parser.SetOption<int>(key, val);
}

void IntegerParam::SetRandomParam(InputParser &parser)
{
   int val = GetRandomInteger();

   parser.SetOption<int>(key, val);
}

const int IntegerParam::GetInteger(const int &param_index)
{
   assert(size > 0);
   assert((param_index >= 0) && (param_index < size));
   int val = -1;

   int Np = (size == 1) ? 1 : (size - 1);
   int dp = (maxval - minval) / Np;
   val = minval + param_index * dp;

   return val;
}

const int IntegerParam::GetRandomInteger()
{
   int val = UniformRandom(minval, maxval);

   return val;
}

/*
   FilenameParam
*/

FilenameParam::FilenameParam(const std::string &input_key, YAML::Node option)
   : IntegerParam(input_key, option),
     format(config.GetRequiredOptionFromDict<std::string>("format", option))
{}

void FilenameParam::SetParam(const int &param_index, InputParser &parser)
{
   int val = GetInteger(param_index);
   std::string filename = string_format(format, val);

   parser.SetOption<std::string>(key, filename);
}

void FilenameParam::SetRandomParam(InputParser &parser)
{
   int val = GetRandomInteger();
   std::string filename = string_format(format, val);

   parser.SetOption<std::string>(key, filename);
}

void FilenameParam::ParseFilenames(std::vector<std::string> &filenames)
{
   for (int k = minval; k <= maxval; k++)
      filenames.push_back(string_format(format, k));
   return;
}