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

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

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
   FilenameParam
*/

FilenameParam::FilenameParam(const std::string &input_key, YAML::Node option)
   : Parameter(input_key),
     format(config.GetRequiredOptionFromDict<std::string>("format", option)),
     minval(config.GetRequiredOptionFromDict<int>("minimum", option)),
     maxval(config.GetRequiredOptionFromDict<int>("maximum", option))
   //   zero_pad(config.GetOptionFromDict<int>("zero_pad", 8, option))
{}

void FilenameParam::SetParam(const int &param_index, InputParser &parser)
{
   std::string filename = GetFilename(param_index);

   parser.SetOption<std::string>(key, filename);
}

void FilenameParam::SetRandomParam(InputParser &parser)
{
   double range = static_cast<double>(maxval - minval + 1);
   double realval = static_cast<double>(minval) + range * UniformRandom();
   int val = std::floor(realval);
   if (val > maxval) val = maxval;

   std::string filename = string_format(format, val);

   parser.SetOption<std::string>(key, filename);
}

const std::string FilenameParam::GetFilename(const int &param_index)
{
   assert(size > 0);
   assert((param_index >= 0) && (param_index < size));
   int val = -1;

   int Np = (size == 1) ? 1 : (size - 1);
   int dp = (maxval - minval) / Np;
   val = minval + param_index * dp;

   return string_format(format, val);
}