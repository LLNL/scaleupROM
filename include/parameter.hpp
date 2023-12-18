// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "mfem.hpp"
#include "input_parser.hpp"

using namespace mfem;

class Parameter
{
protected:
   // key used for config InputParser.
   std::string key = "";

   int size = -1;

public:
   Parameter(const std::string &input_key)
      : key(input_key) {}
   virtual ~Parameter() {}

   const std::string GetKey() { return key; }
   const double GetSize() { return size; }
   void SetSize(const int &sample_size)
   { assert(sample_size > 0); size = sample_size; }

   virtual void SetParam(const int &param_index, InputParser &parser) = 0;
   virtual void SetRandomParam(InputParser &parser) = 0;
};

class DoubleParam : public Parameter
{
protected:
   double minval = -1.;
   double maxval = -1.;
   bool log_scale = false;

public:
   DoubleParam(const std::string &input_key, YAML::Node option);
   virtual ~DoubleParam() {}

   virtual void SetParam(const int &param_index, InputParser &parser);
   virtual void SetRandomParam(InputParser &parser);
};

// TODO(kevin): technically we can extract IntegerParam from FilenameParam,
//              and let FilenameParam inherit from it.
class FilenameParam : public Parameter
{
protected:
   int minval = -1;
   int maxval = -1;
   std::string format = "";

public:
   FilenameParam(const std::string &input_key, YAML::Node option);
   virtual ~FilenameParam() {}

   virtual void SetParam(const int &param_index, InputParser &parser);
   virtual void SetRandomParam(InputParser &parser);

   virtual const std::string GetFilename(const int &param_index);
   virtual void SetMaximumSize() { SetSize(maxval - minval); }
};

#endif
