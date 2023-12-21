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

class IntegerParam : public Parameter
{
protected:
   int minval = -1;
   int maxval = -1;

   const int GetInteger(const int &param_index);
   const int GetRandomInteger();
public:
   IntegerParam(const std::string &input_key, YAML::Node option);
   virtual ~IntegerParam() {}

   virtual void SetParam(const int &param_index, InputParser &parser);
   virtual void SetRandomParam(InputParser &parser);

   virtual void SetMaximumSize() { SetSize(maxval - minval); }
};

class FilenameParam : public IntegerParam
{
protected:
   std::string format = "";

public:
   FilenameParam(const std::string &input_key, YAML::Node option);
   virtual ~FilenameParam() {}

   virtual void SetParam(const int &param_index, InputParser &parser) override;
   virtual void SetRandomParam(InputParser &parser) override;
   void ParseFilenames(std::vector<std::string> &filenames);
};

#endif
