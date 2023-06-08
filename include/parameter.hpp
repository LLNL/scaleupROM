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

#endif
