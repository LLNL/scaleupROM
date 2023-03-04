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

#ifndef INPUT_PARSER_HPP
#define INPUT_PARSER_HPP

#include "yaml.h"
#include "mfem.hpp"

using namespace mfem;

namespace YAML {

template<typename T>
struct convert<Array<T>> {
   static Node encode(const Array<T>& rhs) {
      Node node;
      for (int k = 0; k < rhs.Size(); k++)
         node.push_back(rhs[k]);
      return node;
   }

   static bool decode(const Node& node, Array<T>& rhs) {
      if(!node.IsSequence()) {
         return false;
      }

      rhs.SetSize(node.size());
      for (int k = 0; k < node.size(); k++)
         rhs[k] = node[k].as<T>();

      return true;
   }
};

}

class InputParser
{
protected:
   YAML::Node dict_;

   std::string file_;

public:
   InputParser() {};

   InputParser(const std::string &input_file)
   { file_ = input_file; dict_ = YAML::LoadFile(input_file.c_str()); }

   template<class T>
   const T GetRequiredOption(const std::string &keys)
   {
      YAML::Node node = FindNode(keys);
      if (!node)
      {
         printf("%s does not exist in the input file %s!\n", keys.c_str(), file_.c_str());
         exit(-1);
      }
      return node.as<T>();
   }

   template<class T>
   const T GetOption(const std::string &keys, const T &fallback)
   {
      YAML::Node node = FindNode(keys);
      if (!node)
      {
         return fallback;
      }
      return node.as<T>();
   }

   YAML::Node FindNode(const std::string &keys);

};

extern InputParser config;

#endif
