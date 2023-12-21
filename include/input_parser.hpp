// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef INPUT_PARSER_HPP
#define INPUT_PARSER_HPP

#include "yaml-cpp/yaml.h"
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
}; // struct convert<Array<T>>

template<>
struct convert<Vector> {
   static Node encode(const Vector& rhs) {
      Node node;
      for (int k = 0; k < rhs.Size(); k++)
         node.push_back(rhs[k]);
      return node;
   }

   static bool decode(const Node& node, Vector& rhs) {
      if(!node.IsSequence()) {
         return false;
      }

      rhs.SetSize(node.size());
      for (int k = 0; k < node.size(); k++)
         rhs[k] = node[k].as<double>();

      return true;
   }
}; // struct convert<Vector>

template<>
struct convert<std::vector<std::string>> {
   static Node encode(const std::vector<std::string>& rhs) {
      Node node;
      for (int k = 0; k < rhs.size(); k++)
         node.push_back(rhs[k]);
      return node;
   }

   static bool decode(const Node& node, std::vector<std::string>& rhs) {
      if(!node.IsSequence()) {
         return false;
      }

      rhs.resize(node.size());
      for (int k = 0; k < node.size(); k++)
         rhs[k] = node[k].as<std::string>();

      return true;
   }
}; // struct convert<std::vector<std::string>>

}

class InputParser
{
protected:
   std::string file_;

   int rank;   // MPI rank

public:
   YAML::Node dict_;

public:
   InputParser() {};

   InputParser(const std::string &input_file, const std::string forced_input="");

   template<class T>
   const T GetRequiredOptionFromDict(const std::string &keys, YAML::Node input_dict)
   {
      YAML::Node node = FindNodeFromDict(keys, input_dict);
      if (!node)
      {
         printf("%s does not exist in the input file %s!\n", keys.c_str(), file_.c_str());
         exit(-1);
      }
      return node.as<T>();
   }

   template<class T>
   const T GetOptionFromDict(const std::string &keys, const T &fallback, YAML::Node input_dict)
   {
      YAML::Node node = FindNodeFromDict(keys, input_dict);
      if (!node)
      {
         return fallback;
      }
      return node.as<T>();
   }

   YAML::Node FindNodeFromDict(const std::string &keys, YAML::Node input_dict, bool create=false);

   template<class T>
   const T GetRequiredOption(const std::string &keys)
   { return GetRequiredOptionFromDict<T>(keys, dict_); }

   template<class T>
   const T GetOption(const std::string &keys, const T &fallback)
   { return GetOptionFromDict<T>(keys, fallback, dict_); }

   YAML::Node FindNode(const std::string &keys, bool create=false)
   { return FindNodeFromDict(keys, dict_, create); }

   template<class T>
   void SetOptionInDict(const std::string &keys, const T &value, YAML::Node input_dict)
   {
      YAML::Node node = FindNodeFromDict(keys, input_dict, true);
      node = value;
      return;
   }

   template<class T>
   void SetOption(const std::string &keys, const T &value)
   { SetOptionInDict<T>(keys, value, dict_); }

private:
   void OverwriteOption(const std::string &forced_input);

};

extern InputParser config;

#endif
