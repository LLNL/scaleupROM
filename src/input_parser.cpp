// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "input_parser.hpp"
#include <stdlib.h>

InputParser::InputParser(const std::string &input_file, const std::string forced_input)
{
   file_ = input_file;

   int flag = 0;
   MPI_Initialized(&flag);
   if (flag)   // parallel case
   {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      std::stringstream buffer;
      std::string ss;
      if (rank == 0)
      {
         std::ifstream file(input_file);
         buffer << file.rdbuf();
      }

      int bufferSize = buffer.str().size();
      MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank == 0)
         ss = buffer.str();
      else
         ss.resize(bufferSize);

      MPI_Bcast(&ss[0], ss.capacity(), MPI_CHAR, 0, MPI_COMM_WORLD);

      dict_ = YAML::Load(ss);
   }
   else   // serial case
   {
      rank = 0;
      dict_ = YAML::LoadFile(input_file.c_str());
   }

   OverwriteOption(forced_input);

   return;
}

YAML::Node InputParser::FindNodeFromDict(const std::string &keys, YAML::Node input_dict, bool create)
{
   // Per tutorial of yaml-cpp, operator= *seems* to be a shallow copy.
   // However, in fact they are deep copy, and the following recursive = operation screws up the dict.
   // Now we store the node in a vector.
   std::vector<YAML::Node> nodes(0);
   nodes.push_back(input_dict);

   std::istringstream key_iterator(keys);
   int dd = 0;
   for (std::string s; std::getline(key_iterator, s, '/'); )
   {
      if (!(nodes.back()[s]))
      {
         if (create)
            nodes.back()[s] = YAML::Node();
         else
            return nodes.back()[s];
      }

      nodes.push_back(nodes.back()[s]);
   }
   return nodes.back();
}

void InputParser::OverwriteOption(const std::string &forced_input)
{
   std::stringstream ss(forced_input);
   std::string kv, key, val, dummy;
   bool success = false;

   while (std::getline(ss, kv, ':'))
   {
      std::stringstream kvss(kv);
      success = true;
      success = success && (std::getline(kvss, key, '='));
      success = success && (std::getline(kvss, val, '='));
      success = success && (!std::getline(kvss, dummy, '='));
      if (!success)
      {
         if (rank == 0)
         {
            std::string msg = kv + " is not a valid key=value pair!\n";
            printf("%s", msg.c_str());
         }
         exit(-1);
      }
      SetOption(key, YAML::Load(val));
   }
}

// template int InputParser::GetRequiredOption<int>(const std::string&);

InputParser config;
