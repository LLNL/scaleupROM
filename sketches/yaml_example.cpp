// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "input_parser.hpp"
#include <fstream>
#include <iostream>
// #include "multiblock_nonlinearform.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   YAML::Node new_node;
   if (new_node)
      printf("Newly created node is considered true.\n");
   else
      printf("Newly created node is considered false.\n");

   if (new_node["none"])
      printf("non-existing key from a newly created node is considered true.\n");
   else
      printf("non-existing key from a newly created node is considered false.\n");

   YAML::Node test = YAML::LoadFile("test.yaml");
   config = InputParser("test.yaml");

   YAML::Node zombie = test["non-existing"];
   if (test["students"])
   {
      printf("students key exists.\n");
   }
   if (!zombie)
   {
      printf("non-existing key does not exist.\n");
   }

   std::size_t num = test["students"].size();
   for (int k = 0; k < num; k++)
   {
      YAML::Node student = test["students"][k];
      std::string name = student["name"].as<std::string>();
      int age = student["age"].as<int>();
      double gpa = student["gpa"].as<double>();
      printf("Student %d\n", k);
      printf("Name: %s\n", name.c_str());
      printf("age: %d\n", age);
      printf("gpa: %f\n", gpa);
   }

   std::vector<int> vec = test["vector"].as<std::vector<int>>();
   for (int k = 0; k < vec.size(); k++) printf("vec[%d] = %d\n", k, vec[k]);

   Array<int> vec2 = test["vector"].as<Array<int>>();
   for (int k = 0; k < vec2.Size(); k++) printf("vec2[%d] = %d\n", k, vec2[k]);

   int answer = config.GetOption<int>("texas/austin", 78752);
   printf("texas/austin: %d\n", answer);

   answer = config.GetRequiredOption<int>("california/livermore");
   printf("california/livermore: %d\n", answer);

   // YAML::Node boolnode = test["bool"];
   YAML::Node boolnode = config.FindNode("test-result");
   bool teststr = boolnode.as<bool>();
   printf("test-result: %d\n", teststr);

   MPI_Finalize();
   
   return 0;
}