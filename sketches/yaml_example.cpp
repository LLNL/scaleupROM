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

#include "input_parser.hpp"
#include <fstream>
#include <iostream>
// #include "multiblock_nonlinearform.hpp"

using namespace mfem;

int main(int argc, char *argv[])
{
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
   
   return 0;
}