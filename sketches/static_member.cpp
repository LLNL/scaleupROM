// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iostream>
#include <vector>

namespace nameX
{

class X
{
protected:
int a = -1;

public:
static int num_str;
static std::vector<std::string> str_vec;

public:
X() {};

~X() {};

};

// the friend class should be in the same namespace.
class Y : public X
{
public:

// ISO C++ forbids inheriting static members from the base class.
static int num_str;
static std::vector<std::string> str_vec;

public:
Y() : X() {};

~Y() {};

};

int X::num_str = 1;
int Y::num_str = 2;
std::vector<std::string> X::str_vec({"xstr1"});
std::vector<std::string> Y::str_vec({"ystr1", "ystr2"});

}

using namespace nameX;

int main(int argc, char *argv[])
{
   X *testX = new X();
   Y testY;

   printf("X::num_str: %d\n", X::num_str);
   for (int k = 0; k < X::num_str; k++)
      printf("%s\t", X::str_vec[k].c_str());
   printf("\n");
   printf("Y::num_str: %d\n", Y::num_str);
   for (int k = 0; k < Y::num_str; k++)
      printf("%s\t", Y::str_vec[k].c_str());
   printf("\n");
   
   delete testX;
   return 0;
}