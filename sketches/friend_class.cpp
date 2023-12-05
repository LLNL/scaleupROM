// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iostream>

namespace nameX
{

class X
{

friend class Y;

protected:
int a = -1;

public:
int b = 2;

public:
X() {};

~X() {};

};

// the friend class should be in the same namespace.
class Y
{
protected:

X *x_in_y = NULL;

public:
Y(X *x) : x_in_y(x) {};

~Y() {};

void printA() { printf("x in y: %d\n", x_in_y->a); }
void printA(X *x) { printf("input x: %d\n", x->a); }

};

}

using namespace nameX;

int main(int argc, char *argv[])
{
   X *testX = new X();
   Y testY(testX);

   testY.printA();

   X *anotherX = new X();
   testY.printA(anotherX);
   
   delete testX;
   return 0;
}