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