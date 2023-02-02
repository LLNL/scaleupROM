#include <stdio.h>
#include "mfem.hpp"
#include "test.hpp"

using namespace mfem;

int main(int argc, char* argv[])
{
  printf("Hello World\n");

  testObject tester(2);
  printf("tester: %d\n", tester.a_);
  return 0;
}
