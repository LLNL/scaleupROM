#include "mfem.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int type = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&type, "-t", "--type",
                  "GSSmoother type.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   int N = 4;
   SparseMatrix D(N), L(N), U(N), M(N);
   for (int k = 0; k < N; k++)
      D.Set(k, k, 2.0);
   for (int k = 0; k < N-1; k++)
   {
      L.Set(k+1, k, 0.3);
      U.Set(k, k+1, 0.4);
   }
   D.Finalize();
   L.Finalize();
   U.Finalize();
   M += D;
   M += L;
   M += U;
   M.Finalize();

   GSSmoother gs(M, type);
   gs.iterative_mode = false;

   Vector x(N), y(N), tmp(N);
   y = 0.0;
   for (int k = 0; k < N; k++)
      x(k) = UniformRandom();

   printf("x:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", x(k));
   printf("\n");

   gs.Mult(x, y);
   printf("gs:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y(k));
   printf("\n");

   SparseMatrix DL(N);
   DL += D;
   DL += L;
   DL.Finalize();
   DenseMatrix *DLinv = DL.ToDenseMatrix();
   DLinv->Invert();

   Vector y1(N);
   y1 = 0.0;
   DLinv->Mult(x, y1);
   printf("DLinv:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y1(k));
   printf("\n");

   SparseMatrix DU(N);
   DU += D;
   DU += U;
   DU.Finalize();
   DenseMatrix *DUinv = DU.ToDenseMatrix();
   DUinv->Invert();

   Vector y2(N);
   y2 = 0.0;
   DUinv->Mult(x, y2);
   printf("DUinv:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y2(k));
   printf("\n");

   Vector y3(N);
   y3 = 0.0;
   L.Mult(y1, tmp);
   tmp *= -1.0;
   tmp += x;
   DUinv->Mult(tmp, y3);
   printf("type0:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y3(k));
   printf("\n");

   Vector y40(N), y4(N), x1(N);
   x1 = 0.0;
   y4 = 0.0;
   for (int k = 0; k < N; k++)
      y4(k) = UniformRandom();
   printf("y4:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y4(k));
   printf("\n");
   y40 = y4;

   M.Gauss_Seidel_back(x1, y4);
   printf("gsb(0,y4):\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y4(k));
   printf("\n");

   y4 = y40;
   Vector y5(N);
   y5 = 0.0;
   L.Mult(y4, tmp);
   DUinv->Mult(tmp, y5);
   y5 *= -1.0;
   printf("-DUinvLy4:\t");
   for (int k = 0; k < N; k++)
      printf("%.3E\t", y5(k));
   printf("\n");


   delete DLinv, DUinv;
   return 0;
}
