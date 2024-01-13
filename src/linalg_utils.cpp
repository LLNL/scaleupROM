// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "linalg_utils.hpp"
#include "utils/HDFDatabase.h"

using namespace mfem;
using namespace std;

namespace CAROM
{

void ComputeCtAB(const Operator& A,
                 const CAROM::Matrix& B,  // Non-Distributed matrix
                 const CAROM::Matrix& C,  // Non-Distributed matrix
                 CAROM::Matrix& CtAB)     // Non-distributed (local) matrix
{
   MFEM_VERIFY(B.distributed() && C.distributed() && !CtAB.distributed(),
               "In ComputeCtAB, B, C and CtAB must not be distributed!\n");

   const int num_rows = B.numRows();
   const int num_cols = B.numColumns();
   const int num_rows_A = A.NumRows();

   MFEM_VERIFY(C.numRows() == num_rows_A, "");

   mfem::Vector Bvec(num_rows);
   mfem::Vector ABvec(num_rows_A);

   CAROM::Matrix AB(num_rows_A, num_cols, true);

   for (int i = 0; i < num_cols; ++i) {
      for (int j = 0; j < num_rows; ++j) {
         Bvec[j] = B(j, i);
      }
      A.Mult(Bvec, ABvec);
      for (int j = 0; j < num_rows_A; ++j) {
         AB(j, i) = ABvec[j];
      }
   }

   C.transposeMult(AB, CtAB);
}

void SetBlock(const CAROM::Matrix& blockMat,
               const int iStart, const int iEnd,
               const int jStart, const int jEnd,
               CAROM::Matrix& globalMat)
{
   MFEM_VERIFY(!blockMat.distributed() && !globalMat.distributed(),
               "In SetBlock, blockMat and globalMat must not be distributed!\n");

   const int block_row = blockMat.numRows();
   const int block_col = blockMat.numColumns();
   const int global_row = globalMat.numRows();
   const int global_col = globalMat.numColumns();
   assert(block_row <= global_row);
   assert(block_col <= global_col);
   assert((iStart >= 0) && (iStart < global_row));
   assert((jStart >= 0) && (jStart < global_col));
   assert((iEnd >= 0) && (iEnd <= global_row));
   assert((jEnd >= 0) && (jEnd <= global_col));

   for (int i = iStart; i < iEnd; i++)
   {
      for (int j = jStart; j < jEnd; j++)
      {
         globalMat(i, j) = blockMat(i - iStart, j - jStart);
      }
   }
}

void SetBlock(const CAROM::Vector& blockVec,
               const int iStart, const int iEnd,
               CAROM::Vector& globalVec)
{
   const int block_row = blockVec.dim();
   const int global_row = globalVec.dim();
   assert(block_row <= global_row);
   assert((iStart >= 0) && (iStart < global_row));
   assert((iEnd >= 0) && (iEnd <= global_row));

   for (int i = iStart; i < iEnd; i++)
   {
      globalVec(i) = blockVec(i - iStart);
   }
}

void CopyMatrix(const CAROM::Matrix &carom_mat,
                DenseMatrix &mfem_mat)
{
   const int num_row = carom_mat.numRows();
   const int num_col = carom_mat.numColumns();

   mfem_mat.SetSize(num_row, num_col);
   // This copy is needed since their indexing orders are different.
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
         mfem_mat(i,j) = carom_mat.item(i,j);
}

// Does not support parallelization. for debugging.
void PrintMatrix(const CAROM::Matrix &mat,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < mat.numRows(); i++)
   {
      for (int j = 0; j < mat.numColumns(); j++)
         fprintf(fp, "%.15E\t", mat.item(i,j));
      fprintf(fp, "\n");
   }

   fclose(fp);
   return;
}

void PrintVector(const CAROM::Vector &vec,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < vec.dim(); i++)
      fprintf(fp, "%.15E\n", vec.item(i));

   fclose(fp);
   return;
}

}

namespace mfem
{

void GramSchmidt(DenseMatrix& mat)
{
   const int num_row = mat.NumRows();
   const int num_col = mat.NumCols();
   Vector vec_c, tmp;
   for (int c = 0; c < num_col; c++)
   {
      mat.GetColumnReference(c, vec_c);

      for (int i = 0; i < c; i++)
      {
         mat.GetColumnReference(i, tmp);
         vec_c.Add(-(tmp * vec_c), tmp);
      }

      vec_c /= sqrt(vec_c * vec_c);
   }
}

void Orthonormalize(DenseMatrix& mat1, DenseMatrix& mat)
{
   const int num_row = mat.NumRows();
   const int num_col = mat.NumCols();
   const int num_col1 = mat1.NumCols();
   assert(num_row == mat1.NumRows());

   Vector vec_c, tmp;
   for (int c = 0; c < num_col; c++)
   {
      mat.GetColumnReference(c, vec_c);

      /* we don't assume mat1 is normalized. */
      for (int i = 0; i < num_col1; i++)
      {
         mat1.GetColumnReference(i, tmp);
         vec_c.Add(-(tmp * vec_c) / (tmp * tmp), tmp);
      }

      for (int i = 0; i < c; i++)
      {
         mat.GetColumnReference(i, tmp);
         vec_c.Add(-(tmp * vec_c), tmp);
      }

      vec_c /= sqrt(vec_c * vec_c);
   }
}

void RtAP(DenseMatrix& R,
         const Operator& A,
         DenseMatrix& P,
         DenseMatrix& RtAP)
{
   assert(R.NumRows() == A.NumRows());
   assert(A.NumCols() == P.NumRows());

   const int num_row = R.NumCols();
   const int num_col = P.NumCols();
   RtAP.SetSize(num_row, num_col);
   Vector vec_i, vec_j;
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
      {
         R.GetColumnReference(i, vec_i);
         P.GetColumnReference(j, vec_j);
         // NOTE: mfem::SparseMatrix.InnerProduct(x, y) computes y^t A x
         Vector tmp(vec_i.Size());
         tmp = 0.0;
         A.Mult(vec_j, tmp);
         RtAP(i, j) = vec_i * tmp;
      }
}

DenseMatrix* DenseRtAP(DenseMatrix& R,
   const Operator& A, DenseMatrix& P)
{
   DenseMatrix *mat = new DenseMatrix;
   RtAP(R, A, P, *mat);
   return mat;
}

SparseMatrix* SparseRtAP(DenseMatrix& R,
   const Operator& A, DenseMatrix& P)
{
   assert(R.NumRows() == A.NumRows());
   assert(A.NumCols() == P.NumRows());

   const int num_row = R.NumCols();
   const int num_col = P.NumCols();
   SparseMatrix *RAP = new SparseMatrix(num_row, num_col);
   Vector vec_i, vec_j;
   for (int i = 0; i < num_row; i++)
      for (int j = 0; j < num_col; j++)
      {
         R.GetColumnReference(i, vec_i);
         P.GetColumnReference(j, vec_j);
         // NOTE: mfem::SparseMatrix.InnerProduct(x, y) computes y^t A x
         Vector tmp(vec_i.Size());
         tmp = 0.0;
         A.Mult(vec_j, tmp);
         RAP->Set(i, j, vec_i * tmp);
      }
   RAP->Finalize();

   return RAP;
}

template<typename T>
void PrintMatrix(const T &mat,
                const std::string &filename)
{
   std::ofstream file(filename.c_str());
   mat.PrintMatlab(file);
}

template void PrintMatrix<SparseMatrix>(const SparseMatrix &mat, const std::string &filename);
template void PrintMatrix<BlockMatrix>(const BlockMatrix &mat, const std::string &filename);

// Does not support parallelization. for debugging.
void PrintMatrix(const DenseMatrix &mat,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < mat.NumRows(); i++)
   {
      for (int j = 0; j < mat.NumCols(); j++)
         fprintf(fp, "%.15E\t", mat(i,j));
      fprintf(fp, "\n");
   }

   fclose(fp);
   return;
}

void PrintVector(const Vector &vec,
                 const std::string &filename)
{
   FILE *fp = fopen(filename.c_str(), "w");

   for (int i = 0; i < vec.Size(); i++)
      fprintf(fp, "%.15E\n", vec(i));

   fclose(fp);
   return;
}

// TODO: support parallel I/O.
SparseMatrix* ReadSparseMatrixFromHDF(const std::string filename)
{
   bool success = false;
   CAROM::HDFDatabase hdfIO;

   success = hdfIO.open(filename, "r");
   assert(success);

   int size[2];
   hdfIO.getIntegerArray("size", &size[0], 2);

   const int n_entry = hdfIO.getDoubleArraySize("data");
   Array<int> i_read(n_entry), j_read(n_entry);
   Vector data_read(n_entry);
   hdfIO.getIntegerArray("I", i_read.Write(), size[0]+1);
   hdfIO.getIntegerArray("J", j_read.Write(), i_read[size[0]]);
   hdfIO.getDoubleArray("data", data_read.Write(), i_read[size[0]]);

   // Need to transfer the ownership to avoid segfault or double-free.
   int *ip, *jp;
   i_read.StealData(&ip);
   j_read.StealData(&jp);

   SparseMatrix *mat = new SparseMatrix(ip, jp, data_read.StealData(), size[0], size[1]);

   success = hdfIO.close();
   assert(success);

   return mat;
}

void WriteSparseMatrixToHDF(const SparseMatrix* mat, const std::string filename)
{
   assert(mat->Finalized());
   
   bool success = false;
   CAROM::HDFDatabase hdfIO;

   success = hdfIO.create(filename);
   assert(success);

   const int nnz = mat->NumNonZeroElems();
   const int height = mat->Height();
   const int *i_idx = mat->ReadI();
   const int *j_idx = mat->ReadJ();
   const double *data = mat->ReadData();

   hdfIO.putIntegerArray("I", i_idx, height+1);
   hdfIO.putIntegerArray("J", j_idx, i_idx[height]);
   hdfIO.putDoubleArray("data", data, i_idx[height]);

   int size[2];
   size[0] = mat->NumRows();
   size[1] = mat->NumCols();
   hdfIO.putIntegerArray("size", &size[0], 2);

   success = hdfIO.close();
   assert(success);
}

void MultSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y)
{
   const int nrow = rows.Size(), ncol = mat.NumCols(), height = mat.NumRows();
   assert(x.Size() == ncol);
   y.SetSize(nrow);

   const int rmin = rows.Min(), rmax = rows.Max();
   assert((rmin >= 0) && (rmin < mat.NumRows()));
   assert((rmax >= 0) && (rmax < mat.NumRows()));

   const double *d_mat = mat.Read();
   const double *d_x = x.Read();
   const int *d_rows = rows.Read();
   double *d_y = y.GetData();

   double xc = d_x[0];
   for (int r = 0; r < nrow; r++)
      d_y[r] = d_mat[d_rows[r]] * xc;
   d_mat += height;

   for (int c = 1; c < ncol; c++)
   {
      xc = d_x[c];
      for (int r = 0; r < nrow; r++)
         d_y[r] += d_mat[d_rows[r]] * xc;
      d_mat += height;
   }
}

void AddMultTransposeSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y)
{
   const int nrow = rows.Size(), ncol = mat.NumCols(), height = mat.NumRows();
   assert(x.Size() == nrow);
   assert(y.Size() == ncol);

   const int rmin = rows.Min(), rmax = rows.Max();
   assert((rmin >= 0) && (rmin < mat.NumRows()));
   assert((rmax >= 0) && (rmax < mat.NumRows()));

   const double *d_mat = mat.Read();
   const double *d_x = x.Read();
   const int *d_rows = rows.Read();
   double *d_y = y.GetData();

   double yc;
   for (int c = 0; c < ncol; c++)
   {
      yc = 0.0;
      for (int r = 0; r < nrow; r++)
         yc += d_mat[d_rows[r]] * d_x[r];
      d_y[c] += yc;
      d_mat += height;
   }
}

void MultTransposeSubMatrix(const DenseMatrix &mat, const Array<int> &rows, const Vector &x, Vector &y)
{
   const int nrow = rows.Size(), ncol = mat.NumCols(), height = mat.NumRows();
   assert(x.Size() == nrow);
   assert(y.Size() == ncol);

   const int rmin = rows.Min(), rmax = rows.Max();
   assert((rmin >= 0) && (rmin < mat.NumRows()));
   assert((rmax >= 0) && (rmax < mat.NumRows()));

   const double *d_mat = mat.Read();
   const double *d_x = x.Read();
   const int *d_rows = rows.Read();
   double *d_y = y.GetData();

   double yc;
   for (int c = 0; c < ncol; c++)
   {
      yc = 0.0;
      for (int r = 0; r < nrow; r++)
         yc += d_mat[d_rows[r]] * d_x[r];
      d_y[c] = yc;
      d_mat += height;
   }
}

void AddSubMatrixRtAP(const DenseMatrix& R, const Array<int> &Rrows,
                      const DenseMatrix& A,
                      const DenseMatrix& P, const Array<int> &Prows,
                      DenseMatrix& RAP)
{
   const int Arow = Rrows.Size(), Acol = Prows.Size();
   const int height = R.NumCols(), width = P.NumCols();
   const int Pheight = P.NumRows();
   assert(A.NumRows() == Arow);
   assert(A.NumCols() == Acol);

   {  // row index checks.
      int rmin = Rrows.Min(), rmax = Rrows.Max();
      assert((rmin >= 0) && (rmin < R.NumRows()));
      assert((rmax >= 0) && (rmax < R.NumRows()));
      rmin = Prows.Min();
      rmax = Prows.Max();
      assert((rmin >= 0) && (rmin < P.NumRows()));
      assert((rmax >= 0) && (rmax < P.NumRows()));
   }

   RAP.SetSize(height, width);
   Vector tmp(Arow), RAP_col;

   const double *d_P = P.Read();
   double *d_tmp = tmp.GetData();
   double pr;

   for (int j = 0; j < width; j++)
   {
      const double *d_A = A.Read();

      // step 1: tmp = A * P[:, j]
      pr = d_P[Prows[0]];
      for (int r = 0; r < Arow; r++)
         d_tmp[r] = d_A[r] * pr;
      d_A += Arow;

      for (int c = 1; c < Acol; c++)
      {
         pr = d_P[Prows[c]];
         for (int r = 0; r < Arow; r++)
            d_tmp[r] += d_A[r] * pr;
         d_A += Arow;
      }

      // step 2: RAP[:, j] += R^t * tmp
      RAP.GetColumnReference(j, RAP_col);
      AddMultTransposeSubMatrix(R, Rrows, tmp, RAP_col);

      // Move the pointer to the next column of P
      d_P += Pheight;
   }
}

void TensorContract(const DenseTensor &tensor, const Vector &xi, const Vector &xj, Vector &yk)
{
   assert(xi.Size() == tensor.SizeI());
   assert(xj.Size() == tensor.SizeJ());
   yk.SetSize(tensor.SizeK());

   const double *tdata = tensor.Data();
   const double *dxi, *dxj;
   double *dyk = yk.HostWrite();
   for (int k = 0; k < tensor.SizeK(); k++, dyk++)
   {
      (*dyk) = 0.0;

      dxj = xj.HostRead();
      for (int j = 0; j < tensor.SizeJ(); j++, dxj++)
      {
         dxi = xi.HostRead();
         for (int i = 0; i < tensor.SizeI(); i++, dxi++, tdata++)
         {
            (*dyk) += (*tdata) * (*dxi) * (*dxj);
         }
      }
   }

   return;
}

void TensorAddScaledContract(const DenseTensor &tensor, const double w, const Vector &xi, const Vector &xj, Vector &yk)
{
   assert(xi.Size() == tensor.SizeI());
   assert(xj.Size() == tensor.SizeJ());
   yk.SetSize(tensor.SizeK());

   const double *tdata = tensor.Data();
   const double *dxi, *dxj;
   double *dyk = yk.HostWrite();
   for (int k = 0; k < tensor.SizeK(); k++, dyk++)
   {
      dxj = xj.HostRead();
      for (int j = 0; j < tensor.SizeJ(); j++, dxj++)
      {
         dxi = xi.HostRead();
         for (int i = 0; i < tensor.SizeI(); i++, dxi++, tdata++)
         {
            (*dyk) += w * (*tdata) * (*dxi) * (*dxj);
         }
      }
   }

   return;
}

void TensorAddMultTranspose(const DenseTensor &tensor, const Vector &x, const int axis, DenseMatrix &M)
{
   int size_ax, size_nx;
   int offset, stride;
   switch (axis)
   {
      case 0:
      {
         assert(x.Size() == tensor.SizeI());
         assert(M.NumRows() == tensor.SizeK());
         assert(M.NumCols() == tensor.SizeJ());
         size_ax = tensor.SizeI();
         size_nx = tensor.SizeJ();
         offset = tensor.SizeI();
         stride = 1;
      }
      break;
      case 1:
      {
         assert(x.Size() == tensor.SizeJ());
         assert(M.NumRows() == tensor.SizeK());
         assert(M.NumCols() == tensor.SizeI());
         size_ax = tensor.SizeJ();
         size_nx = tensor.SizeI();
         offset = 1;
         stride = tensor.SizeI();
      }
      break;
      default:
         mfem_error("TensorMult axis should be between 0 and 1!\n");
         break;
   }

   const double *tdata, *dx;
   double *dM = M.HostReadWrite();

   for (int nx = 0; nx < size_nx; nx++)
   {
      for (int k = 0; k < tensor.SizeK(); k++, dM++)
      {
         // commented for AddMultTranspose.
         // (*dM) = 0.0;

         dx = x.HostRead();
         tdata = tensor.GetData(k) + offset * nx;
         for (int ax = 0; ax < size_ax; ax++, dx++)
         {
            (*dM) += (*tdata) * (*dx);
            tdata += stride;
         }
      }
   }

   return;
}

void TensorAddScaledMultTranspose(const DenseTensor &tensor, const double w, const Vector &x, const int axis, DenseMatrix &M)
{
   int size_ax, size_nx;
   int offset, stride;
   switch (axis)
   {
      case 0:
      {
         assert(x.Size() == tensor.SizeI());
         assert(M.NumRows() == tensor.SizeK());
         assert(M.NumCols() == tensor.SizeJ());
         size_ax = tensor.SizeI();
         size_nx = tensor.SizeJ();
         offset = tensor.SizeI();
         stride = 1;
      }
      break;
      case 1:
      {
         assert(x.Size() == tensor.SizeJ());
         assert(M.NumRows() == tensor.SizeK());
         assert(M.NumCols() == tensor.SizeI());
         size_ax = tensor.SizeJ();
         size_nx = tensor.SizeI();
         offset = 1;
         stride = tensor.SizeI();
      }
      break;
      default:
         mfem_error("TensorMult axis should be between 0 and 1!\n");
         break;
   }

   const double *tdata, *dx;
   double *dM = M.HostReadWrite();

   for (int nx = 0; nx < size_nx; nx++)
   {
      for (int k = 0; k < tensor.SizeK(); k++, dM++)
      {
         // commented for AddMultTranspose.
         // (*dM) = 0.0;

         dx = x.HostRead();
         tdata = tensor.GetData(k) + offset * nx;
         for (int ax = 0; ax < size_ax; ax++, dx++)
         {
            (*dM) += w * (*tdata) * (*dx);
            tdata += stride;
         }
      }
   }

   return;
}

double CGOptimizer::Objective(const Vector &rhs, const Vector &sol) const
{
   assert(oper);
   resvec = 0.0;
   oper->Mult(sol, resvec);
   resvec -= rhs;
   return (resvec * resvec);
}

void CGOptimizer::Gradient(const Vector &rhs, const Vector &sol, Vector &grad) const
{
   assert(oper);
   grad.SetSize(sol.Size());
   resvec = 0.0;
   grad = 0.0;

   oper->Mult(sol, resvec);
   resvec -= rhs;
   resvec *= 2.0;
   Operator *jac = &(oper->GetGradient(sol));
   jac->MultTranspose(resvec, grad);
   return;
}

double CGOptimizer::Brent(const Vector &rhs, const Vector &xi, Vector &sol, double b0, double lin_tol) const
{
   sol1.SetSize(sol.Size());

   // initial bracket
   Vector step(N_mnbrak+1), J_brak(N_mnbrak);
   step = 0.0;
   J_brak = 0.0;

   J_brak[0] = Objective(rhs, sol);
   add(sol, b0, xi, sol1); // sol1 = sol + b0 * xi;
   double J1 = Objective(rhs, sol1);
   while (J1 > J_brak[0])
   {
      b0 /= golden_ratio;
      add(sol, b0, xi, sol1); // sol1 = sol + b0 * xi;
      J1 = Objective(rhs, sol1);
   }

   double amp = b0;
   double a, b, c, fa, fb, fc;
   for (int j = 1; j < N_mnbrak+1; j++)
   {
      add(sol, amp, xi, sol1); // sol1 = sol + amp * xi;
      step[j] = amp;
      J_brak[j] = Objective(rhs, sol1);
      
      if (J_brak[j] > J_brak[j-1])
      {
         a = step[j-2]; b = step[j-1]; c = step[j];
         fa = J_brak[j-2]; fb = J_brak[j-1]; fc = J_brak[j];
         break;
      }
      else
         amp *= golden_ratio;
   }

   assert ((a >= 0) && (b > a) && (c > b));
   Array<double> x_arr(4), J_arr(4);

   // parabolic estimation
   for (int j = 0; j < N_para; j++)
   {
      if ( (c-a) < b * lin_tol + eps ) break;
      
      double b_new = b - 0.5 * ( pow(b-a, 2) * (fb-fc) - pow(b-c, 2) * (fb-fa) ) / ( (b-a) * (fb-fc) - (b-c) * (fb-fa) );
      if ((b_new > c) || (b_new < a) || (abs(log10((c-b) / (b-a))) > 1.0))
         if ( b > 0.5 * (a+c) )
            b_new = b - Cr * (b-a);
         else
            b_new = b + Cr * (c-b);
      
      add(sol, b_new, xi, sol1); // sol1 = sol + b_new * xi;
      double fbx = Objective(rhs, sol1);
      
      x_arr[0] = a; x_arr[3] = c;
      J_arr[0] = fa; J_arr[3] = fc;
      if (b_new > b)
      {
         x_arr[1] = b; x_arr[2] = b_new;
         J_arr[1] = fb, J_arr[2] = fbx;
      }
      else
      {
         x_arr[1] = b_new; x_arr[2] = b;
         J_arr[1] = fbx, J_arr[2] = fb;
      }

      fb = J_arr.Min();
      int idx = J_arr.Find(fb); assert(idx > 0);
      fa = J_arr[idx-1]; fc = J_arr[idx+1];
      a = x_arr[idx-1]; b = x_arr[idx]; c = x_arr[idx+1];
   }
   
   sol.Add(b, xi); // sol += b * xi;

   return b;
}

void CGOptimizer::Mult(const Vector &rhs, Vector &sol) const
{
   g.SetSize(sol.Size());
   xi.SetSize(sol.Size());
   xi1.SetSize(sol.Size());
   h.SetSize(sol.Size());

   double Jmin = Objective(rhs, sol);
   double res0 = sqrt(Jmin);
   printf("Initial iteration: ||r|| = %.4E\n", res0);

   Gradient(rhs, sol, g);

   g *= -1.0; xi = g; h = g;
   double gg0 = g * g;

   double b0 = ib / sqrt(gg0);

   double gg, gg1, dgg, dgg1, res;
   double gamma, gamma_FR;
   for (int j = 0; j < max_iter; j++)
   {
      // sol is updated within Brent.
      b0 = Brent(rhs, xi, sol, b0);

      // evaluate one more time to obtain sub-objective functional.
      Jmin = Objective(rhs, sol);
      
      Gradient(rhs, sol, xi);
      gg = g * g;
      
      add(xi, g, xi1);
      dgg = xi1 * xi;
      dgg1 = xi * xi;
      gg1 = g * xi;
      
      res = sqrt(Jmin);
      printf("Iter %d: ||r|| = %.4E, ||r|| / ||r0|| = %.4E\n", j, res, res / res0);
      if ((res < abs_tol) || (res / res0 < rel_tol))
      {
         converged = true;
         printf("CG minimization finished.");
         break;
      }
         
      gamma = dgg / gg;
      gamma_FR = dgg1 / gg;
      if (j > 0)
         if (gamma < -gamma_FR)
            gamma = -gamma_FR;
         else if (gamma > gamma_FR)
            gamma = gamma_FR;
      
      g.Set(-1.0, xi);
      add(g, gamma, h, xi);
      h = xi;
   }

   return;
}

}
