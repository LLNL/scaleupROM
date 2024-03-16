// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "etc.hpp"
#include "linalg_utils.hpp"
#include "nonlinear_integ.hpp"
#include "hyperreduction_integ.hpp"
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"
#include "linalg/NNLS.h"
#include "rom_nonlinearform.hpp"
#include "hdf5_utils.hpp"
#include "steady_ns_solver.hpp"

using namespace std;
using namespace mfem;

static double nu = 0.1;
static double zeta = 1.0;
static bool direct_solve = true;

enum Mode { MMS, SAMPLE, BUILD, COMPARE, PROJECT, NUM_MODE };
enum RomMode { TENSOR, TENSOR2, TENSOR3, TENSOR4, TENSOR5, EQP, NUM_ROMMODE };

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);
void dudx_ex(const Vector & x, Vector & y);
void uux_ex(const Vector & x, Vector & y);

double error(Operator &M, Vector &x, Vector &b)
{
   assert(x.Size() == b.Size());

   Vector res(x.Size());
   M.Mult(x, res);
   res -= b;

   double tmp = 0.0;
   for (int k = 0; k < x.Size(); k++)
      tmp = max(tmp, abs(res(k)));
   return tmp;
}

namespace problem
{

Vector u0, du, offsets;
DenseMatrix k;

void ubdr(const Vector &x, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   
   for (int i = 0; i < dim; i++)
   {
      double kx = 0.0;
      for (int j = 0; j < dim; j++) kx += k(j, i) * x(j);
      kx -= offsets(i);
      kx *= 2.0 * (4.0 * atan(1.0));
      y(i) = u0(i) + du(i) * sin(kx);
   }
}

}

double diff(Vector &a, Vector &b)
{
   assert(a.Size() == b.Size());

   Vector res(a.Size());
   res = a;
   res -= b;

   double tmp = 0.0;
   for (int k = 0; k < a.Size(); k++)
      tmp = max(tmp, abs(res(k)));
   return tmp;
}

/** Nonlinear operator of the form:
    k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
    where M and S are given BilinearForms, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class SteadyNavierStokes : public Operator
{
protected:
   int dim = -1;
   int order = -1;

   double sigma = -1.0;
   // DG terms are employed for velocity space, which is order+1. resulting kappa becomes (order+2)^2.
   double kappa = -1.0;

   bool pres_dbc = false;
   bool use_dg = false;

   Mesh *mesh = NULL;
   FiniteElementCollection *ufec, *pfec;
   FiniteElementSpace *ufes, *pfes;

   ConstantCoefficient nu, zeta;
   ConstantCoefficient zero, one, minus_one, half, minus_half;
   VectorFunctionCoefficient *ubdr = NULL;

   BlockVector *x, *rhs;
   GridFunction *u, *p;

   LinearForm *F, *G;
   BilinearForm *M;
   MixedBilinearFormDGExtension *S;
   NonlinearForm *H;
   Array<int> block_offsets, vblock_offsets;
   Array<int> u_ess_attr, p_ess_attr;

   const IntegrationRule *ir_nl = NULL;

   mutable BlockMatrix *system_jac;
   mutable SparseMatrix *mono_jac, *uu, *up, *pu;

   double atol=1.0e-10, rtol=1.0e-10;
   int maxIter=10000;
   Solver *J_solver = NULL;
   GMRESSolver *J_gmres = NULL;
   MUMPSSolver *J_mumps = NULL;
   NewtonSolver *newton_solver;

   mutable StopWatch solveTimer, multTimer, jacTimer;
   mutable int num_mult, num_jac;
   
   // mutable BlockDiagonalPreconditioner *jac_prec;
   // BilinearForm *pMass = NULL;
   // SparseMatrix *pM = NULL;
   // GSSmoother *p_prec = NULL;
   // OrthoSolver *ortho_p_prec = NULL;

   HYPRE_BigInt glob_size;
   mutable HYPRE_BigInt row_starts[2];
   mutable HypreParMatrix *jac_hypre = NULL;
   // mutable HypreBoomerAMG *u_prec = NULL;

public:
   SteadyNavierStokes(Mesh *mesh_, const int order_, const double nu_ = 1.0, const double zeta_ = 1.0,
                      const bool use_dg_ = false, const bool pres_dbc_ = false)
      : mesh(mesh_), dim(mesh_->Dimension()), order(order_), sigma(-1.0), kappa((order_+2)*(order_+2)),
        nu(nu_), zeta(zeta_), use_dg(use_dg_), pres_dbc(pres_dbc_),
        zero(0.0), one(1.0), minus_one(-1.0), half(0.), minus_half(-0.5),
        system_jac(NULL), mono_jac(NULL), uu(NULL), up(NULL), pu(NULL),
        num_mult(0), num_jac(0)
   {
      if (use_dg)
      {
         ufec = new DG_FECollection(order+1, dim);
         pfec = new DG_FECollection(order, dim);
      }
      else
      {
         ufec = new H1_FECollection(order+1, dim);
         pfec = new H1_FECollection(order, dim);
      }
      ufes = new FiniteElementSpace(mesh, ufec, dim);
      pfes = new FiniteElementSpace(mesh, pfec);

      block_offsets.SetSize(dim+2); // number of variables + 1
      block_offsets[0] = 0;
      for (int d = 1; d <= dim; d++)
         block_offsets[d] = ufes->GetNDofs();
      block_offsets[dim+1] = pfes->GetVSize();
      block_offsets.PartialSum();

      std::cout << "***********************************************************\n";
      for (int d = 1; d < block_offsets.Size(); d++)
         printf("dim(%d) = %d\n", d, block_offsets[d] - block_offsets[d-1]);
      printf("dim(q) = %d\n", block_offsets.Last());
      std::cout << "***********************************************************\n";

      vblock_offsets.SetSize(3); // number of variables + 1
      vblock_offsets[0] = 0;
      vblock_offsets[1] = ufes->GetVSize();
      vblock_offsets[2] = pfes->GetVSize();
      vblock_offsets.PartialSum();

      std::cout << "***********************************************************\n";
      for (int d = 0; d < vblock_offsets.Size(); d++)
         printf("vblock_offset(%d) = %d\n", d, vblock_offsets[d]);
      std::cout << "***********************************************************\n";

      height = width = vblock_offsets.Last();
      glob_size = height;
      row_starts[0] = 0;
      row_starts[1] = height;

      x = new BlockVector(vblock_offsets);
      rhs = new BlockVector(vblock_offsets);
      (*x) = 0.0;

      u = new GridFunction;
      p = new GridFunction;
      u->MakeRef(ufes, x->GetBlock(0), 0);
      p->MakeRef(pfes, x->GetBlock(1), 0);

      u_ess_attr.SetSize(mesh->bdr_attributes.Max());
      p_ess_attr.SetSize(mesh->bdr_attributes.Max());
      u_ess_attr = 1;
      p_ess_attr = 0;

      F = new LinearForm;
      F->Update(ufes, rhs->GetBlock(0), 0);
      G = new LinearForm;
      G->Update(pfes, rhs->GetBlock(1), 0);

      M = new BilinearForm(ufes);
      S = new MixedBilinearFormDGExtension(ufes, pfes);
      H = new NonlinearForm(ufes);

      M->AddDomainIntegrator(new VectorDiffusionIntegrator(nu));
      if (use_dg)
         M->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(nu, sigma, kappa));

      S->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
      if (use_dg)
         S->AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);

      ir_nl = &(IntRules.Get(ufes->GetFE(0)->GetGeomType(), (int)(ceil(1.5 * (2 * ufes->GetMaxElementOrder() - 1)))));
      auto nl_integ = new VectorConvectionTrilinearFormIntegrator(zeta);
      nl_integ->SetIntRule(ir_nl);
      H->AddDomainIntegrator(nl_integ);

      // if (use_dg)
      //    // nVarf->AddInteriorFaceIntegrator(new DGLaxFriedrichsFluxIntegrator(one));
      //    nVarf->AddInteriorFaceIntegrator(new DGTemamFluxIntegrator(half_zeta));

      if (direct_solve)
      {
         J_mumps = new MUMPSSolver(MPI_COMM_WORLD);
         J_mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
         J_solver = J_mumps;
      }
      else
      {
         J_gmres = new GMRESSolver;
         J_gmres->SetAbsTol(atol);
         J_gmres->SetRelTol(rtol);
         J_gmres->SetMaxIter(maxIter);
         J_gmres->SetPrintLevel(-1);
         J_solver = J_gmres;
      }

      newton_solver = new NewtonSolver;
      newton_solver->SetSolver(*J_solver);
      newton_solver->SetOperator(*this);
      newton_solver->SetPrintLevel(1); // print Newton iterations
      newton_solver->SetRelTol(rtol);
      newton_solver->SetAbsTol(atol);
      newton_solver->SetMaxIter(100);
   }
        
   void SetupMMS(VectorFunctionCoefficient &fcoeff, FunctionCoefficient &gcoeff,
                 VectorFunctionCoefficient &ucoeff)
   {
      u_ess_attr = 1;
      pres_dbc = false;

      F->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      F->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(ucoeff, nu, sigma, kappa), u_ess_attr);
      // F->AddBdrFaceIntegrator(new DGBdrTemamLFIntegrator(ucoeff, &minus_half_zeta), u_ess_attr);
      F->Assemble();
      F->SyncAliasMemory(*rhs);

      G->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
      G->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), u_ess_attr);
      G->Assemble();
      G->SyncAliasMemory(*rhs);

      M->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(nu, sigma, kappa), u_ess_attr);
      M->Assemble();
      M->Finalize();

      S->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, u_ess_attr);
      S->Assemble();
      S->Finalize();

      pu = &(S->SpMat());
      up = Transpose(*pu);
   }

   void SetupProblem()
   {
      //     3
      //     __
      //  4 |  |  2
      //     --
      //     1
      // 5: cylinder

      u_ess_attr = 0;
      pres_dbc = true;
      delete ubdr;
      ubdr = new VectorFunctionCoefficient(dim, problem::ubdr);

      if (problem::u0(0) >= 0.0)
         u_ess_attr[3] = 1;
      else
         u_ess_attr[1] = 1;

      if (problem::u0(1) >= 0.0)
         u_ess_attr[0] = 1;
      else
         u_ess_attr[2] = 1;

      F->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(*ubdr, nu, sigma, kappa), u_ess_attr);
      // F->AddBdrFaceIntegrator(new DGBdrTemamLFIntegrator(ucoeff, &minus_half_zeta), u_ess_attr);
      F->Assemble();
      F->SyncAliasMemory(*rhs);

      G->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(*ubdr), u_ess_attr);
      G->Assemble();
      G->SyncAliasMemory(*rhs);

      // no-slip wall cylinder
      u_ess_attr.Last() = 1;

      M->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(nu, sigma, kappa), u_ess_attr);
      M->Assemble();
      M->Finalize();

      S->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, u_ess_attr);
      S->Assemble();
      S->Finalize();

      pu = &(S->SpMat());
      up = Transpose(*pu);
   }

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &x, Vector &y) const
   {
      multTimer.Start();
      Vector x_u(x.GetData()+vblock_offsets[0], M->Height()), x_p(x.GetData()+vblock_offsets[1], S->Height());
      Vector y_u(y.GetData(), M->Height()), y_p(y.GetData()+vblock_offsets[1], S->Height());

      H->Mult(x_u, y_u);
      M->AddMult(x_u, y_u);
      S->AddMultTranspose(x_p, y_u);
      S->Mult(x_u, y_p);

      multTimer.Stop();
      num_mult++;
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const
   {
      jacTimer.Start();

      delete system_jac;
      delete mono_jac;
      delete jac_hypre;
      delete uu;
      const Vector x_u(x.GetData()+vblock_offsets[0], M->Height()), x_p(x.GetData()+vblock_offsets[1], S->Height());

      SparseMatrix *grad_H = dynamic_cast<SparseMatrix *>(&H->GetGradient(x_u));
      uu = Add(1.0, M->SpMat(), 1.0, *grad_H);

      assert(up && pu);

      system_jac = new BlockMatrix(vblock_offsets);
      system_jac->SetBlock(0,0, uu);
      system_jac->SetBlock(0,1, up);
      system_jac->SetBlock(1,0, pu);

      // // update preconditioner.
      // delete u_prec
      // delete uu_hypre;
      // uu_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, uu);
      // u_prec = new HypreBoomerAMG(*uu_hypre);
      // u_prec->SetPrintLevel(0);
      // u_prec->SetSystemsOptions(dim, true);
      // jac_prec->SetDiagonalBlock(0, u_prec);

      mono_jac = system_jac->CreateMonolithic();
      if (direct_solve)
      {
         jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, mono_jac);
      }

      jacTimer.Stop();
      num_jac++;
      if (direct_solve)
         return *jac_hypre;
      else
         return *mono_jac;
   }

   virtual ~SteadyNavierStokes()
   {
      delete system_jac;
      delete mono_jac;
      delete jac_hypre;
      delete uu;
      delete up;
      delete M;
      delete S;
      delete H;
      delete F;
      delete G;
      delete x;
      delete rhs;
      delete u;
      delete p;
      delete ufec;
      delete pfec;
      delete ufes;
      delete pfes;
      delete J_gmres;
      delete J_mumps;
      delete newton_solver;
      // delete ir_nl;
      // delete jac_prec;
      // delete pMass
      // delete p_prec, ortho_p_prec;
   }

   // BlockDiagonalPreconditioner* GetGradientPreconditioner() { return jac_prec; }
   virtual bool Solve()
   {
      printf("FOM solve: %.3E sec\n", solveTimer.RealTime());
      solveTimer.Start();

      newton_solver->Mult(*rhs, *x);
      if (!pres_dbc)
      {
         (*p) -= p->Sum() / p->Size();
      }

      solveTimer.Stop();
      printf("FOM mult: %.3E sec\n", multTimer.RealTime());
      printf("FOM jac: %.3E sec\n", jacTimer.RealTime());
      printf("FOM solve: %.3E sec\n", solveTimer.RealTime());

      // ResetTimer();

      return newton_solver->GetConverged();
   }

   SparseMatrix* GetLinearTerm()
   {
      assert(up && pu);

      BlockMatrix tmp(vblock_offsets);
      tmp.SetBlock(0,0, &(M->SpMat()));
      tmp.SetBlock(0,1, up);
      tmp.SetBlock(1,0, pu);

      return tmp.CreateMonolithic();
   }

   BlockVector* GetSolution() { return x; }
   void SetSolution(const BlockVector &sol) { (*x) = sol; return; }
   BlockVector* GetRHS() { return rhs; }
   GridFunction* GetVelocityGridFunction() { return u; }
   GridFunction* GetPressureGridFunction() { return p; }
   FiniteElementSpace* GetUfes() { return ufes; }
   FiniteElementSpace* GetPfes() { return pfes; }
   ConstantCoefficient* GetZeta() { return &zeta; }
   const IntegrationRule* GetNonlinearIntRule() { return ir_nl; }

   const double GetAvgMultTime() { return multTimer.RealTime() / static_cast<double>(num_mult); }
   const double GetAvgGradTime() { return jacTimer.RealTime() / static_cast<double>(num_jac); }
   const double GetSolveTime() { return solveTimer.RealTime(); }
   void ResetTimer()
   {
      solveTimer.Clear();
      multTimer.Clear();
      jacTimer.Clear();
   }

   DenseTensor* GetReducedTensor(DenseMatrix &proj_basis, DenseMatrix &basis)
   {
      const int num_basis = basis.NumCols();
      const int num_proj_basis = proj_basis.NumCols();

      DenseTensor *nlin_rom = new DenseTensor(num_basis, num_basis, num_proj_basis);
      Vector tmp(ufes->GetVSize());
      // DenseTensor is column major and i is the fastest index. 
      // For fast iteration, we set k to be the test function index.
      for (int i = 0; i < num_basis; i++)
      {
         // Vector basis_i(basis.GetColumn(i), basis.NumRows());
         Vector u_i(basis.GetColumn(i), ufes->GetVSize());
         GridFunction ui_gf(ufes, u_i.GetData());
         VectorGridFunctionCoefficient ui_coeff(&ui_gf);
         NonlinearForm Hi(ufes);
         auto nl_integ_tmp = new VectorConvectionTrilinearFormIntegrator(zeta, &ui_coeff);
         nl_integ_tmp->SetIntRule(ir_nl);
         Hi.AddDomainIntegrator(nl_integ_tmp);
         for (int j = 0; j < num_basis; j++)
         {
            // Vector basis_j(basis.GetColumn(j), basis.NumRows());
            Vector u_j(basis.GetColumn(j), ufes->GetVSize());
            tmp = 0.0;
            Hi.Mult(u_j, tmp);
            
            for (int k = 0; k < num_proj_basis; k++)
            {
               // Vector basis_k(basis.GetColumn(k), basis.NumRows());
               Vector u_k(proj_basis.GetColumn(k), ufes->GetVSize());
               (*nlin_rom)(i, j, k) = u_k * tmp;
            }
         }
      }

      return nlin_rom;
   }
};

class TensorROM : public Operator
{
protected:
   DenseMatrix *lin_op = NULL;
   DenseTensor *nlin_op = NULL;

   mutable DenseMatrix *jac = NULL;

   HYPRE_BigInt glob_size;
   mutable HYPRE_BigInt row_starts[2];
   Array<int> rows;
   mutable SparseMatrix *jac_mono = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;

public:
   mutable StopWatch multTimer, jacTimer;
   mutable int num_mult, num_jac;

public:
   TensorROM(DenseMatrix &lin_op_, DenseTensor &nlin_op_)
      : Operator(lin_op_.Height(), lin_op_.Width()), lin_op(&lin_op_),
        nlin_op(&nlin_op_), jac(NULL), num_mult(0), num_jac(0)
   {
      glob_size = lin_op->Height();
      row_starts[0] = 0;
      row_starts[1] = lin_op->Height();
      
      rows.SetSize(glob_size);
      for (int k = 0; k < glob_size; k++)
         rows[k] = k;
   }

   virtual ~TensorROM()
   {
      delete jac;
      delete jac_mono;
      delete jac_hypre;

      printf("%10s\t%10s\n", "mult", "grad");
      printf("%.3E\t", multTimer.RealTime());
      printf("%.3E\n", jacTimer.RealTime());
   }

   const double GetAvgMultTime() { return multTimer.RealTime() / static_cast<double>(num_mult); }
   const double GetAvgGradTime() { return jacTimer.RealTime() / static_cast<double>(num_jac); }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      multTimer.Start();

      TensorContract(*nlin_op, x, x, y);
      lin_op->AddMult(x, y);

      multTimer.Stop();
      num_mult++;
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const
   {
      jacTimer.Start();

      delete jac;
      delete jac_hypre;
      delete jac_mono;
      jac = new DenseMatrix(*lin_op);
      TensorAddMultTranspose(*nlin_op, x, 0, *jac);
      TensorAddMultTranspose(*nlin_op, x, 1, *jac);

      if (direct_solve)
      {
         jac_mono = new SparseMatrix(jac->NumRows());
         jac_mono->SetSubMatrix(rows, rows, *jac);
         jac_mono->Finalize();
         jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, jac_mono);
      }

      jacTimer.Stop();
      num_jac++;
      if (direct_solve)
         return *jac_hypre;
      else
         return *jac;
   }

};

class TensorROM2 : public TensorROM
{
protected:
   int size_i = -1;
   int size_k = -1;
   Array<int> ridx, cidx;
   mutable DenseMatrix jac_vel;

public:
   TensorROM2(DenseMatrix &lin_op_, DenseTensor &nlin_op_)
      : TensorROM(lin_op_, nlin_op_), size_i(nlin_op_.SizeI()), size_k(nlin_op_.SizeK()),
        jac_vel(nlin_op_.SizeK(), nlin_op_.SizeI()), ridx(nlin_op_.SizeK()), cidx(nlin_op_.SizeI())
   {
      for (int k = 0; k < ridx.Size(); k++) ridx[k] = k;
      for (int k = 0; k < cidx.Size(); k++) cidx[k] = k;
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      multTimer.Start();

      const Vector x_vel(x.GetData(), size_i);
      Vector y_vel(y.GetData(), size_k);
      y = 0.0;

      TensorContract(*nlin_op, x_vel, x_vel, y_vel);
      lin_op->AddMult(x, y);

      multTimer.Stop();
      num_mult++;
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const override
   {
      jacTimer.Start();

      const Vector x_vel(x.GetData(), size_i);

      jac_vel = 0.0;
      TensorAddMultTranspose(*nlin_op, x_vel, 0, jac_vel);
      TensorAddMultTranspose(*nlin_op, x_vel, 1, jac_vel);

      delete jac;
      delete jac_hypre;
      delete jac_mono;
      jac = new DenseMatrix(*lin_op);
      jac->AddSubMatrix(ridx, cidx, jac_vel);

      if (direct_solve)
      {
         jac_mono = new SparseMatrix(jac->NumRows());
         jac_mono->SetSubMatrix(rows, rows, *jac);
         jac_mono->Finalize();
         jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, jac_mono);
      }

      jacTimer.Stop();
      num_jac++;
      if (direct_solve)
         return *jac_hypre;
      else
         return *jac;
   }

};

class EQPROM : public Operator
{
protected:
   DenseMatrix *lin_op = NULL;
   ROMNonlinearForm *rnlf = NULL;

   mutable DenseMatrix *jac = NULL;

   HYPRE_BigInt glob_size;
   mutable HYPRE_BigInt row_starts[2];
   Array<int> rows;
   mutable SparseMatrix *jac_mono = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;

public:
   mutable StopWatch multTimer, jacTimer;
   mutable int num_mult, num_jac;

public:
   EQPROM(DenseMatrix &lin_op_, ROMNonlinearForm &rnlf_)
      : Operator(lin_op_.Height(), lin_op_.Width()), lin_op(&lin_op_),
        rnlf(&rnlf_), jac(NULL), num_mult(0), num_jac(0)
   {
      glob_size = lin_op->Height();
      row_starts[0] = 0;
      row_starts[1] = lin_op->Height();
      
      rows.SetSize(glob_size);
      for (int k = 0; k < glob_size; k++)
         rows[k] = k;
   }

   ~EQPROM()
   {
      delete jac;
      delete jac_mono;
      delete jac_hypre;

      printf("%10s\t%10s\n", "mult", "grad");
      printf("%.3E\t", multTimer.RealTime());
      printf("%.3E\n", jacTimer.RealTime());
   }

   const double GetAvgMultTime() { return multTimer.RealTime() / static_cast<double>(num_mult); }
   const double GetAvgGradTime() { return jacTimer.RealTime() / static_cast<double>(num_jac); }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      multTimer.Start();
      rnlf->Mult(x, y);
      lin_op->AddMult(x, y);

      multTimer.Stop();
      num_mult++;
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const
   {
      jacTimer.Start();

      delete jac, jac_mono, jac_hypre;
      DenseMatrix *grad_H = dynamic_cast<DenseMatrix *>(&rnlf->GetGradient(x));
      jac = new DenseMatrix(*grad_H);
      *jac += *lin_op;

      if (direct_solve)
      {
         jac_mono = new SparseMatrix(jac->NumRows());
         jac_mono->SetSubMatrix(rows, rows, *jac);
         jac_mono->Finalize();
         jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, jac_mono);
      }

      jacTimer.Stop();
      num_jac++;
      if (direct_solve)
         return *jac_hypre;
      else
         return *jac;
   }

};

int main(int argc, char *argv[])
{
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int refine = 0;
   bool pa = false;
   bool pres_dbc = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   bool use_dg = false;
   const char *mode_str = "";
   Mode mode = Mode::NUM_MODE;
   const char *rom_mode_str = "";
   RomMode rom_mode = RomMode::TENSOR;
   bool random_sample = true;
   int nsample = -1;
   int num_basis = -1;
   int num_pbasis = -1;
   int num_psupreme = -1;  // number of pressure supremizer.
   double eqp_tol = 1.0e-5;
   bool precompute = false;
   const char *compare_output_file = "result.h5";
   bool wgt_basis = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&nu, "-nu", "--nu",
                  "Viscosity.");
   args.AddOption(&zeta, "-zeta", "--zeta",
                  "Scalar coefficient for advection.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.AddOption(&pres_dbc, "-pd", "--pressure-dirichlet",
                  "-no-pd", "--no-pressure-dirichlet",
                  "Use pressure Dirichlet condition.");
   args.AddOption(&use_dg, "-dg", "--use-dg", "-no-dg", "--no-use-dg",
                  "Use discontinuous Galerkin scheme.");
   args.AddOption(&mode_str, "-mode", "--mode",
                  "Mode: mms, sample, compare");
   args.AddOption(&rom_mode_str, "-rm", "--rom-mode",
                  "RomMode: tensor, eqp.");
   args.AddOption(&random_sample, "-rs", "--random-sample", "-no-rs", "--no-random-sample",
                  "Sample will be generated randomly.");
   args.AddOption(&nsample, "-ns", "--nsample",
                  "Number of samples to be generated.");
   args.AddOption(&num_basis, "-nb", "--nbasis",
                  "Number of basis for ROM. For TENSOR2, number of basis for velocity.");
   args.AddOption(&num_pbasis, "-npb", "--npbasis",
                  "Number of pressure basis for ROM. Used only for TENSOR2.");
   args.AddOption(&num_psupreme, "-nps", "--npsupreme",
                  "Number of pressure basis for ROM. Used only for TENSOR4.");
   args.AddOption(&direct_solve, "-ds", "--direct-solve", "-no-ds", "--no-direct-solve",
                  "Use direct or iterative solver.");
   args.AddOption(&eqp_tol, "-et", "--eqp-tolerance",
                  "Tolerance for EQP NNLS solver.");
   args.AddOption(&precompute, "-pre", "--precompute", "-no-pre", "--no-precompute",
                  "Precompute hypre-reduction coefficients.");
   args.AddOption(&compare_output_file, "-of", "--output-file",
                  "Output file name to store the comparison result.");
   args.AddOption(&wgt_basis, "-wb", "--weighted-basis", "-no-wb", "--no-weighted-basis",
                  "Perform generalized SVD for ROM basis with specified weight.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   assert(!pres_dbc);
   if (!strcmp(mode_str, "mms"))             mode = Mode::MMS;
   else if (!strcmp(mode_str, "sample"))     mode = Mode::SAMPLE;
   else if (!strcmp(mode_str, "project"))     mode = Mode::PROJECT;
   else if (!strcmp(mode_str, "build"))     mode = Mode::BUILD;
   else if (!strcmp(mode_str, "compare"))    mode = Mode::COMPARE;
   else
   {
      mfem_error("Unknown mode!\n");
   }

   if (!strcmp(rom_mode_str, "tensor"))          rom_mode = RomMode::TENSOR;  // vel-pres unified basis
   else if (!strcmp(rom_mode_str, "tensor2"))    rom_mode = RomMode::TENSOR2; // vel/pres separated basis
   else if (!strcmp(rom_mode_str, "tensor3"))    rom_mode = RomMode::TENSOR3; // vel-only basis
   else if (!strcmp(rom_mode_str, "tensor4"))    rom_mode = RomMode::TENSOR4; // vel/pres seperated, augmented with supremizer
   else if (!strcmp(rom_mode_str, "tensor5"))    rom_mode = RomMode::TENSOR5; // vel/pres seperated, petrov-galerkin augmented with supremizer
   else if (!strcmp(rom_mode_str, "eqp"))        rom_mode = RomMode::EQP;
   else if (!(rom_mode == RomMode::TENSOR))
      mfem_error("Unknown mode!\n");

   if (wgt_basis && (rom_mode != RomMode::TENSOR))
      mfem_error("TENSOR2 does not support weighted basis!\n");

   if (!random_sample)
   {
      nsample = 4;
      num_basis = nsample;
      num_pbasis = nsample;
   }

   if (rom_mode == RomMode::TENSOR5)
      num_psupreme = num_pbasis;

   std::string filename = "ns_rom";

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file);
   int dim = mesh->Dimension();
   problem::u0.SetSize(dim);
   problem::du.SetSize(dim);
   problem::offsets.SetSize(dim);
   problem::k.SetSize(dim);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   SteadyNavierStokes oper(mesh, order, nu, zeta, use_dg, pres_dbc);
   FiniteElementSpace *ufes = oper.GetUfes();
   FiniteElementSpace *pfes = oper.GetPfes();
   const int udim = ufes->GetTrueVSize();
   const int pdim = pfes->GetTrueVSize();
   const double pres_wgt = static_cast<double>(ufes->GetTrueVSize() / dim) / static_cast<double>(pfes->GetTrueVSize());

   if (mode == Mode::MMS)
   {
      VectorFunctionCoefficient fcoeff(dim, fFun);
      FunctionCoefficient gcoeff(gFun);

      VectorFunctionCoefficient ucoeff(dim, uFun_ex);
      FunctionCoefficient pcoeff(pFun_ex);

      GridFunction u_ex(ufes), p_ex(pfes);
      u_ex.ProjectCoefficient(ucoeff);
      p_ex.ProjectCoefficient(pcoeff);
      const double p_const = p_ex.Sum() / static_cast<double>(p_ex.Size());

      // 12. Create the grid functions u and p. Compute the L2 error norms.
      GridFunction *u = oper.GetVelocityGridFunction();
      GridFunction *p = oper.GetPressureGridFunction();

      oper.SetupMMS(fcoeff, gcoeff, ucoeff);
      oper.Solve();

      if (!pres_dbc)
      {
         (*p) += p_const;
      }

      int order_quad = max(2, 2*(order+1)+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double err_u  = u->ComputeL2Error(ucoeff, irs);
      double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
      double err_p  = p->ComputeL2Error(pcoeff, irs);
      double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

      printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
      printf("|| p_h - p_ex || / || p_ex || = %.5E\n", err_p / norm_p);

      // GridFunction tmp(u);
      // nVarf->Mult(u, tmp);

      // printf("u\ttmp\n");
      // for (int k = 0; k < u.Size(); k++)
      //    printf("%.5E\t%.5E\n", u[k], tmp[k]);

      // 15. Save data in the ParaView format
      ParaViewDataCollection paraview_dc("ns_mms_paraview", mesh);
      // paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      // paraview_dc.SetCycle(0);
   //    paraview_dc.SetDataFormat(VTKFormat::BINARY);
   //    paraview_dc.SetHighOrderOutput(true);
   //    paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("velocity", u);
      paraview_dc.RegisterField("pressure", p);
      paraview_dc.Save();
   }  // if (mode == Mode::MMS)

   if (mode == Mode::SAMPLE)
   {
      assert(nsample > 0);

      const int fom_vdofs = oper.Height();

      CAROM::Options option(fom_vdofs, nsample, 1, false);
      CAROM::Options u_option(udim, nsample, 1, false);
      CAROM::Options p_option(pdim, nsample, 1, false);
      CAROM::BasisGenerator snapshot_generator(option, false, filename);
      CAROM::BasisGenerator u_snapshot_generator(u_option, false, filename + "_vel");
      CAROM::BasisGenerator p_snapshot_generator(p_option, false, filename + "_pres");

      int s = 0;
      while (s < nsample)
      {
         if (random_sample)
         {
            for (int d = 0; d < problem::u0.Size(); d++)
            {
               problem::u0(d) = 2.0 * UniformRandom() - 1.0;
               problem::du(d) = 0.1 * (2.0 * UniformRandom() - 1.0);
               problem::offsets(d) = UniformRandom();

               for (int d2 = 0; d2 < problem::u0.Size(); d2++)
                  problem::k(d, d2) = 0.5 * (2.0 * UniformRandom() - 1.0);
            }
         }
         else
         {
            problem::du = 0.0;
            problem::offsets = 0.0;
            problem::k = 0.0;

            problem::u0(0) = (s / 2 == 0) ? 1.0 : -1.0;
            problem::u0(1) = (s % 2 == 0) ? 1.0 : -1.0;
            printf("u0: (%f, %f)\n", problem::u0(0), problem::u0(1));
         }

         SteadyNavierStokes temp(mesh, order, nu, zeta, use_dg, pres_dbc);
         temp.SetupProblem();
         bool converged = temp.Solve();
         if (!converged)
         {
            if (random_sample)
            {
               mfem_warning("Failed to converge in sampling. Collecting another sample.\n");
               continue;
            }
            else
               mfem_error("Failed to converge in sampling!\n");
         }

         snapshot_generator.takeSample(temp.GetSolution()->GetData());
         u_snapshot_generator.takeSample(temp.GetSolution()->GetData());
         p_snapshot_generator.takeSample(temp.GetSolution()->GetData() + udim);

         // GridFunction *u = temp.GetVelocityGridFunction();
         // GridFunction *p = temp.GetPressureGridFunction();

         // // 15. Save data in the ParaView format
         // ParaViewDataCollection paraview_dc("ns_sample_paraview", mesh);
         // paraview_dc.SetLevelsOfDetail(max(3,order+1));
         // paraview_dc.RegisterField("velocity", u);
         // paraview_dc.RegisterField("pressure", p);
         // paraview_dc.Save();

         s++;
      }  // while (s < nsample)

      snapshot_generator.writeSnapshot();
      u_snapshot_generator.writeSnapshot();
      p_snapshot_generator.writeSnapshot();

      snapshot_generator.endSamples();
      const CAROM::Vector *rom_sv = snapshot_generator.getSingularValues();
      printf("Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3E\t", rom_sv->item(d));
      printf("\n");
      std::string sv_file = filename + "_sv.txt";
      CAROM::PrintVector(*rom_sv, sv_file);

      u_snapshot_generator.endSamples();
      rom_sv = u_snapshot_generator.getSingularValues();
      printf("Velocity Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3E\t", rom_sv->item(d));
      printf("\n");
      sv_file = filename + "_vel_sv.txt";
      CAROM::PrintVector(*rom_sv, sv_file);

      p_snapshot_generator.endSamples();
      rom_sv = p_snapshot_generator.getSingularValues();
      printf("Pressure Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3E\t", rom_sv->item(d));
      printf("\n");
      sv_file = filename + "_pres_sv.txt";
      CAROM::PrintVector(*rom_sv, sv_file);
   }  // if (mode == Mode::SAMPLE)

   if (mode == Mode::PROJECT)
   {
      assert(nsample > 0);
      assert(num_basis > 0);
      if (wgt_basis)
         mfem_error("not implemented for weighted basis!\n");

      double sigma = -1.0, kappa = (order+2) * (order+2);
      ConstantCoefficient nu_coeff(nu), zeta_coeff(zeta);
      ConstantCoefficient minus_one(-1.0);
      const IntegrationRule *ir = oper.GetNonlinearIntRule();

      BilinearForm M(ufes);
      NonlinearForm H(ufes);
      MixedBilinearFormDGExtension S(ufes, pfes);

      M.AddDomainIntegrator(new VectorDiffusionIntegrator(nu_coeff));
      if (use_dg)
         M.AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(nu_coeff, sigma, kappa));

      S.AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
      if (use_dg)
         S.AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);

      auto nl_integ = new VectorConvectionTrilinearFormIntegrator(zeta_coeff);
      nl_integ->SetIntRule(ir);
      H.AddDomainIntegrator(nl_integ);

      M.Assemble();
      S.Assemble();
      M.Finalize();
      S.Finalize();
      PrintMatrix(S.SpMat(), filename + "_div_op.txt");

      DenseMatrix basis;
      CAROM::BasisReader basis_reader(filename);
      const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis);
      CAROM::CopyMatrix(*carom_basis, basis);

      const int fom_vdofs = oper.Height();

      std::string res_filename = filename + "_res";
      std::string proj_res_filename = filename + "_proj_res";
      CAROM::Options option(fom_vdofs, nsample, 1, false);
      CAROM::BasisGenerator sol_snapshot(option, false, filename + "_sol");
      CAROM::BasisGenerator proj_sol_snapshot(option, false, filename + "_proj_sol");
      CAROM::BasisGenerator residual_snapshot(option, false, res_filename);
      CAROM::BasisGenerator proj_residual_snapshot(option, false, proj_res_filename);

      for (int s = 0; s < nsample; s++)
      {
         for (int d = 0; d < problem::u0.Size(); d++)
         {
            problem::u0(d) = 2.0 * UniformRandom() - 1.0;
            problem::du(d) = 0.1 * (2.0 * UniformRandom() - 1.0);
            problem::offsets(d) = UniformRandom();

            for (int d2 = 0; d2 < problem::u0.Size(); d2++)
               problem::k(d, d2) = 0.5 * (2.0 * UniformRandom() - 1.0);
         }

         SteadyNavierStokes fom(mesh, order, nu, zeta, use_dg, pres_dbc);
         fom.SetupProblem();
         bool converged = fom.Solve();
         assert(converged);

         BlockVector *fom_sol = fom.GetSolution();
         BlockVector *fom_rhs = fom.GetRHS();
         Vector tmp1(num_basis), fom_proj(fom_sol->Size()), fom_res(fom_sol->Size());
         basis.MultTranspose(*fom_sol, tmp1);
         basis.Mult(tmp1, fom_proj);

         {
            Vector x_u(fom_proj.GetData(), M.Height()), x_p(fom_proj.GetData()+M.Height(), S.Height());
            Vector y_u(fom_res.GetData(), M.Height()), y_p(fom_res.GetData()+M.Height(), S.Height());

            H.Mult(x_u, y_u);
            M.AddMult(x_u, y_u);
            S.AddMultTranspose(x_p, y_u);
            S.Mult(x_u, y_p);
         }
         // fom.Mult(fom_proj, fom_res);
         // fom_res -= (*fom_rhs);

         Vector proj_res(fom_sol->Size());
         basis.MultTranspose(fom_res, tmp1);
         basis.Mult(tmp1, proj_res);

         Vector fomsol_div(pdim), projsol_div(pdim), res_div(pdim), projres_div(pdim);
         S.Mult(fom_sol->GetBlock(0), fomsol_div);
         tmp1.MakeRef(fom_proj, 0, udim);
         S.Mult(tmp1, projsol_div);
         tmp1.MakeRef(fom_res, 0, udim);
         S.Mult(tmp1, res_div);
         tmp1.MakeRef(proj_res, 0, udim);
         S.Mult(tmp1, projres_div);

         sol_snapshot.takeSample(fom_sol->GetData());
         proj_sol_snapshot.takeSample(fom_proj.GetData());
         residual_snapshot.takeSample(fom_res.GetData());
         proj_residual_snapshot.takeSample(proj_res.GetData());

         Array<GridFunction *> ugf(4), pgf(4), divgf(4);
         for (int k = 0; k < 4; k++) ugf[k] = new GridFunction;
         for (int k = 0; k < 4; k++) pgf[k] = new GridFunction;
         for (int k = 0; k < 4; k++) divgf[k] = new GridFunction;
         ugf[0]->MakeRef(ufes, fom_sol->GetBlock(0), 0);
         pgf[0]->MakeRef(pfes, fom_sol->GetBlock(1), 0);
         divgf[0]->MakeRef(pfes, fomsol_div, 0);
         ugf[1]->MakeRef(ufes, fom_proj, 0);
         pgf[1]->MakeRef(pfes, fom_proj, fom_sol->GetBlock(0).Size());
         divgf[1]->MakeRef(pfes, projsol_div, 0);
         ugf[2]->MakeRef(ufes, fom_res, 0);
         pgf[2]->MakeRef(pfes, fom_res, fom_sol->GetBlock(0).Size());
         divgf[2]->MakeRef(pfes, res_div, 0);
         ugf[3]->MakeRef(ufes, proj_res, 0);
         pgf[3]->MakeRef(pfes, proj_res, fom_sol->GetBlock(0).Size());
         divgf[3]->MakeRef(pfes, projres_div, 0);

         std::vector<std::string> suffix(0);
         suffix.push_back("_fomsol");
         suffix.push_back("_projsol");
         suffix.push_back("_fomres");
         suffix.push_back("_projres");

         // 15. Save data in the ParaView format
         ParaViewDataCollection paraview_dc("ns_project_paraview_"+std::to_string(s), mesh);
         paraview_dc.SetLevelsOfDetail(max(3,order+1));
         for (int k = 0; k < 4; k++)
         {
            paraview_dc.RegisterField("u"+suffix[k], ugf[k]);
            paraview_dc.RegisterField("p"+suffix[k], pgf[k]);
            paraview_dc.RegisterField("div"+suffix[k], divgf[k]);
         }
         paraview_dc.Save();
      }

      sol_snapshot.writeSnapshot();
      proj_sol_snapshot.writeSnapshot();
      residual_snapshot.writeSnapshot();
      proj_residual_snapshot.writeSnapshot();
   }  // if (mode == Mode::PROJECT)

   if (mode == Mode::BUILD)
   {
      oper.SetupProblem();

      DenseMatrix basis;
      if (wgt_basis)
      {
         printf("weight on pressure: %.3E\n", pres_wgt);

         std::string basis_file = filename + "_basis.h5";
         if (FileExists(basis_file))
         {  // load basis from a hdf5 format.
            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(basis_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "basis", basis);

            errf = H5Fclose(file_id);
            assert(errf >= 0);

            if (num_basis != basis.NumCols())
            {
               assert(num_basis < basis.NumCols());
               basis.SetSize(basis.NumRows(), num_basis);
            }
         }  // if (FileExists(basis_file))
         else
         {  // perform generalized SVD with a weight on the pressure.
            assert(nsample > 0);
            const int fom_vdofs = oper.Height();
            CAROM::Options option(fom_vdofs, nsample, 1, false);
            CAROM::BasisGenerator snapshot_reader(option, false, filename);
            snapshot_reader.loadSamples(filename + "_snapshot", "snapshot");
            const CAROM::Matrix *snapshots = snapshot_reader.getSnapshotMatrix();
            DenseMatrix wgted_snapshots;
            CAROM::CopyMatrix(*snapshots, wgted_snapshots);
            for (int i = ufes->GetTrueVSize(); i < wgted_snapshots.NumRows(); i++)
               for (int j = 0; j < wgted_snapshots.NumCols(); j++)
                  wgted_snapshots(i,j) *= sqrt(pres_wgt);

            CAROM::BasisGenerator basis_generator(option, false, filename);
            for (int j = 0; j < wgted_snapshots.NumCols(); j++)
               basis_generator.takeSample(wgted_snapshots.GetColumn(j));
            basis_generator.endSamples();

            const CAROM::Matrix *carom_basis = basis_generator.getSpatialBasis();
            CAROM::CopyMatrix(*carom_basis, basis);
            for (int i = ufes->GetTrueVSize(); i < basis.NumRows(); i++)
               for (int j = 0; j < basis.NumCols(); j++)
                  basis(i,j) /= sqrt(pres_wgt);

            {  // save basis in a hdf5 format.
               hid_t file_id;
               herr_t errf = 0;
               file_id = H5Fcreate(basis_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
               assert(file_id >= 0);

               hdf5_utils::WriteDataset(file_id, "basis", basis);

               errf = H5Fclose(file_id);
               assert(errf >= 0);
            }

            if (num_basis != basis.NumCols())
            {
               assert(num_basis < basis.NumCols());
               basis.SetSize(basis.NumRows(), num_basis);
            }
         }  // if !(FileExists(basis_file))
      }  // if (wgt_basis)
      else
      {
         CAROM::BasisReader basis_reader(filename);
         const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis);
         CAROM::CopyMatrix(*carom_basis, basis);
      }

      if (rom_mode == RomMode::TENSOR)
      {
         DenseTensor *nlin_rom = oper.GetReducedTensor(basis, basis);

         {  // save the sample to a hdf5 file.
            std::string tensor_file = filename + "_tensor.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fcreate(tensor_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::WriteDataset(file_id, "tensor", *nlin_rom);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         delete nlin_rom;
      }  // if (rom_mode == RomMode::TENSOR)
      else if ((rom_mode == RomMode::TENSOR2) || (rom_mode == RomMode::TENSOR3))
      {
         DenseMatrix ubasis;
         CAROM::BasisReader ubasis_reader(filename + "_vel");
         const CAROM::Matrix *carom_ubasis = ubasis_reader.getSpatialBasis(num_basis);
         CAROM::CopyMatrix(*carom_ubasis, ubasis);

         DenseTensor *nlin_rom = oper.GetReducedTensor(ubasis, ubasis);
         {  // save the sample to a hdf5 file.
            std::string tensor_file = filename + "_tensor.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fcreate(tensor_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::WriteDataset(file_id, "tensor", *nlin_rom);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

      }  // if ((rom_mode == RomMode::TENSOR2) || (rom_mode == RomMode::TENSOR3))
      else if ((rom_mode == RomMode::TENSOR4) || (rom_mode == RomMode::TENSOR5))
      {
         assert(num_psupreme >= 0);

         DenseMatrix ubasis, pbasis;
         CAROM::BasisReader ubasis_reader(filename + "_vel");
         CAROM::BasisReader pbasis_reader(filename + "_pres");
         const CAROM::Matrix *carom_ubasis = ubasis_reader.getSpatialBasis(num_basis);
         const CAROM::Matrix *carom_pbasis = pbasis_reader.getSpatialBasis(num_pbasis);
         CAROM::CopyMatrix(*carom_ubasis, ubasis);
         CAROM::CopyMatrix(*carom_pbasis, pbasis);

         basis.SetSize(ubasis.NumRows(), num_basis + num_psupreme);

         // Copy the standard u basis first.
         Array<int> uridx(ubasis.NumRows()), ucidx(ubasis.NumCols());
         for (int k = 0; k < ubasis.NumRows(); k++) uridx[k] = k;
         for (int k = 0; k < ubasis.NumCols(); k++) ucidx[k] = k;
         basis = 0.0;
         basis.SetSubMatrix(uridx, ucidx, ubasis);

         // Supremizer operator. Not factorizing with mass matrix yet.
         MixedBilinearFormDGExtension S(ufes, pfes);
         ConstantCoefficient minus_one(-1.0);
         S.AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
         if (use_dg)
            S.AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);
         S.Assemble();
         S.Finalize();

         DenseMatrix supreme(ubasis.NumRows(), num_psupreme);
         Vector pvec, svec;
         for (int i = 0; i < num_psupreme; i++)
         {
            pbasis.GetColumnReference(i, pvec);
            supreme.GetColumnReference(i, svec);
            // basis.GetColumnReference(num_basis + i, svec);
            S.MultTranspose(pvec, svec);
         }

         // Copy the supremizer.
         ucidx.SetSize(num_psupreme);
         for (int k = 0; k < num_psupreme; k++) ucidx[k] = num_basis + k;
         basis.SetSubMatrix(uridx, ucidx, supreme);

         // Orthonormalize over the entire velocity basis for now.
         modifiedGramSchmidt(basis);

         DenseTensor *nlin_rom = NULL;
         if (rom_mode == RomMode::TENSOR4)
            nlin_rom = oper.GetReducedTensor(basis, basis);
         else  // if (rom_mode == RomMode::TENSOR5)
            nlin_rom = oper.GetReducedTensor(basis, ubasis);

         {  // save the supremizer-enriched vel basis to a hdf5 file.
            std::string tensor_file = filename + "_velsup.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fcreate(tensor_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::WriteDataset(file_id, "basis", basis);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }
         {  // save the tensor to a hdf5 file.
            std::string tensor_file = filename + "_tensor.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fcreate(tensor_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::WriteDataset(file_id, "tensor", *nlin_rom);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         delete nlin_rom;
      }  // if ((rom_mode == RomMode::TENSOR4) || (rom_mode == RomMode::TENSOR5))
      else if (rom_mode == RomMode::EQP)
      {
         const int fom_vdofs = oper.Height();
         CAROM::Options option(fom_vdofs, nsample, 1, false);
         CAROM::BasisGenerator snapshot_generator(option, false, filename);
         // std::string snapshot_filename = filename + "_snapshot";
         snapshot_generator.loadSamples(filename + "_snapshot", "snapshot");
         // TODO: what happen if we do not deep-copy?
         const CAROM::Matrix *snapshots = snapshot_generator.getSnapshotMatrix();

         DenseMatrix u_basis;
         u_basis.CopyRows(basis, 0, ufes->GetTrueVSize() - 1); // indexes are inclusive.

         const IntegrationRule *ir = oper.GetNonlinearIntRule();
         auto nl_integ = new VectorConvectionTrilinearFormIntegrator(*(oper.GetZeta()));
         nl_integ->SetIntRule(ir);

         ROMNonlinearForm *rom_nlinf = new ROMNonlinearForm(num_basis, ufes);
         rom_nlinf->AddDomainIntegrator(nl_integ);
         rom_nlinf->SetBasis(u_basis);

         rom_nlinf->TrainEQP(*snapshots, eqp_tol);
         Array<int> sample_el, sample_qp;
         Array<double> sample_qw;
         rom_nlinf->GetEQPForIntegrator(IntegratorType::DOMAIN, 0, sample_el, sample_qp, sample_qw);

         {  // save empirical quadrature point locations.
            ElementTransformation *T;
            Vector loc(dim);
            DenseMatrix qp_loc(sample_el.Size(), dim);
            for (int p = 0; p < sample_el.Size(); p++)
            {
               T = ufes->GetElementTransformation(sample_el[p]);
               const IntegrationPoint &ip = ir->IntPoint(sample_qp[p]);
               T->Transform(ip, loc);

               for (int d = 0; d < dim; d++)
                  qp_loc(p, d) = loc(d);
            }
            PrintMatrix(qp_loc, filename + "_eqp_loc.txt");
         }

         {  // save the sample to a hdf5 file.
            std::string sample_file = filename + "_sample.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fcreate(sample_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert(file_id >= 0);

            rom_nlinf->SaveEQPForIntegrator(IntegratorType::DOMAIN, 0, file_id, "sample");

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }
      }  // if (rom_mode == RomMode::EQP)
   }  // if (mode == Mode::BUILD)

   if (mode == Mode::COMPARE)
   {
      if (random_sample)
      {
         for (int d = 0; d < problem::u0.Size(); d++)
         {
            problem::u0(d) = 2.0 * UniformRandom() - 1.0;
            problem::du(d) = 0.1 * (2.0 * UniformRandom() - 1.0);
            problem::offsets(d) = UniformRandom();

            for (int d2 = 0; d2 < problem::u0.Size(); d2++)
               problem::k(d, d2) = 0.5 * (2.0 * UniformRandom() - 1.0);
         }
      }
      else
      {
         problem::du = 0.0;
         problem::offsets = 0.0;
         problem::k = 0.0;

         problem::u0(0) = -1.0;
         problem::u0(1) = 1.0;
         printf("u0: (%f, %f)\n", problem::u0(0), problem::u0(1));
      }

      oper.SetupProblem();

      DenseMatrix basis, u_basis, ubasis, pbasis, basisM;
      DenseMatrix *proj_basis = NULL;
      if (!(rom_mode == RomMode::TENSOR5))
         proj_basis = &basis;

      if (wgt_basis)
      {
         {  // load basis from a hdf5 format.
            std::string basis_file = filename + "_basis.h5";
            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(basis_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "basis", basis);

            errf = H5Fclose(file_id);
            assert(errf >= 0);

            if (num_basis != basis.NumCols())
            {
               assert(num_basis < basis.NumCols());
               basis.SetSize(basis.NumRows(), num_basis);
            }
         }

         printf("weight on pressure: %.3E\n", pres_wgt);
         Vector wgt(oper.Height());
         wgt = 1.0;
         for (int k = ufes->GetTrueVSize(); k < wgt.Size(); k++) wgt(k) = pres_wgt;
         basisM = basis;
         basisM.RightScaling(wgt);
      }  // if (wgt_basis)
      else if (rom_mode == RomMode::TENSOR2)
      {
         CAROM::BasisReader ubasis_reader(filename + "_vel");
         CAROM::BasisReader pbasis_reader(filename + "_pres");
         const CAROM::Matrix *carom_ubasis = ubasis_reader.getSpatialBasis(num_basis);
         const CAROM::Matrix *carom_pbasis = pbasis_reader.getSpatialBasis(num_pbasis);
         CAROM::CopyMatrix(*carom_ubasis, ubasis);
         CAROM::CopyMatrix(*carom_pbasis, pbasis);

         basis.SetSize(ubasis.NumRows() + pbasis.NumRows(), ubasis.NumCols() + pbasis.NumCols());
         Array<int> uridx(ubasis.NumRows()), ucidx(ubasis.NumCols()), pridx(pbasis.NumRows()), pcidx(pbasis.NumCols());
         for (int k = 0; k < ubasis.NumRows(); k++) uridx[k] = k;
         for (int k = 0; k < ubasis.NumCols(); k++) ucidx[k] = k;
         for (int k = 0; k < pbasis.NumRows(); k++) pridx[k] = k + ubasis.NumRows();
         for (int k = 0; k < pbasis.NumCols(); k++) pcidx[k] = k + ubasis.NumCols();
         basis = 0.0;
         basis.SetSubMatrix(uridx, ucidx, ubasis);
         basis.SetSubMatrix(pridx, pcidx, pbasis);
      }  // if (rom_mode == RomMode::TENSOR2)
      else if (rom_mode == RomMode::TENSOR3)
      {
         CAROM::BasisReader ubasis_reader(filename + "_vel");
         const CAROM::Matrix *carom_ubasis = ubasis_reader.getSpatialBasis(num_basis);
         CAROM::CopyMatrix(*carom_ubasis, ubasis);

         basis.SetSize(ubasis.NumRows() + pdim, ubasis.NumCols());
         Array<int> uridx(ubasis.NumRows()), ucidx(ubasis.NumCols());
         for (int k = 0; k < ubasis.NumRows(); k++) uridx[k] = k;
         for (int k = 0; k < ubasis.NumCols(); k++) ucidx[k] = k;
         basis = 0.0;
         basis.SetSubMatrix(uridx, ucidx, ubasis);
      }  // if (rom_mode == RomMode::TENSOR3)
      else if (rom_mode == RomMode::TENSOR4)
      {
         assert(num_psupreme >= 0);

         {  // load the supremizer-enriched velocity basis from a hdf5 file.
            std::string tensor_file = filename + "_velsup.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(tensor_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "basis", ubasis);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         CAROM::BasisReader pbasis_reader(filename + "_pres");
         const CAROM::Matrix *carom_pbasis = pbasis_reader.getSpatialBasis(num_pbasis);
         CAROM::CopyMatrix(*carom_pbasis, pbasis);

         basis.SetSize(ubasis.NumRows() + pbasis.NumRows(), ubasis.NumCols() + pbasis.NumCols());
         Array<int> uridx(ubasis.NumRows()), ucidx(ubasis.NumCols()), pridx(pbasis.NumRows()), pcidx(pbasis.NumCols());
         for (int k = 0; k < ubasis.NumRows(); k++) uridx[k] = k;
         for (int k = 0; k < ubasis.NumCols(); k++) ucidx[k] = k;
         for (int k = 0; k < pbasis.NumRows(); k++) pridx[k] = k + ubasis.NumRows();
         for (int k = 0; k < pbasis.NumCols(); k++) pcidx[k] = k + ubasis.NumCols();
         basis = 0.0;
         basis.SetSubMatrix(uridx, ucidx, ubasis);
         basis.SetSubMatrix(pridx, pcidx, pbasis);
      }  // if (rom_mode == RomMode::TENSOR4)
      else if (rom_mode == RomMode::TENSOR5)
      {
         CAROM::BasisReader ubasis_reader(filename + "_vel");
         CAROM::BasisReader pbasis_reader(filename + "_pres");
         const CAROM::Matrix *carom_ubasis = ubasis_reader.getSpatialBasis(num_basis);
         const CAROM::Matrix *carom_pbasis = pbasis_reader.getSpatialBasis(num_pbasis);
         CAROM::CopyMatrix(*carom_ubasis, ubasis);
         CAROM::CopyMatrix(*carom_pbasis, pbasis);

         basis.SetSize(ubasis.NumRows() + pbasis.NumRows(), ubasis.NumCols() + pbasis.NumCols());
         Array<int> uridx(ubasis.NumRows()), ucidx(ubasis.NumCols()), pridx(pbasis.NumRows()), pcidx(pbasis.NumCols());
         for (int k = 0; k < ubasis.NumRows(); k++) uridx[k] = k;
         for (int k = 0; k < ubasis.NumCols(); k++) ucidx[k] = k;
         for (int k = 0; k < pbasis.NumRows(); k++) pridx[k] = k + ubasis.NumRows();
         for (int k = 0; k < pbasis.NumCols(); k++) pcidx[k] = k + ubasis.NumCols();
         basis = 0.0;
         basis.SetSubMatrix(uridx, ucidx, ubasis);
         basis.SetSubMatrix(pridx, pcidx, pbasis);

         {  // load the supremizer-enriched velocity basis from a hdf5 file.
            std::string tensor_file = filename + "_velsup.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(tensor_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "basis", ubasis);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         proj_basis = new DenseMatrix(ubasis.NumRows() + pbasis.NumRows(), num_basis + num_pbasis);
         (*proj_basis) = 0.0;

         ucidx.SetSize(num_basis + num_pbasis);
         for (int k = 0; k < ucidx.Size(); k++) ucidx[k] = k;
         proj_basis->SetSubMatrix(uridx, ucidx, ubasis);
         proj_basis->SetSubMatrix(pridx, pcidx, pbasis);

         Vector scale(num_basis + num_pbasis);
         scale = 1.0;
         for (int k = num_basis; k < scale.Size(); k++) scale(k) /= sqrt(2.0);

         proj_basis->RightScaling(scale);
         
      }  // if (rom_mode == RomMode::TENSOR5)
      else
      {
         CAROM::BasisReader basis_reader(filename);
         const CAROM::Matrix *carom_basis = basis_reader.getSpatialBasis(num_basis);
         CAROM::CopyMatrix(*carom_basis, basis);
      }

      SparseMatrix *linear_term = oper.GetLinearTerm();
      DenseMatrix lin_rom;
      if (wgt_basis)
         mfem::RtAP(basisM, *linear_term, basis, lin_rom);
      else
         mfem::RtAP(*proj_basis, *linear_term, basis, lin_rom);

      PrintMatrix(*linear_term, filename+"_linop.txt");

      u_basis.CopyRows(basis, 0, ufes->GetTrueVSize() - 1); // indexes are inclusive.

      DenseTensor *nlin_rom = NULL;
      ROMNonlinearForm *rom_nlinf = NULL;
      Operator *rom_oper = NULL;
      TensorROM *tenrom = NULL;
      EQPROM *eqprom = NULL;
      SteadyNSEQPROM *eqprom1 = NULL;
      SparseMatrix lin_rom1(lin_rom.NumRows(), lin_rom.NumCols());
      Array<int> idx(lin_rom.NumRows());
      for (int k = 0; k < idx.Size(); k++) idx[k] = k;
      lin_rom1.AddSubMatrix(idx, idx, lin_rom);
      lin_rom1.Finalize();
      Array<ROMNonlinearForm *> eqp_opers(1);
      eqp_opers = NULL;
      Array<int> block_offsets(2);
      block_offsets[0] = 0;
      block_offsets[1] = lin_rom.NumRows();

      if ((rom_mode == RomMode::TENSOR) || (rom_mode == RomMode::TENSOR3))
      {
         nlin_rom = new DenseTensor;
         {  // load the sample from a hdf5 file.
            std::string tensor_file = filename + "_tensor.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(tensor_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "tensor", *nlin_rom);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         tenrom = new TensorROM(lin_rom, *nlin_rom);
         rom_oper = tenrom;
      }  // if ((rom_mode == RomMode::TENSOR) || (rom_mode == RomMode::TENSOR3))
      else if ((rom_mode == RomMode::TENSOR2) || (rom_mode == RomMode::TENSOR4) || (rom_mode == RomMode::TENSOR5))
      {
         nlin_rom = new DenseTensor;
         {  // load the sample from a hdf5 file.
            std::string tensor_file = filename + "_tensor.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(tensor_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            hdf5_utils::ReadDataset(file_id, "tensor", *nlin_rom);

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         tenrom = new TensorROM2(lin_rom, *nlin_rom);
         rom_oper = tenrom;
      }  // if (rom_mode == RomMode::TENSOR2)
      else if (rom_mode == RomMode::EQP)
      {
         const IntegrationRule *ir = oper.GetNonlinearIntRule();
         auto nl_integ = new VectorConvectionTrilinearFormIntegrator(*(oper.GetZeta()));
         nl_integ->SetIntRule(ir);

         rom_nlinf = new ROMNonlinearForm(num_basis, ufes);
         rom_nlinf->AddDomainIntegrator(nl_integ);
         rom_nlinf->SetBasis(u_basis);
         rom_nlinf->SetPrecomputeMode(precompute);

         {  // load the sample from a hdf5 file.
            std::string sample_file = filename + "_sample.h5";

            hid_t file_id;
            herr_t errf = 0;
            file_id = H5Fopen(sample_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            assert(file_id >= 0);

            rom_nlinf->LoadEQPForIntegrator(IntegratorType::DOMAIN, 0, file_id, "sample");

            errf = H5Fclose(file_id);
            assert(errf >= 0);
         }

         if (precompute)
            rom_nlinf->PrecomputeCoefficients();

         eqprom = new EQPROM(lin_rom, *rom_nlinf);
         // eqp_opers[0] = rom_nlinf;
         // eqprom1 = new SteadyNSEQPROM(&lin_rom1, eqp_opers, block_offsets);
         rom_oper = eqprom;
         // rom_oper = eqprom1;
      }  // if (rom_mode == RomMode::EQP)
      else
         mfem_error("ROM Mode is not set!");
      
      Vector rom_rhs(basis.NumCols()), rom_sol(basis.NumCols());
      proj_basis->MultTranspose((*oper.GetRHS()), rom_rhs);
      rom_sol = 0.0;

      double atol=1.0e-10, rtol=1.0e-10;
      int maxIter=10000;

      GMRESSolver J_gmres;
      J_gmres.SetAbsTol(atol);
      J_gmres.SetRelTol(rtol);
      J_gmres.SetMaxIter(maxIter);
      J_gmres.SetPrintLevel(-1);

      MUMPSSolver J_mumps(MPI_COMM_WORLD);
      J_mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);

      Solver *J_solver;
      if (direct_solve)
         J_solver = &J_mumps;
      else
         J_solver = &J_gmres;

      NewtonSolver newton_solver;
      newton_solver.SetSolver(*J_solver);
      newton_solver.SetOperator(*rom_oper);
      newton_solver.SetPrintLevel(1); // print Newton iterations
      newton_solver.SetRelTol(rtol);
      newton_solver.SetAbsTol(atol);
      newton_solver.SetMaxIter(100);

      StopWatch solveTimer;
      solveTimer.Start();
      newton_solver.Mult(rom_rhs, rom_sol);
      solveTimer.Stop();
      // printf("ROM mult: %.3E sec\n", rom_oper.multTimer.RealTime());
      // printf("ROM jac: %.3E sec\n", rom_oper.jacTimer.RealTime());
      printf("ROM solve: %.3E sec\n", solveTimer.RealTime());

      double rom_mult, rom_jac;
      switch (rom_mode)
      {
         case RomMode::TENSOR:
         case RomMode::TENSOR2:
         case RomMode::TENSOR3:
         case RomMode::TENSOR4:
         {
            rom_mult = tenrom->GetAvgMultTime();
            rom_jac = tenrom->GetAvgGradTime();
         }
         break;
         case RomMode::EQP:
         {
            rom_mult = eqprom->GetAvgMultTime();
            rom_jac = eqprom->GetAvgGradTime();
         }
         break;
      }

      Vector sol(basis.NumRows());
      basis.Mult(rom_sol, sol);
      GridFunction *u_rom = new GridFunction;
      GridFunction *p_rom = new GridFunction;
      u_rom->MakeRef(ufes, sol, 0);
      p_rom->MakeRef(pfes, sol, ufes->GetVSize());

      // FOM solve.
      oper.Solve();
      const double fom_mult = oper.GetAvgMultTime();
      const double fom_jac = oper.GetAvgGradTime();
      const double fom_solve = oper.GetSolveTime();

      // 12. Create the grid functions u and p. Compute the L2 error norms.
      GridFunction *u = oper.GetVelocityGridFunction();
      GridFunction *p = oper.GetPressureGridFunction();

      GridFunction *u_error = new GridFunction(ufes);
      GridFunction *p_error = new GridFunction(pfes);
      *u_error = *u_rom;
      *u_error -= *u;
      *p_error = *p_rom;
      *p_error -= *p;

      // set a pressure constant just for relative error.
      (*p) += 1.0;
      (*p_rom) += 1.0;
      VectorGridFunctionCoefficient u_coeff(u), p_coeff(p);

      int order_quad = max(2, 2*(order+1)+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double err_u  = u_rom->ComputeL2Error(u_coeff, irs);
      double norm_u = ComputeLpNorm(2., u_coeff, *mesh, irs);
      double err_p  = p_rom->ComputeL2Error(p_coeff, irs);
      double norm_p = ComputeLpNorm(2., p_coeff, *mesh, irs);

      printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
      printf("|| p_h - p_ex || / || p_ex || = %.5E\n", err_p / norm_p);

      if ( (err_p / norm_p) > 1.0e-1 )
      {
         printf("Pressure relative error > 0.1\n");
         printf("u0: ");
         for (int d = 0; d < problem::u0.Size(); d++)
            printf("%.3E\t", problem::u0(d));
         printf("\n");
         printf("du: ");
         for (int d = 0; d < problem::u0.Size(); d++)
            printf("%.3E\t", problem::du(d));
         printf("\n");
         printf("offsets: ");
         for (int d = 0; d < problem::u0.Size(); d++)
            printf("%.3E\t", problem::offsets(d));
         printf("\n");
         printf("k:\n");
         for (int d = 0; d < problem::u0.Size(); d++)
         {
            for (int d2 = 0; d2 < problem::u0.Size(); d2++)
               printf("%.3E\t", problem::k(d, d2));
            printf("\n");
         }
      }

      // 15. Save data in the ParaView format
      ParaViewDataCollection paraview_dc("ns_paraview", mesh);
      paraview_dc.SetLevelsOfDetail(order+1);
      paraview_dc.RegisterField("velocity", u);
      paraview_dc.RegisterField("pressure", p);
      paraview_dc.RegisterField("vel_rom", u_rom);
      paraview_dc.RegisterField("pres_rom", p_rom);
      paraview_dc.RegisterField("vel_error", u_error);
      paraview_dc.RegisterField("pres_error", p_error);
      Array<GridFunction *> basisgf_u(num_basis), basisgf_p(num_basis);
      basisgf_u = NULL; basisgf_p = NULL;
      for (int k = 0; k < num_basis; k++)
      {
         basisgf_u[k] = new GridFunction(ufes);
         basisgf_u[k]->MakeRef(ufes, basis.GetColumn(k));
         std::string ustr = "u_basis" + std::to_string(k);
         paraview_dc.RegisterField(ustr.c_str(), basisgf_u[k]);

         if (rom_mode == RomMode::TENSOR)
         {
            basisgf_p[k] = new GridFunction(pfes);
            basisgf_p[k]->MakeRef(pfes, basis.GetColumn(k) + ufes->GetVSize());
            std::string pstr = "p_basis" + std::to_string(k);
            paraview_dc.RegisterField(pstr.c_str(), basisgf_p[k]);
         }
      }
      if ((rom_mode == RomMode::TENSOR2) || (rom_mode == RomMode::TENSOR4) || (rom_mode == RomMode::TENSOR5))
      {
         basisgf_p.SetSize(num_pbasis);
         for (int k = 0; k < num_pbasis; k++)
         {
            basisgf_p[k] = new GridFunction(pfes);
            basisgf_p[k]->MakeRef(pfes, pbasis.GetColumn(k));
            std::string pstr = "p_basis" + std::to_string(k);
            paraview_dc.RegisterField(pstr.c_str(), basisgf_p[k]);
         }
      }
      if (rom_mode == RomMode::TENSOR4)
      {
         for (int k = 0; k < num_psupreme; k++)
         {
            basisgf_u.Append(new GridFunction(ufes));
            basisgf_u.Last()->MakeRef(ufes, basis.GetColumn(num_basis + k));
            std::string ustr = "u_supreme" + std::to_string(k);
            paraview_dc.RegisterField(ustr.c_str(), basisgf_u.Last());
         }
      }
      else if (rom_mode == RomMode::TENSOR5)
      {
         for (int k = 0; k < num_psupreme; k++)
         {
            basisgf_u.Append(new GridFunction(ufes));
            basisgf_u.Last()->MakeRef(ufes, proj_basis->GetColumn(num_basis + k));
            std::string ustr = "u_supreme" + std::to_string(k);
            paraview_dc.RegisterField(ustr.c_str(), basisgf_u.Last());
         }
      }

      paraview_dc.Save();

      {  // save the comparison result to a hdf5 file.
         hid_t file_id;
         herr_t errf = 0;
         file_id = H5Fcreate(compare_output_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
         assert(file_id >= 0);

         Array<double> params;
         for (int d = 0; d < problem::u0.Size(); d++) params.Append(problem::u0(d));
         for (int d = 0; d < problem::u0.Size(); d++) params.Append(problem::du(d));
         for (int d = 0; d < problem::u0.Size(); d++) params.Append(problem::offsets(d));
         for (int d = 0; d < problem::u0.Size(); d++)
            for (int d2 = 0; d2 < problem::u0.Size(); d2++)
               params.Append(problem::k(d, d2));

         hdf5_utils::WriteAttribute(file_id, "fom/mult", fom_mult);
         hdf5_utils::WriteAttribute(file_id, "fom/grad", fom_jac);
         hdf5_utils::WriteAttribute(file_id, "fom/solve", fom_solve);
         hdf5_utils::WriteAttribute(file_id, "rom/mult", rom_mult);
         hdf5_utils::WriteAttribute(file_id, "rom/grad", rom_jac);
         hdf5_utils::WriteAttribute(file_id, "rom/solve", solveTimer.RealTime());
         hdf5_utils::WriteAttribute(file_id, "rel_error/u", err_u / norm_u);
         hdf5_utils::WriteAttribute(file_id, "rel_error/p", err_p / norm_p);
         hdf5_utils::WriteDataset(file_id, "params", params);

         errf = H5Fclose(file_id);
         assert(errf >= 0);
      }

      for (int k = 0; k < basisgf_u.Size(); k++)
      {
         delete basisgf_u[k];
      }
      for (int k = 0; k < basisgf_p.Size(); k++)
      {
         delete basisgf_p[k];
      }
      delete linear_term;
      delete u_rom;
      delete p_rom;
      delete u_error;
      delete p_error;
      delete nlin_rom;
      delete rom_oper;
      delete rom_nlinf;
      if (rom_mode == RomMode::TENSOR5)
         delete proj_basis;
   }  // if (mode == Mode::COMPARE)
   
   // 17. Free the used memory.
   delete mesh;

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * nu * sin(xi)*sin(yi);
}

void fFun(const Vector & x, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   // f(0) = exp(xi)*sin(yi);
   // f(1) = exp(xi)*cos(yi);
   f(0) = 4.0 * nu * cos(xi) * sin(yi);
   f(1) = 0.0;

   f(0) += - zeta * sin(xi) * cos(xi);
   f(1) += - zeta * sin(yi) * cos(yi);
}

double gFun(const Vector & x)
{
   assert(x.Size() == 2);
   return 0;
}

double f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}

void dudx_ex(const Vector & x, Vector & y)
{
   assert(x.Size() == 2);
   y.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   y(0) = - nu * sin(xi)*sin(yi);
   y(1) = - nu * cos(xi)*cos(yi);
}

void uux_ex(const Vector & x, Vector & y)
{
   assert(x.Size() == 2);
   y.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   uFun_ex(x, y);
   y(1) *= - y(0);
   y(0) *= - y(0);
}
