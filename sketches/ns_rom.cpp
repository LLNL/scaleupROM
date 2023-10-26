//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/star.mesh -pa
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/beam-hex.mesh -pa
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex5 -m ../data/star.mesh -pa -d cuda
//               ex5 -m ../data/star.mesh -pa -d raja-cuda
//               ex5 -m ../data/star.mesh -pa -d raja-omp
//               ex5 -m ../data/beam-hex.mesh -pa -d cuda
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
//                                 - div u      = g
//
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockOperator class, as
//               well as the collective saving of several grid functions in
//               VisIt (visit.llnl.gov) and ParaView (paraview.org) formats.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "etc.hpp"
#include "linalg_utils.hpp"
#include "nonlinear_integ.hpp"
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

static double nu = 0.1;
static double zeta = 1.0;

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

   mutable BlockMatrix *system_jac;
   mutable SparseMatrix *mono_jac, *uu, *up, *pu;

   double atol=1.0e-10, rtol=1.0e-10;
   int maxIter=10000;
   GMRESSolver *J_solver;
   NewtonSolver *newton_solver;
   
   // mutable BlockDiagonalPreconditioner *jac_prec;
   // BilinearForm *pMass = NULL;
   // SparseMatrix *pM = NULL;
   // GSSmoother *p_prec = NULL;
   // OrthoSolver *ortho_p_prec = NULL;

   // HYPRE_BigInt glob_size;
   // mutable HYPRE_BigInt row_starts[2];
   // mutable HypreParMatrix *uu_hypre = NULL;
   // mutable HypreBoomerAMG *u_prec = NULL;

public:
   SteadyNavierStokes(Mesh *mesh_, const int order_, const double nu_ = 1.0, const double zeta_ = 1.0,
                      const bool use_dg_ = false, const bool pres_dbc_ = false)
      : mesh(mesh_), dim(mesh_->Dimension()), order(order_), sigma(-1.0), kappa((order_+2)*(order_+2)),
        nu(nu_), zeta(zeta_), use_dg(use_dg_), pres_dbc(pres_dbc_),
        zero(0.0), one(1.0), minus_one(-1.0), half(0.), minus_half(-0.5),
        system_jac(NULL), mono_jac(NULL), uu(NULL), up(NULL), pu(NULL)
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

      H->AddDomainIntegrator(new VectorConvectionNLFIntegrator(zeta));
      // if (use_dg)
      //    // nVarf->AddInteriorFaceIntegrator(new DGLaxFriedrichsFluxIntegrator(one));
      //    nVarf->AddInteriorFaceIntegrator(new DGTemamFluxIntegrator(half_zeta));

      J_solver = new GMRESSolver;
      J_solver->SetAbsTol(atol);
      J_solver->SetRelTol(rtol);
      J_solver->SetMaxIter(maxIter);
      J_solver->SetPrintLevel(-1);

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
      Vector x_u(x.GetData()+vblock_offsets[0], M->Height()), x_p(x.GetData()+vblock_offsets[1], S->Height());
      Vector y_u(y.GetData(), M->Height()), y_p(y.GetData()+vblock_offsets[1], S->Height());

      H->Mult(x_u, y_u);
      M->AddMult(x_u, y_u);
      S->AddMultTranspose(x_p, y_u);
      S->Mult(x_u, y_p);
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const
   {
      delete system_jac, mono_jac, uu;
      const Vector x_u(x.GetData()+vblock_offsets[0], M->Height()), x_p(x.GetData()+vblock_offsets[1], S->Height());

      SparseMatrix *grad_H = dynamic_cast<SparseMatrix *>(&H->GetGradient(x_u));
      uu = Add(1.0, M->SpMat(), 1.0, *grad_H);

      assert(up && pu);

      system_jac = new BlockMatrix(vblock_offsets);
      system_jac->SetBlock(0,0, uu);
      system_jac->SetBlock(0,1, up);
      system_jac->SetBlock(1,0, pu);

      // // update preconditioner.
      // delete u_prec, uu_hypre;
      // uu_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, uu);
      // u_prec = new HypreBoomerAMG(*uu_hypre);
      // u_prec->SetPrintLevel(0);
      // u_prec->SetSystemsOptions(dim, true);
      // jac_prec->SetDiagonalBlock(0, u_prec);

      mono_jac = system_jac->CreateMonolithic();
      return *mono_jac;
   }

   virtual ~SteadyNavierStokes()
   {
      delete system_jac, mono_jac, uu, up;
      delete M, S, H, F, G;
      delete x, rhs;
      delete u, p;
      delete ufec, pfec, ufes, pfes;
      delete J_solver, newton_solver;
      // delete jac_prec;
      // delete pMass, p_prec, ortho_p_prec;
   }

   // BlockDiagonalPreconditioner* GetGradientPreconditioner() { return jac_prec; }
   virtual void Solve()
   {
      newton_solver->Mult(*rhs, *x);
      if (!pres_dbc)
      {
         (*p) -= p->Sum() / p->Size();
      }
   }

   BlockVector* GetSolution() { return x; }
   GridFunction* GetVelocityGridFunction() { return u; }
   GridFunction* GetPressureGridFunction() { return p; }
   FiniteElementSpace* GetUfes() { return ufes; }
   FiniteElementSpace* GetPfes() { return pfes; }
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
   bool mms = false;
   bool sample = false;
   int nsample = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.AddOption(&pres_dbc, "-pd", "--pressure-dirichlet",
                  "-no-pd", "--no-pressure-dirichlet",
                  "Use pressure Dirichlet condition.");
   args.AddOption(&use_dg, "-dg", "--use-dg", "-no-dg", "--no-use-dg",
                  "Use discontinuous Galerkin scheme.");
   args.AddOption(&mms, "-mms", "--mms", "-no-mms", "--no-mms",
                  "Solve for manufactured solution.");
   args.AddOption(&sample, "-sample", "--sample", "-no-sample", "--no-sample",
                  "Generate samples. specify -ns.");
   args.AddOption(&nsample, "-ns", "--nsample",
                  "Number of samples to be generated.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   assert(!pres_dbc);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   if (mms)
   {
      SteadyNavierStokes oper(mesh, order, nu, zeta, use_dg, pres_dbc);

      VectorFunctionCoefficient fcoeff(dim, fFun);
      FunctionCoefficient gcoeff(gFun);

      VectorFunctionCoefficient ucoeff(dim, uFun_ex);
      FunctionCoefficient pcoeff(pFun_ex);

      FiniteElementSpace *ufes = oper.GetUfes();
      FiniteElementSpace *pfes = oper.GetPfes();
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
      ParaViewDataCollection paraview_dc("stokes_mms_paraview", mesh);
      // paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(order);
      // paraview_dc.SetCycle(0);
   //    paraview_dc.SetDataFormat(VTKFormat::BINARY);
   //    paraview_dc.SetHighOrderOutput(true);
   //    paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("velocity", u);
      paraview_dc.RegisterField("pressure", p);
      paraview_dc.Save();
   }  // mms mode
   else if (sample)
   {
      assert(nsample > 0);

      problem::u0.SetSize(mesh->Dimension());
      problem::du.SetSize(mesh->Dimension());
      problem::offsets.SetSize(mesh->Dimension());
      problem::k.SetSize(mesh->Dimension(), mesh->Dimension());

      SteadyNavierStokes *temp = new SteadyNavierStokes(mesh, order, nu, zeta, use_dg, pres_dbc);
      const int fom_vdofs = temp->Height();
      delete temp;
      std::string filename = "ns_rom";
      CAROM::Options option(fom_vdofs, nsample, 1, false);
      CAROM::BasisGenerator snapshot_generator(option, false, filename);

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

         SteadyNavierStokes oper(mesh, order, nu, zeta, use_dg, pres_dbc);
         oper.SetupProblem();
         oper.Solve();

         snapshot_generator.takeSample(oper.GetSolution()->GetData(), 0.0, 0.01);

         // GridFunction *u = oper.GetVelocityGridFunction();
         // GridFunction *p = oper.GetPressureGridFunction();

         // // 15. Save data in the ParaView format
         // ParaViewDataCollection paraview_dc("ns_sample_paraview", mesh);
         // paraview_dc.SetLevelsOfDetail(max(3,order+1));
         // paraview_dc.RegisterField("velocity", u);
         // paraview_dc.RegisterField("pressure", p);
         // paraview_dc.Save();
      }

      snapshot_generator.writeSnapshot();
      snapshot_generator.endSamples();

      const CAROM::Vector *rom_sv = snapshot_generator.getSingularValues();
      printf("Singular values: ");
      for (int d = 0; d < rom_sv->dim(); d++)
         printf("%.3E\t", rom_sv->item(d));
      printf("\n");
   }
   
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