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
#include "interfaceinteg.hpp"
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

static double nu = 0.1;
static double zeta = 1.0;
static bool direct_solve = true;

enum class Scheme
{
   DOMAIN,
   LF,
   TEMAM
};

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);
void dudx_ex(const Vector & x, Vector & y);
void uux_ex(const Vector & x, Vector & y);
void fvec_natural(const Vector & x, Vector & y);

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
private:
   int dim = -1;

   BilinearForm *M;
   MixedBilinearForm *S;
   NonlinearForm *H;
   Array<int> block_offsets;

   mutable BlockMatrix *system_jac;
   mutable SparseMatrix *mono_jac, *uu, *up, *pu;

   // double atol=1.0e-10, rtol=1.0e-10;
   // int maxIter=10000;
   // MINRESSolver *J_solver;
   bool pres_dbc = false;

   HYPRE_BigInt glob_size;
   mutable HYPRE_BigInt row_starts[2];
   mutable HypreParMatrix *jac_hypre = NULL;

public:
   SteadyNavierStokes(BilinearForm *M_, MixedBilinearForm *S_, NonlinearForm *H_, bool pres_dbc_=false)
      : Operator(M_->Height() + S_->Height()), dim(M_->FESpace()->GetVDim()), M(M_), S(S_), H(H_), pres_dbc(pres_dbc_),
        system_jac(NULL), mono_jac(NULL), uu(NULL)
   { 
      block_offsets.SetSize(3);
      block_offsets = 0;
      block_offsets[1] = M_->Height();
      block_offsets[2] = S_->Height();
      block_offsets.PartialSum();

      pu = &(S->SpMat());
      up = Transpose(*pu);
      
      glob_size = block_offsets.Last();
      row_starts[0] = 0;
      row_starts[1] = block_offsets.Last();
   }

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector x_u(x.GetData()+block_offsets[0], M->Height()), x_p(x.GetData()+block_offsets[1], S->Height());
      Vector y_u(y.GetData()+block_offsets[0], M->Height()), y_p(y.GetData()+block_offsets[1], S->Height());
      H->Mult(x_u, y_u);
      M->AddMult(x_u, y_u);
      S->AddMultTranspose(x_p, y_u);
      S->Mult(x_u, y_p);
   }

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &x) const
   {
      delete system_jac;
      delete mono_jac;
      delete uu;
      const Vector x_u(x.GetData()+block_offsets[0], M->Height()), x_p(x.GetData()+block_offsets[1], S->Height());

      SparseMatrix *grad_H = dynamic_cast<SparseMatrix *>(&H->GetGradient(x_u));
      uu = Add(1.0, M->SpMat(), 1.0, *grad_H);

      system_jac = new BlockMatrix(block_offsets);
      system_jac->SetBlock(0,0, uu);
      system_jac->SetBlock(0,1, up);
      system_jac->SetBlock(1,0, pu);

      mono_jac = system_jac->CreateMonolithic();
      if (direct_solve)
      {
         jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, mono_jac);
         return *jac_hypre;
      }
      else
         return *mono_jac;
   }

   virtual ~SteadyNavierStokes()
   {
      delete system_jac;
      delete mono_jac;
      delete uu;
      delete up;
      delete jac_hypre;
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
   const char *scheme_str = "domain";
   Scheme scheme = Scheme::DOMAIN;

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
   args.AddOption(&direct_solve, "-ds", "--direct-solve", "-no-ds", "--no-direct-solve",
                  "Use discontinuous Galerkin scheme.");
   args.AddOption(&zeta, "-z", "--zeta",
                  "Constant coefficient for nonlinear convection.");
   args.AddOption(&scheme_str, "-s", "--scheme",
                  "Discretization scheme for nonlinear convection.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (strcmp(scheme_str, "domain") == 0)
   {
      printf("Domain nonlinear scheme\n");
      scheme = Scheme::DOMAIN;
   }
   else if (strcmp(scheme_str, "lf") == 0)
   {
      printf("Lax-Friedrichs nonlinear scheme\n");
      scheme = Scheme::LF;
   }
   else if (strcmp(scheme_str, "temam") == 0)
   {
      printf("Temam nonlinear scheme\n");
      scheme = Scheme::TEMAM;
   }
   else
      mfem_error("Unknown discretization scheme!\n");

   // assert(!pres_dbc);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   double sigma = -1.0;
   // DG terms are employed for velocity space, which is order+1. resulting kappa becomes (order+2)^2.
   double kappa = (order + 2) * (order + 2);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *dg_coll(new DG_FECollection(order+1, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *pdg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *ph1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes = new FiniteElementSpace(mesh, h1_coll);
   FiniteElementSpace *ufes, *pfes;
   if (use_dg)
   {
      ufes = new FiniteElementSpace(mesh, dg_coll, dim);
      pfes = new FiniteElementSpace(mesh, pdg_coll);
   }
   else
   {
      ufes = new FiniteElementSpace(mesh, h1_coll, dim);
      pfes = new FiniteElementSpace(mesh, ph1_coll);
   }

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(dim+2); // number of variables + 1
   block_offsets[0] = 0;
   for (int d = 1; d <= dim; d++)
      block_offsets[d] = ufes->GetNDofs();
   block_offsets[dim+1] = pfes->GetVSize();
   block_offsets.PartialSum();

   Array<int> vblock_offsets(3); // number of variables + 1
   vblock_offsets[0] = 0;
   vblock_offsets[1] = ufes->GetVSize();
   vblock_offsets[2] = pfes->GetVSize();
   vblock_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   for (int d = 1; d < block_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, block_offsets[d] - block_offsets[d-1]);
   printf("dim(q) = %d\n", block_offsets.Last());
   std::cout << "***********************************************************\n";

   Array<int> u_ess_attr(mesh->bdr_attributes.Max());
   Array<int> p_ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   u_ess_attr = 1;
   p_ess_attr = 0;
   if (pres_dbc)
   {
      u_ess_attr[1] = 0;
      p_ess_attr[1] = 1;
   }

   Array<int> u_ess_tdof, p_ess_tdof, empty;
   // ufes->GetEssentialTrueDofs(u_ess_attr, u_ess_tdof);
   // pfes->GetEssentialTrueDofs(p_ess_attr, p_ess_tdof);

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(nu), zeta_coeff(zeta), half_zeta(0.5 * zeta), minus_zeta(-zeta), minus_half_zeta(-0.5 * zeta);
   ConstantCoefficient minus_one(-1.0), one(1.0), half(0.5), minus_half(-0.5);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   VectorFunctionCoefficient fvecnatcoeff(dim*dim, fvec_natural);
   VectorFunctionCoefficient dudxcoeff(dim, dudx_ex);
   VectorFunctionCoefficient uuxcoeff(dim, uux_ex);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   GridFunction u_ex(ufes), p_ex(pfes);
   u_ex.ProjectCoefficient(ucoeff);
   p_ex.ProjectCoefficient(pcoeff);
   const double p_const = p_ex.Sum() / static_cast<double>(p_ex.Size());

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
//    MemoryType mt = device.GetMemoryType();
   BlockVector x(vblock_offsets), rhs(vblock_offsets);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(ufes, x.GetBlock(0), 0);
   p.MakeRef(pfes, x.GetBlock(1), 0);

   u = 0.0;
   p = 0.0;

   LinearForm *fform(new LinearForm);
   fform->Update(ufes, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   fform->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(ucoeff, k, sigma, kappa), u_ess_attr);

   if (use_dg)
      fform->AddBdrFaceIntegrator(new BoundaryNormalStressLFIntegrator(fvecnatcoeff), p_ess_attr);
   else
      fform->AddBoundaryIntegrator(new BoundaryNormalStressLFIntegrator(fvecnatcoeff), p_ess_attr);

   LinearForm *gform(new LinearForm);
   gform->Update(pfes, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), u_ess_attr);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k \grad u_h \cdot \grad v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(ufes));
   MixedBilinearFormDGExtension *bVarf(new MixedBilinearFormDGExtension(ufes, pfes));
   NonlinearForm *nVarf(new NonlinearForm(ufes));

//    // mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(k));
   if (use_dg)
      mVarf->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(k, sigma, kappa));
   mVarf->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(k, sigma, kappa), u_ess_attr);
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
   if (use_dg)
      bVarf->AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);
   bVarf->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, u_ess_attr);
   bVarf->Assemble();
   bVarf->Finalize();

   /* this integration rule can be used only for domain integrator. */
   IntegrationRule gll_ir_nl = IntRules.Get(ufes->GetFE(0)->GetGeomType(),
                                             (int)(ceil(1.5 * (2 * ufes->GetMaxElementOrder() - 1))));

   auto *domain_integ1 = new VectorConvectionTrilinearFormIntegrator(zeta_coeff);
   domain_integ1->SetIntRule(&gll_ir_nl);

   auto *lf_integ1 = new IncompressibleInviscidFluxNLFIntegrator(minus_zeta);
   auto *lf_integ2 = new DGLaxFriedrichsFluxIntegrator(minus_zeta);
   auto *lf_bdr_integ1 = new DGLaxFriedrichsFluxIntegrator(minus_zeta, &ucoeff);
   lf_integ1->SetIntRule(&gll_ir_nl);

   auto *temam_integ1 = new VectorConvectionTrilinearFormIntegrator(half_zeta);
   auto *temam_integ2 = new IncompressibleInviscidFluxNLFIntegrator(minus_half_zeta);
   auto *temam_integ3 = new DGTemamFluxIntegrator(minus_half_zeta);
   auto *temam_bdr_integ1 = new DGBdrTemamLFIntegrator(ucoeff, &minus_half_zeta);
   temam_integ1->SetIntRule(&gll_ir_nl);
   temam_integ2->SetIntRule(&gll_ir_nl);

   switch (scheme)
   {
      case Scheme::DOMAIN:
      {
         nVarf->AddDomainIntegrator(domain_integ1);
      }
      break;
      case Scheme::LF:
      {
         nVarf->AddDomainIntegrator(lf_integ1);
         if (use_dg)
            nVarf->AddInteriorFaceIntegrator(lf_integ2);
         nVarf->AddBdrFaceIntegrator(lf_bdr_integ1, u_ess_attr);
         /* having linear-only term causes sub-optimal convergence */
         // fform->AddBdrFaceIntegrator(new DGBdrLaxFriedrichsLFIntegrator(ucoeff, &zeta_coeff), u_ess_attr);
      }
      break;
      case Scheme::TEMAM:
      {
         nVarf->AddDomainIntegrator(temam_integ1);
         nVarf->AddDomainIntegrator(temam_integ2);
         if (use_dg)
            nVarf->AddInteriorFaceIntegrator(temam_integ3);

         fform->AddBdrFaceIntegrator(temam_bdr_integ1, u_ess_attr);
      }
      break;
   }

   fform->Assemble();
   fform->SyncAliasMemory(rhs);
   gform->Assemble();
   gform->SyncAliasMemory(rhs);   

   SteadyNavierStokes oper(mVarf, bVarf, nVarf);
{
   int maxIter(10000);
   double rtol(1.e-10);
   double atol(1.e-10);

   Solver *J_solver = NULL;
   GMRESSolver *J_gmres = NULL;
   MUMPSSolver *J_mumps = NULL;
   if (direct_solve)
   {
      J_mumps = new MUMPSSolver(MPI_COMM_WORLD);
      J_mumps->SetPrintLevel(-1);
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

   NewtonSolver newton_solver;
   // newton_solver.iterative_mode = false;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(oper);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rtol);
   newton_solver.SetAbsTol(atol);
   newton_solver.SetMaxIter(100);
   for (int k = 0; k < x.Size(); k++)
      x(k) = UniformRandom();
   // u.ProjectCoefficient(ucoeff);
   // p.ProjectCoefficient(pcoeff);
   newton_solver.Mult(rhs, x);
   
   delete J_mumps;
   delete J_gmres;
}

   if (!pres_dbc)
   {
      p -= p.Sum() / static_cast<double>(p.Size());
      p += p_const;
   }

   int order_quad = max(2, 2*(order+1)+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
   double err_p  = p.ComputeL2Error(pcoeff, irs);
   double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

   printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
   printf("|| p_h - p_ex || / || p_ex || = %.5E\n", err_p / norm_p);

   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("stokes_mms_paraview", mesh);
   // paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);

   // BlockVector res(vblock_offsets);
   // Vector tmp(u_ex.Size() + p_ex.Size());
   // tmp.SetVector(u_ex, 0);
   // tmp.SetVector(p_ex, u_ex.Size());

   // res = 0.0;
   // oper.Mult(tmp, res);

   // // 12. Create the grid functions u and p. Compute the L2 error norms.
   // GridFunction res_u_ex, res_p_ex;
   // res_u_ex.MakeRef(ufes, res.GetBlock(0), 0);
   // res_p_ex.MakeRef(pfes, res.GetBlock(1), 0);
   // paraview_dc.RegisterField("res_u_ex",&res_u_ex);
   // paraview_dc.RegisterField("res_p_ex",&res_p_ex);

   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.RegisterField("u_ex",&u_ex);
   paraview_dc.RegisterField("p_ex",&p_ex);
   paraview_dc.Save();

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete mVarf;
   delete bVarf;
   delete nVarf;
   delete fes;
   delete ufes;
   delete pfes;
   delete h1_coll;
   delete ph1_coll;
   delete l2_coll;
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

void fvec_natural(const Vector & x, Vector & y)
{
   assert(x.Size() == 2);
   y.SetSize(x.Size() * x.Size());

   double xi(x(0));
   double yi(x(1));

   // Grad u = du_i/dx_j - column-major order
   y(0) = - sin(xi)*sin(yi);
   y(1) = - cos(xi)*cos(yi);
   y(2) = cos(xi)*cos(yi);
   y(3) = sin(xi)*sin(yi);

   y *= nu;

   y(0) -= pFun_ex(x);
   y(3) -= pFun_ex(x);
}