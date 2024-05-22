// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "linalg_utils.hpp"
#include "nonlinear_integ.hpp"
#include "hyperreduction_integ.hpp"
#include "interfaceinteg.hpp"
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

static double nu = 0.1;
static double zeta = 1.0;

void uFun_ex(const Vector & x, Vector & u);

class NavierSolver
{
private:

   // int max_bdf_order = 3;
   // int cur_step = 0;
   // std::vector<double> dthist = {0.0, 0.0, 0.0};

   // BDFk/EXTk coefficients.
   /* use first order for now. */
   double bd0 = 1.0;
   double bd1 = -1.0;
   double bd2 = 0.0;
   double bd3 = 0.0;
   double ab1 = 1.0;
   double ab2 = 0.0;
   double ab3 = 0.0;

   /* For coupled solution approach */
   BlockMatrix *system_mat = NULL;
   SparseMatrix *uu = NULL;
   SparseMatrix *up = NULL;
   SparseMatrix *mono_mat = NULL;
   HypreParMatrix *hypre_mat = NULL;
   HYPRE_BigInt glob_size;
   HYPRE_BigInt row_starts[2];
   MUMPSSolver *mumps = NULL;
   BlockVector rhs;

   /* velocity and its convection at previous time step */
   Vector u1;
   Vector Cu1;

protected:

   int dim = -1;
   Mesh *mesh = NULL;   // not owned

   FiniteElementSpace *ufes = NULL; // not owned
   FiniteElementSpace *pfes = NULL; // not owned

   int order = -1;
   const double sigma = -1.0;
   double kappa = -1.0;
   double dt = -1.0;

   Array<int> block_offsets;
   Array<int> vblock_offsets;

   Array<int> neumann_attr;

   BlockVector *x = NULL;     // owned
   BlockVector *rhs0 = NULL;   // owned

   GridFunction *u = NULL;    // owned
   GridFunction *p = NULL;    // owned

   LinearForm *fform = NULL;  // owned
   LinearForm *gform = NULL;  // owned

   BilinearForm *mass = NULL;      // owned
   BilinearForm *visc = NULL;      // owned
   MixedBilinearFormDGExtension *div = NULL; // owned
   NonlinearForm *conv = NULL;     // owned

   ConstantCoefficient *nu_coeff = NULL;  // owned
   ConstantCoefficient *zeta_coeff = NULL;  // owned
   ConstantCoefficient *minus_zeta = NULL;  // owned
   ConstantCoefficient *minus_one = NULL; // owned

public:
   NavierSolver(Mesh *mesh_, FiniteElementSpace *ufes_, FiniteElementSpace *pfes_)
      : mesh(mesh_), ufes(ufes_), pfes(pfes_), dim(mesh_->Dimension()),
        order(pfes_->GetMaxElementOrder()), minus_one(new ConstantCoefficient(-1.0))
   {
      block_offsets.SetSize(dim+2); // dimension + 1
      block_offsets[0] = 0;
      for (int d = 1; d <= dim; d++)
         block_offsets[d] = ufes->GetNDofs();
      block_offsets[dim+1] = pfes->GetVSize();
      block_offsets.PartialSum();

      vblock_offsets.SetSize(3); // number of variables + 1
      vblock_offsets[0] = 0;
      vblock_offsets[1] = ufes->GetVSize();
      vblock_offsets[2] = pfes->GetVSize();
      vblock_offsets.PartialSum();

      std::cout << "***********************************************************\n";
      for (int d = 1; d < block_offsets.Size(); d++)
         printf("dim(%d) = %d\n", d, block_offsets[d] - block_offsets[d-1]);
      printf("dim(q) = %d\n", block_offsets.Last());
      std::cout << "***********************************************************\n";

      /* solution/rhs vectors */
      x = new BlockVector(vblock_offsets);
      rhs0 = new BlockVector(vblock_offsets);

      /* grid functions that represent blocks of solution vector */
      u = new GridFunction;
      p = new GridFunction;
      u->MakeRef(ufes, x->GetBlock(0), 0);
      p->MakeRef(pfes, x->GetBlock(1), 0);

      /* linear forms that represent blocks of rhs vector */
      fform = new LinearForm;
      gform = new LinearForm;
      fform->Update(ufes, rhs0->GetBlock(0), 0);
      gform->Update(pfes, rhs0->GetBlock(1), 0);

      /* operators that are applied to grid functions */
      mass = new BilinearForm(ufes);
      visc = new BilinearForm(ufes);
      div = new MixedBilinearFormDGExtension(ufes, pfes);
      conv = new NonlinearForm(ufes);

      nu_coeff = new ConstantCoefficient(nu);
      zeta_coeff = new ConstantCoefficient(zeta);
      minus_zeta = new ConstantCoefficient(-zeta);
      kappa = (order + 2) * (order + 2);
   }

   virtual ~NavierSolver()
   {
      delete x;
      delete rhs0;

      delete u;
      delete p;

      delete fform;
      delete gform;

      delete mass;
      delete visc;
      delete conv;
      delete div;

      delete nu_coeff;
      delete zeta_coeff;
      delete minus_zeta;
      delete minus_one;

      delete system_mat;
      delete uu;
      delete up;

      delete mono_mat;
      delete hypre_mat;
   }

   GridFunction* GetVelGridFunction() { return u; }
   GridFunction* GetPresGridFunction() { return p; }

   void SetupOperators(Array<Array<int> *> &u_ess_attr, Array<VectorCoefficient *> &u_coeff)
   {
      assert(u_ess_attr.Size() == u_coeff.Size());

      mass->AddDomainIntegrator(new VectorMassIntegrator);

      visc->AddDomainIntegrator(new VectorDiffusionIntegrator(*nu_coeff));
      visc->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));
      for (int k = 0; k < u_ess_attr.Size(); k++)
      {
         assert(u_ess_attr[k]->Size() == mesh->bdr_attributes.Max());
         visc->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa), *u_ess_attr[k]);
      }

      div->AddDomainIntegrator(new VectorDivergenceIntegrator(*minus_one));
      div->AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);
      for (int k = 0; k < u_ess_attr.Size(); k++)
      {
         div->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, *u_ess_attr[k]);
      }

      /* domain integrator */
      // conv->AddDomainIntegrator(new VectorConvectionTrilinearFormIntegrator(*zeta_coeff));
      // /* Lax-Friedrichs integrator */
      conv->AddDomainIntegrator(new IncompressibleInviscidFluxNLFIntegrator(*minus_zeta));
      conv->AddInteriorFaceIntegrator(new DGLaxFriedrichsFluxIntegrator(*minus_zeta));
      for (int k = 0; k < u_ess_attr.Size(); k++)
      {
         conv->AddBdrFaceIntegrator(new DGLaxFriedrichsFluxIntegrator(*minus_zeta, u_coeff[k]), *u_ess_attr[k]);
      }
      /* Lax-Friedrichs solver needs neumann boundary integrator as well */
      neumann_attr.SetSize(mesh->bdr_attributes.Max());
      neumann_attr = 1;
      for (int k = 0; k < u_ess_attr.Size(); k++)
         for (int b = 0; b < mesh->bdr_attributes.Max(); b++)
         {
            neumann_attr[b] = (neumann_attr[b] && !((*u_ess_attr[k])[b]));
         }
      conv->AddBdrFaceIntegrator(new DGLaxFriedrichsFluxIntegrator(*minus_zeta), neumann_attr);

      // fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      for (int k = 0; k < u_ess_attr.Size(); k++)
      {
         assert(u_coeff[k]);
         fform->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(*u_coeff[k], *nu_coeff, sigma, kappa), *u_ess_attr[k]);
      }

      // gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
      for (int k = 0; k < u_ess_attr.Size(); k++)
      {
         assert(u_coeff[k]);
         gform->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(*u_coeff[k]), *u_ess_attr[k]);
      }
   }

   void AssembleOperators()
   {
      mass->Assemble();
      visc->Assemble();
      div->Assemble();

      /* mass will be finalized later, to build other sparse matrices. */
      visc->Finalize();
      div->Finalize();

      fform->Assemble();
      gform->Assemble();

      fform->SyncAliasMemory(*rhs0);
      gform->SyncAliasMemory(*rhs0);   
   }

   void InitializeTimeIntegration(const double &dt_)
   {
      assert(dt_ > 0.0);
      dt = dt_;

      uu = new SparseMatrix(mass->SpMat());
      (*uu) *= bd0 / dt;
      (*uu) += visc->SpMat();
      uu->Finalize();
      mass->Finalize();
      up = Transpose(div->SpMat());

      system_mat = new BlockMatrix(vblock_offsets);
      system_mat->SetBlock(0, 0, uu);
      system_mat->SetBlock(0, 1, up);
      system_mat->SetBlock(1, 0, &(div->SpMat()));

      mono_mat = system_mat->CreateMonolithic();
      glob_size = vblock_offsets.Last();
      row_starts[0] = 0;
      row_starts[1] = vblock_offsets.Last();
      hypre_mat = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, mono_mat);

      mumps = new MUMPSSolver(MPI_COMM_WORLD);
      mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps->SetOperator(*hypre_mat);

      Cu1.SetSize(u->Size());
      rhs.Update(vblock_offsets);
   }

   void Step(double &time, int step)
   {
      /* copy velocity */
      u1 = (*u);

      /* evaluate nonlinear advection at previous time step */
      conv->Mult(*u, Cu1);

      /* Base right-hand side for boundary conditions and forcing */
      rhs = (*rhs0);

      /* Add nonlinear convection */
      rhs.GetBlock(0).Add(-ab1, Cu1);

      /* Add time derivative term */
      // TODO: extend for high order bdf schemes
      mass->AddMult(u1, rhs.GetBlock(0), -bd1 / dt);

      /* Solve for the next step */
      mumps->Mult(rhs, *x);

      time += dt;
   }

   void SanityCheck()
   {
      assert(!isnan(u->Min()) && !isnan(u->Max()));
      assert(!isnan(p->Min()) && !isnan(p->Max()));
   }

   double ComputeCFL(double dt)
   {
      int vdim = ufes->GetVDim();

      Vector ux, uy, uz;
      Vector ur, us, ut;
      double cflx = 0.0;
      double cfly = 0.0;
      double cflz = 0.0;
      double cflm = 0.0;
      double cflmax = 0.0;

      for (int e = 0; e < ufes->GetNE(); ++e)
      {
         const FiniteElement *fe = ufes->GetFE(e);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                fe->GetOrder());
         ElementTransformation *tr = ufes->GetElementTransformation(e);

         u->GetValues(e, ir, ux, 1);
         ur.SetSize(ux.Size());
         u->GetValues(e, ir, uy, 2);
         us.SetSize(uy.Size());
         if (vdim == 3)
         {
            u->GetValues(e, ir, uz, 3);
            ut.SetSize(uz.Size());
         }

         double hmin = mesh->GetElementSize(e, 1) /
                     (double) ufes->GetElementOrder(0);

         for (int i = 0; i < ir.GetNPoints(); ++i)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);
            tr->SetIntPoint(&ip);
            const DenseMatrix &invJ = tr->InverseJacobian();
            const double detJinv = 1.0 / tr->Jacobian().Det();

            if (vdim == 2)
            {
               ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)) * detJinv;
               us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)) * detJinv;
            }
            else if (vdim == 3)
            {
               ur(i) = (ux(i) * invJ(0, 0) + uy(i) * invJ(1, 0)
                        + uz(i) * invJ(2, 0))
                     * detJinv;
               us(i) = (ux(i) * invJ(0, 1) + uy(i) * invJ(1, 1)
                        + uz(i) * invJ(2, 1))
                     * detJinv;
               ut(i) = (ux(i) * invJ(0, 2) + uy(i) * invJ(1, 2)
                        + uz(i) * invJ(2, 2))
                     * detJinv;
            }

            cflx = fabs(dt * ux(i) / hmin);
            cfly = fabs(dt * uy(i) / hmin);
            if (vdim == 3)
            {
               cflz = fabs(dt * uz(i) / hmin);
            }
            cflm = cflx + cfly + cflz;
            cflmax = fmax(cflmax, cflm);
         }
      }

      double cflmax_global = cflmax;

      return cflmax_global;
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
   double dt = 1e-2;
   int Nt = 1;
   bool pres_dbc = false;
   const char *device_config = "cpu";
   bool visualization = 1;
   int vis_interval = 10;

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
   args.AddOption(&zeta, "-z", "--zeta",
                  "constant factor for nonlinear convection.");
   args.AddOption(&nu, "-nu", "--nu",
                  "nondimensional viscosity. inverse of Reynolds number.");
   args.AddOption(&dt, "-dt", "--dt",
                  "time step size for time integration.");
   args.AddOption(&Nt, "-nt", "--nt",
                  "number of time steps for time integration.");
   args.AddOption(&vis_interval, "-vi", "--visual-interval",
                  "time step interval for visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *dg_coll(new DG_FECollection(order+1, dim));
   FiniteElementCollection *pdg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *ph1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes = new FiniteElementSpace(mesh, dg_coll);
   FiniteElementSpace *ufes = new FiniteElementSpace(mesh, dg_coll, dim);
   FiniteElementSpace *pfes = new FiniteElementSpace(mesh, pdg_coll);

   assert(mesh->bdr_attributes.Max() >= 4);
   Array<int> inlet_attr(mesh->bdr_attributes.Max());
   Array<int> noslip_attr(mesh->bdr_attributes.Max());
   Array<Array<int> *> u_attrs(2);
   u_attrs[0] = &inlet_attr;
   u_attrs[1] = &noslip_attr;
   
   inlet_attr = 0;
   noslip_attr = 0;
   inlet_attr[3] = 1;
   // noslip_attr[0] = 1;
   // noslip_attr[2] = 1;
   if (mesh->bdr_attributes.Max() > 4)
      noslip_attr[4] = 1;

   Vector u_inlet(dim), u_zero(dim);
   u_zero = 0.0;
   u_inlet = 0.0;
   u_inlet(0) = 1.0;
   Array<VectorCoefficient *> u_coeffs(2);
   // u_coeffs[0] = new VectorFunctionCoefficient(dim, uFun_ex);
   u_coeffs[0] = new VectorConstantCoefficient(u_inlet);
   u_coeffs[1] = new VectorConstantCoefficient(u_zero);

   NavierSolver navier(mesh, ufes, pfes);

   navier.SetupOperators(u_attrs, u_coeffs);
   navier.AssembleOperators();

   GridFunction *u = navier.GetVelGridFunction();
   GridFunction *p = navier.GetPresGridFunction();

   u->ProjectCoefficient(*u_coeffs[0]);
   (*p) = 0.0;

   ParaViewDataCollection pd("NavierStokes", mesh);
   pd.SetPrefixPath("NavierStokes_ParaView");
   pd.RegisterField("vel", u);
   pd.RegisterField("pres", p);
   pd.SetLevelsOfDetail(order);

   navier.InitializeTimeIntegration(dt);
   double time = 0.0;
   for (int step = 0; step < Nt; step++)
   {
      if (step % vis_interval == 0)
      {
         double cfl = navier.ComputeCFL(dt);
         printf("time step: %d, CFL: %.3e\n", step, cfl);
         pd.SetCycle(step);
         pd.SetTime(time);
         pd.Save();
      }

      navier.Step(time, step);
      navier.SanityCheck();
   }

   // 17. Free the used memory.
   DeletePointers(u_coeffs);
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
   double yi(x(1));

   u.SetSize(x.Size());
   u = 0.0;
   u(0) = 1.0 - 4.0 * (yi - 0.5) * (yi - 0.5);
}