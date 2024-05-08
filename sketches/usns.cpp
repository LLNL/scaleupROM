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
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

static double nu = 0.1;
static double zeta = 1.0;

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

protected:

   int dim = -1;
   Mesh *mesh = NULL;   // not owned

   FiniteElementSpace *ufes = NULL; // not owned
   FiniteElementSpace *pfes = NULL; // not owned

   int order = -1;
   const double sigma = -1.0;
   double kappa = -1.0;

   Array<int> block_offsets;
   Array<int> vblock_offsets;

   BlockVector *x = NULL;     // owned
   BlockVector *rhs = NULL;   // owned

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
      rhs = new BlockVector(vblock_offsets);

      /* grid functions that represent blocks of solution vector */
      u = new GridFunction;
      p = new GridFunction;
      u->MakeRef(ufes, x->GetBlock(0), 0);
      p->MakeRef(pfes, x->GetBlock(1), 0);

      /* linear forms that represent blocks of rhs vector */
      fform = new LinearForm;
      gform = new LinearForm;
      fform->Update(ufes, rhs->GetBlock(0), 0);
      gform->Update(pfes, rhs->GetBlock(1), 0);

      /* operators that are applied to grid functions */
      mass = new BilinearForm(ufes);
      visc = new BilinearForm(ufes);
      div = new MixedBilinearFormDGExtension(ufes, pfes);
      conv = new NonlinearForm(ufes);

      nu_coeff = new ConstantCoefficient(nu);
      zeta_coeff = new ConstantCoefficient(zeta);
      kappa = (order + 2) * (order + 2);
   }

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

      conv->AddDomainIntegrator(new VectorConvectionTrilinearFormIntegrator(*zeta_coeff));

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

      mass->Finalize();
      visc->Finalize();
      div->Finalize();

      fform->Assemble();
      gform->Assemble();

      fform->SyncAliasMemory(*rhs);
      gform->SyncAliasMemory(*rhs);   
   }

   virtual ~NavierSolver()
   {
      delete x;
      delete rhs;

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
      delete minus_one;
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
   bool pres_dbc = false;
   const char *device_config = "cpu";
   bool visualization = 1;

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
   // FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *ph1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes = new FiniteElementSpace(mesh, h1_coll);
   FiniteElementSpace *ufes = new FiniteElementSpace(mesh, h1_coll, dim);
   FiniteElementSpace *pfes = new FiniteElementSpace(mesh, ph1_coll);

   assert(mesh->bdr_attributes.Max() >= 4);
   Array<int> inlet_attr(mesh->bdr_attributes.Max());
   Array<int> noslip_attr(mesh->bdr_attributes.Max());
   Array<Array<int> *> u_attrs(2);
   u_attrs[0] = &inlet_attr;
   u_attrs[1] = &noslip_attr;
   
   inlet_attr = 0;
   noslip_attr = 0;
   inlet_attr[3] = 1;
   if (mesh->bdr_attributes.Max() > 4)
      noslip_attr[4] = 1;

   Vector u_inlet(dim), u_zero(dim);
   u_zero = 0.0;
   u_inlet = 1.0;
   Array<VectorCoefficient *> u_coeffs(2);
   u_coeffs[0] = new VectorConstantCoefficient(u_inlet);
   u_coeffs[1] = new VectorConstantCoefficient(u_zero);

   NavierSolver navier(mesh, ufes, pfes);

   navier.SetupOperators(u_attrs, u_coeffs);
   navier.AssembleOperators();

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