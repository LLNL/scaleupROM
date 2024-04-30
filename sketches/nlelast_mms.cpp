// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "linalg_utils.hpp"
#include "nonlinear_integ.hpp"
#include "nlelast_integ.hpp"
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

static double K = 1.0;



// A proxy Operator used for FOM Newton Solver.
// Similar to SteadyNSOperator.
class SimpleNLElastOperator : public Operator
{
protected:
   NonlinearForm nlform;

   // Jacobian matrix objects
   mutable SparseMatrix *J = NULL;

public:
   SimpleNLElastOperator(const int hw, NonlinearForm &nlform_);

   virtual ~SimpleNLElastOperator();

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};


SimpleNLElastOperator::SimpleNLElastOperator(const int hw, NonlinearForm &nlform_)
   : Operator(hw, hw),nlform(nlform_)
{
}

SimpleNLElastOperator::~SimpleNLElastOperator()
{
   //delete J;
   //delete nlform;
}

void SimpleNLElastOperator::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   nlform.Mult(x, y);
}

Operator& SimpleNLElastOperator::GetGradient(const Vector &x) const
{
   return nlform.GetGradient(x);
}

// Define the analytical solution and forcing terms / boundary conditions
void SimpleExactRHSNeoHooke(const Vector &x, Vector &u);
void ExactSolutionNeoHooke(const Vector &x, Vector &u);

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


void SimpleExactSolutionNeoHooke(const Vector &X, Vector &U)
   {
      int dim = 2;
      int dof = U.Size()/dim;
      U = 0.0;
      for (size_t i = 0; i < U.Size()/dim; i++)
      {
         U(i) = pow(X(i), 2.0) + X(i);
         U(dof + i) = pow(X(dof + i), 2.0) + X(dof + i);
      }
   }

int main(int argc, char *argv[])
{
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "meshes/1x1.mesh";
   
   int order = 1;
   int refine = 0;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
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
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));


   FiniteElementSpace *fes = new FiniteElementSpace(mesh, dg_coll, dim);

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient exact_sol(dim, ExactSolutionNeoHooke);
   VectorFunctionCoefficient exact_RHS(dim, SimpleExactRHSNeoHooke);


   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   //    MemoryType mt = device.GetMemoryType();
    int fomsize = fes->GetTrueVSize();

   Vector x(fomsize), rhs(fomsize);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u;
   u.MakeRef(fes, x, 0);

   u = 0.0;
   u.ProjectCoefficient(exact_sol);
   

   double kappa = -1.0;
   double mu = 0.0;
   //double K = 1.0;
   NeoHookeanHypModel model(mu, K);

   LinearForm *gform = new LinearForm(fes);
   gform->AddBdrFaceIntegrator(new DGHyperelasticDirichletLFIntegrator(
               exact_sol, &model, 0.0, kappa));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices
   NonlinearForm *nlform(new NonlinearForm(fes));
   nlform->AddDomainIntegrator(new HyperelasticNLFIntegratorHR(&model));
   nlform->AddBdrFaceIntegrator( new DGHyperelasticNLFIntegrator(&model, 0.0, kappa));

   SimpleNLElastOperator oper(fomsize, *nlform);

   // Test applying nonlinear form
   Vector _x(x);
   Vector _y0(x);
   Vector _y1(x);

   GridFunction x_ref(fes);
   mesh->GetNodes(x_ref);
   _x = x_ref.GetTrueVector();

    _y0 = 0.0;
    SimpleExactSolutionNeoHooke(_x, _y0);

    _y1 = 0.0;
   nlform->Mult(_y0, _y1); //MFEM Neohookean

   cout<<"_y1.Norml2() is: "<<_y1.Norml2()<<endl;

   for (size_t i = 0; i < _y1.Size(); i++)
   {
      cout<<"LHS(i) is: "<<_y1(i)<<endl;
      cout<<"RHS(i) is: "<<gform->Elem(i)<<endl;
      _y1(i) -= gform->Elem(i);
      cout<<"res(i) is: "<<_y1(i)<<endl;
      cout<<endl;

   }
   
   cout<<"(_y1 - rhs).Norml2() is: "<<_y1.Norml2()<<endl;


bool solve = true;
if (solve)
{
   int maxIter(10000);
   double rtol(1.e-10);
   double atol(1.e-10);

   // MINRESSolver J_solver;
   GMRESSolver J_solver;
   J_solver.SetAbsTol(atol);
   J_solver.SetRelTol(rtol);
   J_solver.SetMaxIter(maxIter);
   J_solver.SetPrintLevel(-1);

   NewtonSolver newton_solver;
   newton_solver.iterative_mode = false;
   newton_solver.SetSolver(J_solver);
   newton_solver.SetOperator(oper);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rtol);
   newton_solver.SetAbsTol(atol);
   newton_solver.SetMaxIter(1);
   newton_solver.Mult(rhs, x);
   
}

   int order_quad = max(2, 2*(order+1)+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(exact_sol, irs);
   double norm_u = ComputeLpNorm(2., exact_sol, *mesh, irs);

   printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);

/* 
   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("nlelast_mms_paraview", mesh);
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save(); */

   // 17. Free the used memory.
   delete gform;
   delete fes;
   delete dg_coll;
   delete mesh;

   return 0;
}


void SimpleExactRHSNeoHooke(const Vector &x, Vector &u)
   {
      u = 0.0;
      u(0) = 2 * K * pow(1.0 + 2.0 * x(1), 2.0);
      u(1) = 2 * K * pow(1.0 + 2.0 * x(0), 2.0); 
      u *= -1.0;
   }

   void ExactSolutionNeoHooke(const Vector &x, Vector &u)
   {
      u = 0.0;
      //assert(dim == 2);
      assert(x.Size() == 2);
      u(0) = pow(x(0), 2.0) + x(0);
      u(1) = pow(x(1), 2.0) + x(1);
   }
