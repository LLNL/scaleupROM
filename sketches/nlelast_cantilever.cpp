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

static double K = 5.0;
static double mu = 5.0;
static const double pi = 4.0 * atan(1.0);
//static double f = -1.0;
static double f = 1.0;
//static double mu = 1.0;

// A proxy Operator used for FOM Newton Solver.
// Similar to SteadyNSOperator.
class SimpleNLElastOperator : public Operator
{
protected:
   NonlinearForm nlform;

   // Jacobian matrix objects
   mutable SparseMatrix *J = NULL;
   mutable HypreParMatrix *jac_hypre = NULL;
   HYPRE_BigInt sys_glob_size;
   mutable HYPRE_BigInt sys_row_starts[2];

public:
   SimpleNLElastOperator(const int hw, NonlinearForm &nlform_);

   virtual ~SimpleNLElastOperator();

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual Operator &GetGradient(const Vector &x) const;
};


SimpleNLElastOperator::SimpleNLElastOperator(const int hw, NonlinearForm &nlform_)
   : Operator(hw, hw),nlform(nlform_)
{
   // TODO: this needs to be changed for parallel implementation.
   sys_glob_size = hw;
   sys_row_starts[0] = 0;
   sys_row_starts[1] = hw;
}

SimpleNLElastOperator::~SimpleNLElastOperator()
{
   //delete J;
   delete jac_hypre;
   //delete nlform;
}

void SimpleNLElastOperator::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   nlform.Mult(x, y);
}

Operator& SimpleNLElastOperator::GetGradient(const Vector &x) const
{

   J = dynamic_cast<SparseMatrix *>(&(nlform.GetGradient(x)));
   jac_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, J);
   return *jac_hypre;
   //return nlform.GetGradient(x);
}

// Define the analytical solution and forcing terms / boundary conditions
void NullSolution(const Vector &x, Vector &u);
void NullDefSolution(const Vector &x, Vector &u);
void cantileverf(const Vector &x, Vector &u);
void CheckGradient(NonlinearForm oper, FiniteElementSpace *fes);


void CheckGradient(NonlinearForm oper, FiniteElementSpace *fes);

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
   bool solve = false;
   bool check_grad = false;
   bool nonlinear = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.AddOption(&solve, "-s", "--solve","-ns", "--nosolve",
                  "Solve the system.");
   args.AddOption(&check_grad, "-cg", "--checkgrad","-ncg", "--nocheckgrad",
                  "Check gradients.");
   args.AddOption(&nonlinear, "-nl", "--nonlinear","-lin", "--linear",
                  "toggle linear/nonlinear elasticity.");
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
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));

   cout<<"dg_coll is: "<<dg_coll<<endl;
   cout<<"h1_coll is: "<<h1_coll<<endl;

   FiniteElementSpace *fes;
   bool use_dg = false;
   if (use_dg)
   {
      fes = new FiniteElementSpace(mesh, dg_coll, dim);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, h1_coll, dim);
   }

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient* dir;
   VectorFunctionCoefficient* force;
   VectorFunctionCoefficient* n_force;

   dir = new VectorFunctionCoefficient(dim, NullDefSolution);
   force = new VectorFunctionCoefficient(dim, cantileverf); 
   n_force = new VectorFunctionCoefficient(dim, NullSolution);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   //    MemoryType mt = device.GetMemoryType();
    int fomsize = fes->GetTrueVSize();

   Vector x(fomsize), rhs(fomsize), rhs1(fomsize), rhs2(fomsize);
   rhs = 0.0;

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u;
   u.MakeRef(fes, x, 0);

   u = 0.0;
   if (nonlinear)
   {
   u.ProjectCoefficient(*dir);
   }
   
   
   double kappa = (order+1)*(order+1);

   DGHyperelasticModel* model;
   
   if (nonlinear)
   {
model = new NeoHookeanHypModel(mu, K);
   }
   else
   {
model = new LinElastMaterialModel(mu, K);
   }
   
   
   Array<int> u_ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   u_ess_attr = 0;
   u_ess_attr[3] = 1;

   Array<int> u_f_attr(mesh->bdr_attributes.Max());
   // this array of integer fentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   u_f_attr = 0;
   u_f_attr[1] = 1;

   Array<int> u_ess_n_attr(mesh->bdr_attributes.Max());
   u_ess_attr = 1;
   u_ess_attr[3] = 0;
   u_ess_attr[1] = 0;

   Array<int> u_ess_tdof;
   fes->GetEssentialTrueDofs(u_ess_attr, u_ess_tdof);

   LinearForm *gform = new LinearForm(fes);
   LinearForm *tempform = new LinearForm(fes);
   gform->Update(fes, rhs1, 0);
   gform->AddBdrFaceIntegrator(new DGHyperelasticDirichletLFIntegrator(
               *dir, model, 0.0, kappa), u_ess_attr);
   gform->Assemble();
   gform->SyncAliasMemory(rhs1);

   cout<<"u_ess_tdof.Size() is: "<<u_ess_tdof.Size()<<endl;
   cout<<"fomsize is: "<<fomsize<<endl;

   tempform->Update(fes, rhs2, 0);
   tempform->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(*force), u_f_attr);
   //tempform->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(*n_force), u_ess_n_attr);
   //tempform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*force), u_f_attr);
   tempform->Assemble();
   gform->SyncAliasMemory(rhs2);


   for (size_t i = 0; i < u_ess_tdof.Size(); i++)
   {
      rhs2[u_ess_tdof[i]] = 0.0;
   }

   //rhs += rhs1;
   cout<<"rhs.Norml2() is: "<<rhs.Norml2()<<endl;
   rhs += rhs2;
   cout<<"rhs.Norml2() is: "<<rhs.Norml2()<<endl;

   // 9. Assemble the finite element matrices
   
   NonlinearForm *nlform(new NonlinearForm(fes));
   nlform->AddDomainIntegrator(new HyperelasticNLFIntegratorHR(model));
   nlform->AddBdrFaceIntegrator(new DGHyperelasticNLFIntegrator(model, 0.0, kappa), u_ess_attr);
   //nlform->AddInteriorFaceIntegrator( new DGHyperelasticNLFIntegrator(model, 0.0, kappa));
   nlform->SetEssentialTrueDofs(u_ess_tdof);

   SimpleNLElastOperator oper(fomsize, *nlform);

   if (check_grad)
   {
   CheckGradient(*nlform, fes);
   /* SparseMatrix *jac = dynamic_cast<SparseMatrix *>(&(nlform->GetGradient(x)));
   DenseMatrix J(*(jac->ToDenseMatrix()));
   Vector ev(J.Size());
   cout<<"J.Det() is: "<<J.Det()<<endl; */
   }

   // Test applying nonlinear form
   Vector _x(x);
   Vector _y0(x);
   Vector _y1(x);

   //GridFunction x_ref(fes);
   //mesh->GetNodes(x_ref);
   //x_ref.ProjectCoefficient(exact_sol);
   //_x = x_ref.GetTrueVector();

   _y1 = 0.0;
   oper.Mult(_x, _y1); //MFEM Neohookean

   double _y1_norm = _y1.Norml2();

   

   /* for (size_t i = 0; i < _y1.Size(); i++)
   {
      cout<<"LHS(i) is: "<<_y1(i)<<endl;
      cout<<"RHS(i) is: "<<rhs(i)<<endl;
      _y1(i) -= rhs(i);
      cout<<"res(i) is: "<<_y1(i)<<endl;
      cout<<endl;

   } */

   /* for (size_t i = 0; i < u_ess_tdof.Size(); i++)
   {
      //_y1[u_ess_tdof[i]] -= rhs[u_ess_tdof[i]];
      if (abs(_y1[u_ess_tdof[i]] - rhs[u_ess_tdof[i]]) > 0.001)
      {
      cout<<"LHS(i) is: "<<_y1[u_ess_tdof[i]]<<endl;
      cout<<"RHS(i) is: "<<rhs[u_ess_tdof[i]]<<endl;
      cout<<"res(i) is: "<<_y1[u_ess_tdof[i]] - rhs[u_ess_tdof[i]]<<endl;
      cout<<"i is: "<<u_ess_tdof[i]<<endl;
      cout<<"ui is: "<<u(u_ess_tdof[i])<<endl;
      cout<<endl;

      }
      
   } */

    /* for (size_t i = 0; i < _y1.Size(); i++)
   {
      const double res = _y1(i) - rhs(i);
      if (abs(res) > 0.001)
      {
      cout<<"LHS(i) is: "<<_y1(i)<<endl;
      cout<<"RHS(i) is: "<<rhs(i)<<endl;
      cout<<"res(i) is: "<<res<<endl;
      cout<<"i is: "<<i<<endl;
      cout<<"ui is: "<<u(i)<<endl;
      cout<<endl;
      }
      _y1(i) -= rhs(i);
   }  */

   cout<<"(_y1 - rhs).Norml2() is: "<<_y1.Norml2()<<endl;
   cout<<"_y1_norm is: "<<_y1_norm<<endl;
   cout<<"rel_err is: "<<_y1.Norml2()/_y1_norm<<endl;

if (solve)
{
   int maxIter(10000);
   double rtol(1.e-9);
   double atol(1.e-9);

   // MINRESSolver J_solver;
   MUMPSSolver J_solver(MPI_COMM_SELF);
   J_solver.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   J_solver.SetPrintLevel(-1);

   NewtonSolver newton_solver;
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(J_solver);
   newton_solver.SetOperator(oper);
   newton_solver.SetPrintLevel(1); // print Newton iterations
   newton_solver.SetRelTol(rtol);
   newton_solver.SetAbsTol(atol);
   newton_solver.SetMaxIter(30);
   newton_solver.Mult(rhs, x);

   
   
}

for (size_t i = 0; i < _x.Size(); i++)
   {
      //cout<<"u0(i) is: "<<_x(i)<<endl;
      //cout<<"u(i) is: "<<u(i)<<endl;
      x(i) -= _x(i);
      //cout<<"disp(i) is: "<<x(i)<<endl;
      //cout<<endl;
   }
   
   cout<<"minimum displacement is: "<<x.Min()<<endl;
   cout<<"maximum displacement is: "<<x.Max()<<endl;

/* 
   // 15. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("nlelast_mms_paraview", mesh);
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save(); */

   // 17. Free the used memory.
   delete gform;
   delete tempform;
   delete fes;
   delete dg_coll;
   delete h1_coll;
   delete mesh;
   delete model;
   delete dir;
   delete force;

   return 0;
}



    void NullSolution(const Vector &x, Vector &u)
   {
      u = 0.0;
      //u -= x;
   }

   void NullDefSolution(const Vector &x, Vector &u)
   {
      u = 0.0;
      u = x;
      //u = 1.0;
   }

    void cantileverf(const Vector &x, Vector &u)
   {
      u = 0.0;
      u(1) = -f;
   }

void CheckGradient(NonlinearForm oper, FiniteElementSpace *fes)
{   
   // if (!use_dg)
   //    fes->GetEssentialTrueDofs(ess_attr, ess_tdof);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u(fes), us(fes);
   for (int k = 0; k < u.Size(); k++)
   {
      u[k] = UniformRandom();
      us[k] = UniformRandom();
   }

   ConstantCoefficient one(1.0);

   Vector Nu(u.Size());
   oper.Mult(u, Nu);
   double J0 = us * (Nu);
   printf("J0: %.5E\n", J0);

   SparseMatrix *jac = dynamic_cast<SparseMatrix *>(&(oper.GetGradient(u)));
   Vector grad(u.Size());
   jac->MultTranspose(us, grad);
   double gg = grad * grad;
   printf("gg: %.5E\n", gg);

   GridFunction u0(fes);
   u0 = u;

   double error1 = 1.0e10;
   printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "amp", "J0", "J1", "dJdx", "error");
   for (int k = 0; k < 40; k++)
   {
      //double amp = pow(10.0, -0.25 * k);
      double amp = pow(10.0, -5.0-0.25 * k);
      double dx = amp;
      if (gg > 1.0e-14) dx /= sqrt(gg);

      u.Set(1.0, u0);
      u.Add(dx, grad);

      oper.Mult(u, Nu);
      double J1 = us * (Nu);
      double dJdx = (J1 - J0) / dx;
      double error = abs((dJdx - gg));
      if (gg > 1.0e-14) error /= abs(gg);

      printf("%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n", amp, J0, J1, dJdx, error);

      if (k > 4)
      {
         if (error > error1)
            break;
         else
            error1 = error;
      }
   }

   return;
}
