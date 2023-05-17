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
#include "linalg_utils.hpp"

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

/** \f$ (f, v \cdot n)_{\partial\Omega} \f$ for vector test function
    v=(v1,...,vn) where all vi are in the same scalar FE space and f is a
    scalar function. */
class VectorBoundaryTangentLFIntegrator : public LinearFormIntegrator
{
private:
   double Sign;
   VectorCoefficient *f;
   Vector shape, nor, fvec;

public:
   VectorBoundaryTangentLFIntegrator(VectorCoefficient &QG, double s = 1.0,
                                    const IntegrationRule *ir = NULL)
      : LinearFormIntegrator(ir), Sign(s), f(&QG) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
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
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));
   FiniteElementCollection *ph1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes = new FiniteElementSpace(mesh, h1_coll);
   FiniteElementSpace *ufes = new FiniteElementSpace(mesh, h1_coll, dim);
   FiniteElementSpace *pfes = new FiniteElementSpace(mesh, ph1_coll);

   // 6. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(dim+2); // number of variables + 1
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

   Array<int> u_ess_attr(mesh->bdr_attributes.Max());
   Array<int> p_ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   u_ess_attr = 1;
   p_ess_attr = 0;
   u_ess_attr[1] = 0;
   p_ess_attr[1] = 1;
   Array<int> u_ess_tdof, p_ess_tdof, empty;
   ufes->GetEssentialTrueDofs(u_ess_attr, u_ess_tdof);
   pfes->GetEssentialTrueDofs(p_ess_attr, p_ess_tdof);

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0), minus_one(-1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient minus_fcoeff(dim, fFun, &minus_one);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   GridFunction g_gf(pfes);
   g_gf.ProjectCoefficient(gcoeff);
   g_gf *= -k.constant;
   GradientGridFunctionCoefficient grad_gcoeff(&g_gf);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
//    MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets), rhs(block_offsets);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(ufes, x.GetBlock(0), 0);
   p.MakeRef(pfes, x.GetBlock(dim), 0);

   u = 0.0;
   u.ProjectBdrCoefficient(ucoeff, u_ess_attr);

   p.ProjectCoefficient(pcoeff);
   const double p_const = p.Sum() / static_cast<double>(p.Size());
   p = 0.0;
   p.ProjectBdrCoefficient(pcoeff, p_ess_attr);

   // Solve for pressure first.
   // Taking the divergence of the first equation, with \nabla \cdot u = g, leads to
   //       \nabla^2 p = \nabla \cdot (- k \nabla g + f)

   BilinearForm *pVarf(new BilinearForm(pfes));
   pVarf->AddDomainIntegrator(new DiffusionIntegrator);
   pVarf->Assemble();
   pVarf->Finalize();

   LinearForm *pform(new LinearForm);
   pform->Update(pfes, rhs.GetBlock(dim), 0);
   pform->AddDomainIntegrator(new DomainLFGradIntegrator(fcoeff));
   pform->AddDomainIntegrator(new DomainLFGradIntegrator(grad_gcoeff));
   pform->Assemble();
   pform->SyncAliasMemory(rhs);

   // A \ F2.
   SparseMatrix AP;
   Vector P1, G1;
   pVarf->FormLinearSystem(p_ess_tdof, p, *pform, AP, P1, G1);

   int maxIter(10000);
   double rtol(1.e-15);
   double atol(1.e-15);
   CGSolver solver2;
   solver2.SetOperator(AP);
   solver2.SetPrintLevel(0);
   solver2.SetAbsTol(rtol);
   solver2.SetMaxIter(maxIter);
   OrthoSolver ortho;
   if (p_ess_tdof.Size() == 0)
   {
      ortho.SetSolver(solver2);
      ortho.SetOperator(AP);
   }
   printf("Solving for pressure\n");
   // printf("%d ?= %d ?= %d\n", R2.Size(), p.Size(), ortho.Height());
   // solver2.Mult(R2, p);
   if (p_ess_tdof.Size() == 0)
      ortho.Mult(G1, P1);
   else
      solver2.Mult(G1, P1);
   printf("Pressure is solved.\n");

   BilinearForm *uVarf(new BilinearForm(ufes));
   uVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(k));
   uVarf->Assemble();
   uVarf->Finalize();

   GradientGridFunctionCoefficient grad_pcoeff(&p);
   OperatorHandle Bh;
   MixedBilinearForm *bVarf(new MixedBilinearForm(pfes, ufes));
   bVarf->AddDomainIntegrator(new GradientIntegrator);
   bVarf->Assemble();
   bVarf->FormRectangularSystemMatrix(empty, empty, Bh);
   bVarf->Finalize();

   LinearForm *uform(new LinearForm);
   uform->Update(ufes, rhs.GetBlock(0), 0);
   uform->AddDomainIntegrator(new VectorDomainLFIntegrator(minus_fcoeff));
   // uform->AddDomainIntegrator(new VectorDomainLFIntegrator(grad_pcoeff));
   uform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff), p_ess_attr);
   // uform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(minus_fcoeff), p_ess_attr);
   uform->AddBoundaryIntegrator(new VectorBoundaryTangentLFIntegrator(minus_fcoeff), p_ess_attr);
   uform->Assemble();
   bVarf->AddMult(p, *uform);
   uform->SyncAliasMemory(rhs);

   SparseMatrix AU;
   Vector U1, F1;
   uVarf->FormLinearSystem(u_ess_tdof, u, *uform, AU, U1, F1);

   CGSolver solver;
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(AU);
   solver.SetPrintLevel(0);

   printf("Solving for velocity\n");
   solver.Mult(F1, U1);
   printf("Velocity is solved.\n");

   if (p_ess_tdof.Size() == 0)
      p += p_const;

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
   // paraview_dc.SetCycle(0);
//    paraview_dc.SetDataFormat(VTKFormat::BINARY);
//    paraview_dc.SetHighOrderOutput(true);
//    paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   // 17. Free the used memory.
   delete pVarf;
   delete pform;
   delete uVarf;
   delete uform;
   delete fes;
   delete ufes;
   delete pfes;
   // delete qfes;
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

   u(0) = - exp(xi)*sin(yi);
   u(1) = - exp(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return exp(xi)*sin(yi);
}

void fFun(const Vector & x, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = exp(xi)*sin(yi);
   f(1) = exp(xi)*cos(yi);
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

void VectorBoundaryTangentLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   fvec.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      CalcOrtho(Tr.Jacobian(), nor);

      double tmp = 0.0;
      for (int d = 0; d < dim; d++) tmp += nor(d) * nor(d);

      el.CalcShape (ip, shape);
      f->Eval(fvec, Tr, ip);
      fvec.Add(-(nor * fvec) / tmp, nor);
      fvec *= sqrt(tmp) * Sign * ip.weight;
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += fvec(k) * shape(j);
         }
   }
}