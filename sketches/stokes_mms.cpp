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

class SchurOperator : public Operator
{
protected:
   Operator *A, *B;//, *Bt;
   CGSolver *solver = NULL;

   int maxIter = 10000;
   double rtol = 1.0e-15;
   double atol = 1.0e-15;

public:
   SchurOperator(Operator* const A_, Operator* const B_)
      : Operator(B_->Height()), A(A_), B(B_)
   {
      solver = new CGSolver();
      solver->SetRelTol(rtol);
      solver->SetMaxIter(maxIter);
      solver->SetOperator(*A);
      solver->SetPrintLevel(0);
   };

   virtual ~SchurOperator()
   {
      delete solver;
   }
   
   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector x1(A->NumCols());
      B->MultTranspose(x, x1);

      Vector y1(x1.Size());
      solver->Mult(x1, y1);

      B->Mult(y1, y);
   }
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
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
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

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0), minus_one(-1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
//    MemoryType mt = device.GetMemoryType();
   BlockVector x(block_offsets), rhs(block_offsets);

   LinearForm *fform(new LinearForm);
   fform->Update(ufes, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff));
   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(pfes, rhs.GetBlock(dim), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();
   gform->SyncAliasMemory(rhs);

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k \grad u_h \cdot \grad v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(ufes));
   MixedBilinearForm *bVarf(new MixedBilinearForm(ufes, pfes));
   // MixedBilinearForm *btVarf(new MixedBilinearForm(fes, ufes));

//    // mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(k));
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
   bVarf->Assemble();
   bVarf->Finalize();

   // btVarf->AddDomainIntegrator(new MixedScalarWeakGradientIntegrator);
   // btVarf->Assemble();
   // btVarf->Finalize();

//    Array<int> vblock_offsets(3); // number of variables + 1
//    vblock_offsets[0] = 0;
//    vblock_offsets[1] = ufes->GetVSize();
//    vblock_offsets[2] = pfes->GetVSize();
//    vblock_offsets.PartialSum();
//    BlockMatrix darcyOp(vblock_offsets);

//    SparseMatrix *Bt = NULL;

//    SparseMatrix &M(mVarf->SpMat());
   // SparseMatrix &B(bVarf->SpMat());
//    {
//       printf("M size: %d x %d\n", M.NumRows(), M.NumCols());
//       printf("B size: %d x %d\n", B.NumRows(), B.NumCols());
//       std::string filename;
//       filename = "stokes.M.txt";
//       PrintMatrix(M, filename);
//       filename = "stokes.B.txt";
//       PrintMatrix(B, filename);
//    }
   // B *= -1.;
//    Bt = TransposeAbstractSparseMatrix(B, 0);

//    darcyOp.SetBlock(0,0, &M);
//    darcyOp.SetBlock(0,1, Bt);
//    darcyOp.SetBlock(1,0, &B);
// {
//    SparseMatrix *D = darcyOp.CreateMonolithic();
//    std::string filename;
//    filename = "stokes.D.txt";
//    PrintMatrix(*D, filename);
// }

   Array<int> ess_attr(mesh->bdr_attributes.Max());
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   ess_attr = 1;
   Array<int> u_ess_tdof, p_ess_tdof, empty;
   ufes->GetEssentialTrueDofs(ess_attr, u_ess_tdof);
   pfes->GetEssentialTrueDofs(ess_attr, p_ess_tdof);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(ufes, x.GetBlock(0), 0);
   p.MakeRef(pfes, x.GetBlock(dim), 0);

   u = 0.0;
   p = 0.0;

   // add dirichlet boundary condition.
   // Vector ud(dim);
   // ud[0] = 1.0; ud[1] = 0.0;
   // VectorConstantCoefficient one(ud);
   Array<int> bdrAttr(mesh->bdr_attributes.Max());
   bdrAttr = 1;
   // bdrAttr[2] = 1;
   // u.ProjectBdrCoefficient(one, bdrAttr);
   u.ProjectBdrCoefficient(ucoeff, bdrAttr);
   p.ProjectCoefficient(pcoeff);
   const double p_const = p.Sum() / static_cast<double>(p.Size());

   // A \ F2.
   SparseMatrix A;
   OperatorHandle Bh;
   Vector U1, F1;
   Vector P1, G1;
   mVarf->FormLinearSystem(u_ess_tdof, u, *fform, A, U1, F1);

   bVarf->FormRectangularLinearSystem(u_ess_tdof, empty, u, *gform, Bh, U1, G1);
   Operator &B(*Bh);

   int maxIter(10000);
   double rtol(1.e-15);
   double atol(1.e-15);
   chrono.Clear();
   chrono.Start();
   // MINRESSolver solver;
   CGSolver solver;
   solver.SetAbsTol(atol);
   // solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(A);
   // solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(0);
   // x = 0.0;
   solver.Mult(F1, U1);
   // if (device.IsEnabled()) { x.HostRead(); }
   chrono.Stop();

   // B * A^{-1} * F1 - G1
   Vector R2(pfes->GetVSize());
   bVarf->Mult(U1, R2);
   R2 -= G1;

   // B.BuildTranspose();
   SchurOperator schur(&A, &B);
   CGSolver solver2;
   solver2.SetOperator(schur);
   // solver2.SetPrintLevel(1);
   // solver2.SetAbsTol(rtol);
   // solver2.SetMaxIter(maxIter);
   OrthoSolver ortho;
   ortho.SetSolver(solver2);
   ortho.SetOperator(schur);
   printf("Solving for pressure\n");
   // printf("%d ?= %d ?= %d\n", R2.Size(), p.Size(), ortho.Height());
   // solver2.Mult(R2, p);
   ortho.Mult(R2, p);
   p += p_const;
   printf("Pressure is solved.\n");

   // AU = F - B^T * P;
   Vector F3(F1.Size());
   B.MultTranspose(p, F3);
   F3 *= -1.0;
   F3 += F1;

   printf("Solving for velocity\n");
   solver.Mult(F3, u);
   printf("Velocity is solved.\n");

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
   ParaViewDataCollection paraview_dc("Example5", mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   // paraview_dc.SetCycle(0);
//    paraview_dc.SetDataFormat(VTKFormat::BINARY);
//    paraview_dc.SetHighOrderOutput(true);
//    paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("velocity",&u);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.Save();

   // 17. Free the used memory.
   delete fform;
   delete gform;
   // delete invM;
   // delete invS;
   // delete S;
   // delete Bt;
   // delete MinvBt;
   delete mVarf;
   delete bVarf;
   delete fes;
   delete ufes;
   delete pfes;
   // delete qfes;
   delete h1_coll;
   delete ph1_coll;
   // delete W_space;
   // delete R_space;
   delete l2_coll;
   // delete hdiv_coll;
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
