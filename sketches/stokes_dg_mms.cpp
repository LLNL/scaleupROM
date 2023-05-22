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
void mlap_uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void grad_pFun_ex(const Vector & x, Vector & y);
double mlap_pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);
void dudx_ex(const Vector & x, Vector & y);

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

/// Abstract base class BilinearFormIntegrator
class MixedBilinearFormFaceIntegrator : public BilinearFormIntegrator
{
protected:
   MixedBilinearFormFaceIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

public:
   virtual ~MixedBilinearFormFaceIntegrator() {};

   /** Abstract method used for assembling InteriorFaceIntegrators in a
       MixedBilinearFormDGExtension. */
   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat)
   { mfem_error("Abstract method MixedBilinearFormFaceIntegrator::AssembleFaceMatrix!\n"); }

};

class MixedBilinearFormDGExtension : public MixedBilinearForm
{
protected:
   /// interface integrators.
   Array<MixedBilinearFormFaceIntegrator*> interior_face_integs;

   /// Set of boundary face Integrators to be applied.
   Array<MixedBilinearFormFaceIntegrator*> boundary_face_integs;
   Array<Array<int>*> boundary_face_integs_marker; ///< Entries are not owned.

public:
   /** @brief Construct a MixedBilinearForm on the given trial, @a tr_fes, and
       test, @a te_fes, FiniteElementSpace%s. */
   /** The pointers @a tr_fes and @a te_fes are not owned by the newly
       constructed object. */
   MixedBilinearFormDGExtension(FiniteElementSpace *tr_fes, FiniteElementSpace *te_fes)
      : MixedBilinearForm(tr_fes, te_fes) {};

   /** @brief Create a MixedBilinearForm on the given trial, @a tr_fes, and
       test, @a te_fes, FiniteElementSpace%s, using the same integrators as the
       MixedBilinearForm @a mbf.

       The pointers @a tr_fes and @a te_fes are not owned by the newly
       constructed object.

       The integrators in @a mbf are copied as pointers and they are not owned
       by the newly constructed MixedBilinearForm. */
   MixedBilinearFormDGExtension(FiniteElementSpace *tr_fes,
                     FiniteElementSpace *te_fes,
                     MixedBilinearFormDGExtension *mbf);

   /// Adds new interior Face Integrator. Assumes ownership of @a bfi.
   void AddInteriorFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi);

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi. The array @a bdr_marker is stored internally
       as a pointer to the given Array<int> object. */
   void AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi,
                             Array<int> &bdr_marker);

   /// Access all integrators added with AddInteriorFaceIntegrator().
   Array<MixedBilinearFormFaceIntegrator*> *GetFBFI() { return &interior_face_integs; }

   /// Access all integrators added with AddBdrFaceIntegrator().
   Array<MixedBilinearFormFaceIntegrator*> *GetBFBFI() { return &boundary_face_integs; }
   /** @brief Access all boundary markers added with AddBdrFaceIntegrator().
       If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetBFBFI_Marker()
   { return &boundary_face_integs_marker; }

   virtual void Assemble(int skip_zeros = 1);

   // /// Compute the element matrix of the given element
   // void ComputeElementMatrix(int i, DenseMatrix &elmat);

   // /// Compute the boundary element matrix of the given boundary element
   // void ComputeBdrElementMatrix(int i, DenseMatrix &elmat);

   // /// Assemble the given element matrix
   // /** The element matrix @a elmat is assembled for the element @a i, i.e.
   //     added to the system matrix. The flag @a skip_zeros skips the zero
   //     elements of the matrix, unless they are breaking the symmetry of
   //     the system matrix.
   // */
   // void AssembleElementMatrix(int i, const DenseMatrix &elmat,
   //                            int skip_zeros = 1);

   // /// Assemble the given element matrix
   // /** The element matrix @a elmat is assembled for the element @a i, i.e.
   //     added to the system matrix. The vdofs of the element are returned
   //     in @a trial_vdofs and @a test_vdofs. The flag @a skip_zeros skips
   //     the zero elements of the matrix, unless they are breaking the symmetry
   //     of the system matrix.
   // */
   // void AssembleElementMatrix(int i, const DenseMatrix &elmat,
   //                            Array<int> &trial_vdofs, Array<int> &test_vdofs,
   //                            int skip_zeros = 1);

   // /// Assemble the given boundary element matrix
   // /** The boundary element matrix @a elmat is assembled for the boundary
   //     element @a i, i.e. added to the system matrix. The flag @a skip_zeros
   //     skips the zero elements of the matrix, unless they are breaking the
   //     symmetry of the system matrix.
   // */
   // void AssembleBdrElementMatrix(int i, const DenseMatrix &elmat,
   //                               int skip_zeros = 1);

   // /// Assemble the given boundary element matrix
   // /** The boundary element matrix @a elmat is assembled for the boundary
   //     element @a i, i.e. added to the system matrix. The vdofs of the element
   //     are returned in @a trial_vdofs and @a test_vdofs. The flag @a skip_zeros
   //     skips the zero elements of the matrix, unless they are breaking the
   //     symmetry of the system matrix.
   // */
   // void AssembleBdrElementMatrix(int i, const DenseMatrix &elmat,
   //                               Array<int> &trial_vdofs, Array<int> &test_vdofs,
   //                               int skip_zeros = 1);

   virtual ~MixedBilinearFormDGExtension();
};

class DGNormalFluxIntegrator : public MixedBilinearFormFaceIntegrator//, DGTraceIntegrator
{
private:
   int dim;
   int order;
   int p;

   int trial_dof1, trial_dof2, test_dof1, test_dof2;
   int trial_vdof1, trial_vdof2;

   double w, wn;
   int i, j, idof, jdof, jm;

   Vector nor, wnor;
   Vector shape1, shape2;
   // Vector divshape;
   Vector trshape1, trshape2;
   // DenseMatrix vshape1, vshape2;
   // Vector vshape1_n, vshape2_n;

public:
   DGNormalFluxIntegrator() {};
   virtual ~DGNormalFluxIntegrator() {};
   // /// Construct integrator with rho = 1, b = 0.5*a.
   // DGNormalFluxIntegrator(VectorCoefficient &u_, double a)
   //    : DGTraceIntegrator(u_, a) {};

   // /// Construct integrator with rho = 1.
   // DGNormalFluxIntegrator(VectorCoefficient &u_, double a, double b)
   //    : DGTraceIntegrator(u_, a, b) {};

   // DGNormalFluxIntegrator(Coefficient &rho_, VectorCoefficient &u_,
   //                         double a, double b)
   //    : DGTraceIntegrator(rho_, u_, a, b) {};

   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

// DGDiffusionFaceIntegrator
class DGVectorDiffusionIntegrator : public DGElasticityIntegrator
{
public:
   DGVectorDiffusionIntegrator(double alpha_, double kappa_)
      : DGElasticityIntegrator(alpha_, kappa_) {}

   DGVectorDiffusionIntegrator(Coefficient &mu_,
                              double alpha_, double kappa_)
      : DGElasticityIntegrator(alpha_, kappa_) { mu = &mu_; }

   using DGElasticityIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

protected:

   static void AssembleBlock(
      const int dim, const int row_ndofs, const int col_ndofs,
      const int row_offset, const int col_offset, const double jmatcoef,
      const Vector &row_shape, const Vector &col_shape, const Vector &col_dshape_dnM,
      DenseMatrix &elmat, DenseMatrix &jmat);
};

class DGVectorDirichletLFIntegrator : public DGElasticityDirichletLFIntegrator
{
public:
   DGVectorDirichletLFIntegrator(VectorCoefficient &uD_, Coefficient &mu_,
                                 double alpha_, double kappa_)
      : DGElasticityDirichletLFIntegrator(uD_, mu_, mu_, alpha_, kappa_) { lambda = NULL; }

   // virtual void AssembleRHSElementVect(const FiniteElement &el,
   //                                     ElementTransformation &Tr,
   //                                     Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

   using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for boundary integration \f$ L(v) = (g \cdot n, v) \f$
class DGBoundaryNormalLFIntegrator : public LinearFormIntegrator
{
private:
   Vector shape;
   VectorCoefficient &Q;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   DGBoundaryNormalLFIntegrator(VectorCoefficient &QG)
      : Q(QG) { }

   virtual bool SupportsDevice() { return false; }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
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

   double sigma = -1.0;
   double kappa = (order + 1) * (order + 1);

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
   FiniteElementCollection *dg_coll(new DG_FECollection(order+1, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *pdg_coll(new DG_FECollection(order, dim));
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
   // u_ess_attr[1] = 0;
   // p_ess_attr[1] = 1;
   Array<int> u_ess_tdof, p_ess_tdof, empty;
   ufes->GetEssentialTrueDofs(u_ess_attr, u_ess_tdof);
   pfes->GetEssentialTrueDofs(p_ess_attr, p_ess_tdof);
   bool pres_dbc = false;
   // for (int k = 0; k < p_ess_attr.Size(); k++) pres_dbc = (pres_dbc || static_cast<bool>(p_ess_attr[k]));

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0), minus_one(-1.0), one(1.0), zero(0.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient dudxcoeff(dim, dudx_ex);
   VectorFunctionCoefficient minus_fcoeff(dim, fFun, &minus_one);
   // VectorFunctionCoefficient grad_pcoeff(dim, grad_pFun_ex);
   VectorFunctionCoefficient mlap_ucoeff(dim, mlap_uFun_ex);
   // FunctionCoefficient mlap_pcoeff(mlap_pFun_ex);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);
   GridFunction u_ex(ufes), p_ex(pfes);
   u_ex.ProjectCoefficient(ucoeff);
   p_ex.ProjectCoefficient(pcoeff);

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
   // u.ProjectBdrCoefficient(ucoeff, u_ess_attr);

   p.ProjectCoefficient(pcoeff);
   const double p_const = p.Sum() / static_cast<double>(p.Size());
   p = 0.0;
   // p.ProjectBdrCoefficient(pcoeff, p_ess_attr);

   LinearForm *fform(new LinearForm);
   fform->Update(ufes, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
   // fform->AddDomainIntegrator(new VectorDomainLFIntegrator(mlap_ucoeff));

   // // Currently, mfem does not have a way to impose general tensor bc.
   // // dg fe space does not support boundary integrators. needs reimplmentation.
   // fform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff), p_ess_attr);
   // fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dudxcoeff), p_ess_attr);

   fform->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(ucoeff, k, sigma, kappa), u_ess_attr);

   fform->Assemble();
   fform->SyncAliasMemory(rhs);

   LinearForm *gform(new LinearForm);
   gform->Update(pfes, rhs.GetBlock(dim), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   // dg fe space does not support boundary integrators. needs reimplmentation.
   // Below two operators are essentially the same. Integration order must be set as 2 * order to guarantee the right convergence rate.
   gform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(ucoeff, 2, 0), u_ess_attr);
   // gform->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), u_ess_attr);
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
   MixedBilinearFormDGExtension *bVarf(new MixedBilinearFormDGExtension(ufes, pfes));

   mVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(k));
   mVarf->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(k, sigma, kappa), u_ess_attr);
   mVarf->Assemble();
   mVarf->Finalize();

   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
   bVarf->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, u_ess_attr);
   bVarf->Assemble();
   bVarf->Finalize();

   Vector R1(ufes->GetVSize());
   R1 = 0.0;
   // SparseMatrix M;
   // Vector F1(ufes->GetVSize());
   // mVarf->FormLinearSystem(u_ess_tdof, u, *fform, M, R1, F1);

// {
//    PrintMatrix(M, "stokes.M.txt");
//    PrintVector(F1, "stokes.f.txt");
// }

   printf("Setting up pressure RHS\n");
   int maxIter(10000);
   double rtol(1.e-10);
   double atol(1.e-10);
   // chrono.Clear();
   // chrono.Start();
   // MINRESSolver solver;
   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   // solver.SetOperator(M);
   solver.SetOperator(*mVarf);
   // solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(0);
   // x = 0.0;
   solver.Mult(*fform, R1);
   // solver.Mult(*fform, u);
   // mVarf->RecoverFEMSolution(R1, *fform, u);
   // if (device.IsEnabled()) { x.HostRead(); }
   // chrono.Stop();
   printf("Set up pressure RHS\n");

// {
//    int order_quad = max(2, 2*(order+1)+1);
//    const IntegrationRule *irs[Geometry::NumGeom];
//    for (int i=0; i < Geometry::NumGeom; ++i)
//    {
//       irs[i] = &(IntRules.Get(i, order_quad));
//    }

//    double err_u  = u.ComputeL2Error(ucoeff, irs);
//    double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);

//    printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
// }

   // B * A^{-1} * F1 - G1
   Vector R2(pfes->GetVSize());
   bVarf->Mult(R1, R2);
   R2 -= (*gform);

   SchurOperator schur(mVarf, bVarf);
   CGSolver solver2;
   solver2.SetOperator(schur);
   solver2.SetPrintLevel(0);
   solver2.SetAbsTol(rtol);
   solver2.SetMaxIter(maxIter);
   
   OrthoSolver ortho;
   if (!pres_dbc)
   {
      printf("Setting up OrthoSolver\n");
      ortho.SetSolver(solver2);
      ortho.SetOperator(schur);
      printf("OrthoSolver Set up.\n");
   }
   
   printf("Solving for pressure\n");
   // printf("%d ?= %d ?= %d\n", R2.Size(), p.Size(), ortho.Height());
   if (pres_dbc)
      solver2.Mult(R2, p);
   else
      ortho.Mult(R2, p);
   printf("Pressure is solved.\n");

   // AU = F - B^T * P;
   Vector F3(ufes->GetVSize());
   F3 = 0.0;
   // bVarf->MultTranspose(p, F3);
   // F3 *= -1.0;
   F3 += (*fform);

   printf("Solving for velocity\n");
   solver.Mult(F3, u);
   printf("Velocity is solved.\n");

   if (!pres_dbc)
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
   paraview_dc.RegisterField("u_exact",&u_ex);
   paraview_dc.RegisterField("pressure",&p);
   paraview_dc.RegisterField("p_exact",&p_ex);
   paraview_dc.Save();

   // 17. Free the used memory.
   delete fform;
   delete gform;
   delete mVarf;
   delete bVarf;
   delete fes;
   delete ufes;
   delete pfes;
   // delete qfes;
   delete dg_coll;
   delete h1_coll;
   delete pdg_coll;
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

void mlap_uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = 2.0 * cos(xi)*sin(yi);
   u(1) = - 2.0 * sin(xi)*cos(yi);
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * sin(xi)*sin(yi);
}

void grad_pFun_ex(const Vector & x, Vector & y)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   y.SetSize(2);

   y(0) = 2.0 * cos(xi)*sin(yi);
   y(1) = 2.0 * sin(xi)*cos(yi);
   return;
}

double mlap_pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));

   assert(x.Size() == 2);

   return 2.0 * sin(xi)*sin(yi);
   // return 0.0;
}

void fFun(const Vector & x, Vector & f)
{
   assert(x.Size() == 2);
   f.SetSize(x.Size());

   double xi(x(0));
   double yi(x(1));

   f(0) = 4.0 * cos(xi) * sin(yi);
   f(1) = 0.0;
}

double gFun(const Vector & x)
{
   assert(x.Size() == 2);

   return 0.0;
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

   y(0) = - exp(xi)*sin(yi);
   y(1) = - exp(xi)*cos(yi);
}

MixedBilinearFormDGExtension::MixedBilinearFormDGExtension(FiniteElementSpace *tr_fes,
                                                            FiniteElementSpace *te_fes,
                                                            MixedBilinearFormDGExtension *mbf)
   : MixedBilinearForm(tr_fes, te_fes, mbf)
{
   interior_face_integs = mbf->interior_face_integs;
   boundary_face_integs = mbf->boundary_face_integs;
   boundary_face_integs_marker = mbf->boundary_face_integs_marker;
};

void MixedBilinearFormDGExtension::AddInteriorFaceIntegrator(MixedBilinearFormFaceIntegrator * bfi)
{
   interior_face_integs.Append(bfi);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi)
{
   boundary_face_integs.Append(bfi);
   // NULL marker means apply everywhere
   boundary_face_integs_marker.Append(NULL);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(MixedBilinearFormFaceIntegrator *bfi,
                                                         Array<int> &bdr_marker)
{
   boundary_face_integs.Append(bfi);
   boundary_face_integs_marker.Append(&bdr_marker);
}

MixedBilinearFormDGExtension::~MixedBilinearFormDGExtension()
{
   if (!extern_bfs)
   {
      int i;
      for (i = 0; i < interior_face_integs.Size(); i++) { delete interior_face_integs[i]; }
      for (i = 0; i < boundary_face_integs.Size(); i++)
      { delete boundary_face_integs[i]; }
   }
}

void MixedBilinearFormDGExtension::Assemble(int skip_zeros)
{
   MixedBilinearForm::Assemble(skip_zeros);
   if (ext)
      return;

   Mesh *mesh = test_fes -> GetMesh();

   assert(mat != NULL);

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Array<int> trial_vdofs2, test_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         // ftr = mesh->GetFaceElementTransformations(i);
         // trial_fes->GetFaceVDofs(i, trial_vdofs);
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            trial_fes->GetElementVDofs(tr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(tr->Elem1No, test_vdofs);
            
            trial_fes->GetElementVDofs(tr->Elem2No, trial_vdofs2);
            test_fes->GetElementVDofs(tr->Elem2No, test_vdofs2);
            trial_vdofs.Append(trial_vdofs2);
            test_vdofs.Append(test_vdofs2);

            trial_fe1 = trial_fes->GetFE(tr->Elem1No);
            test_fe1 = test_fes->GetFE(tr->Elem1No);
            trial_fe2 = trial_fes->GetFE(tr->Elem2No);
            test_fe2 = test_fes->GetFE(tr->Elem2No);

            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *trial_fe2, *test_fe1,
                                                            *test_fe2, *tr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Array<int> trial_vdofs2, test_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary trace face"
                     "integrator #" << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < trial_fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr)
         {
            trial_fes->GetElementVDofs(tr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(tr->Elem1No, test_vdofs);
            trial_fe1 = trial_fes->GetFE(tr->Elem1No);
            test_fe1 = test_fes->GetFE(tr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            trial_fe2 = trial_fe1;
            test_fe2 = test_fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0)
               { continue; }

               boundary_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *trial_fe2,
                                                            *test_fe1, *test_fe2,
                                                            *tr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }
}

void DGNormalFluxIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                                const FiniteElement &trial_fe2,
                                                const FiniteElement &test_fe1,
                                                const FiniteElement &test_fe2,
                                                FaceElementTransformations &Trans,
                                                DenseMatrix &elmat)
{
   dim = trial_fe1.GetDim();
   trial_dof1 = trial_fe1.GetDof();
   trial_vdof1 = dim * trial_dof1;
   test_dof1 = test_fe1.GetDof();

   nor.SetSize(dim);
   wnor.SetSize(dim);

   // vshape1.SetSize(trial_dof1, dim);
   // vshape1_n.SetSize(trial_dof1);
   trshape1.SetSize(trial_dof1);
   shape1.SetSize(test_dof1);

   if (Trans.Elem2No >= 0)
   {
      trial_dof2 = trial_fe2.GetDof();
      trial_vdof2 = dim * trial_dof2;
      test_dof2 = test_fe2.GetDof();

      // vshape2.SetSize(trial_dof2, dim);
      // vshape2_n.SetSize(trial_dof2);
      trshape2.SetSize(trial_dof2);
      shape2.SetSize(test_dof2);
   }
   else
   {
      trial_dof2 = 0;
      test_dof2 = 0;
   }

   elmat.SetSize((test_dof1 + test_dof2), (trial_vdof1 + trial_vdof2));
   elmat = 0.0;

   // TODO: need to revisit this part for proper convergence rate.
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  max(trial_fe1.GetOrder(), trial_fe2.GetOrder()) +
                  max(test_fe1.GetOrder(), test_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + trial_fe1.GetOrder() + test_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }  // if (ir == NULL)

   for (p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      // trial_fe1.CalcVShape(eip1, vshape1);
      trial_fe1.CalcShape(eip1, trshape1);
      test_fe1.CalcShape(eip1, shape1);
      // vshape1.Mult(nor, vshape1_n);

      w = ip.weight;
      if (trial_dof2)
         w *= 0.5;

      wnor.Set(w, nor);
      
      for (jm = 0, j = 0; jm < dim; jm++)
      {
         wn = wnor(jm);
         for (jdof = 0; jdof < trial_dof1; jdof++, j++)
            for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
               elmat(i, j) += wn * shape1(idof) * trshape1(jdof);
      }

      if (trial_dof2)
      {
         // trial_fe2.CalcVShape(eip2, vshape2);
         trial_fe2.CalcShape(eip2, trshape2);
         test_fe2.CalcShape(eip2, shape2);
         // vshape2.Mult(nor, vshape2_n);

         for (jm = 0, j = 0; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof1; jdof++, j++)
               for (idof = 0, i = test_dof1; idof < test_dof2; idof++, i++)
                  elmat(i, j) += wn * shape2(idof) * trshape1(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof1; j++)
         //    {
         //       elmat(test_dof1+i, j) += w * shape2(i) * vshape1_n(j);
         //    }

         for (jm = 0, j = trial_vdof1; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = test_dof1; idof < test_dof2; idof++, i++)
                  elmat(i, j) -= wn * shape2(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof2; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(test_dof1+i, trial_dof1+j) -= w * shape2(i) * vshape2_n(j);
         //    }

         for (jm = 0, j = trial_vdof1; jm < dim; jm++)
         {
            wn = wnor(jm);
            for (jdof = 0; jdof < trial_dof2; jdof++, j++)
               for (idof = 0, i = 0; idof < test_dof1; idof++, i++)
                  elmat(i, j) -= wn * shape1(idof) * trshape2(jdof);
         }
         // for (int i = 0; i < test_dof1; i++)
         //    for (int j = 0; j < trial_dof2; j++)
         //    {
         //       elmat(i, trial_dof1+j) -= w * shape1(i) * vshape2_n(j);
         //    }
      }  // if (trial_dof2)
   }  // for (p = 0; p < ir->GetNPoints(); p++)
}

void DGVectorDiffusionIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
#ifdef MFEM_THREAD_SAFE
   // For descriptions of these variables, see the class declaration.
   Vector shape1, shape2;
   DenseMatrix dshape1, dshape2;
   DenseMatrix adjJ;
   DenseMatrix dshape1_ps, dshape2_ps;
   Vector nor;
   // Vector nL1, nL2;
   Vector nM1, nM2;
   Vector dshape1_dnM, dshape2_dnM;
   DenseMatrix jmat;
#endif

   const int dim = el1.GetDim();
   const int ndofs1 = el1.GetDof();
   const int ndofs2 = (Trans.Elem2No >= 0) ? el2.GetDof() : 0;
   const int nvdofs = dim*(ndofs1 + ndofs2);

   // Initially 'elmat' corresponds to the term:
   //    < { mu grad(u) . n }, [v] >
   // But eventually, it's going to be replaced by:
   //    elmat := -elmat + alpha*elmat^T + jmat
   elmat.SetSize(nvdofs);
   elmat = 0.;

   const bool kappa_is_nonzero = (kappa != 0.0);
   if (kappa_is_nonzero)
   {
      jmat.SetSize(nvdofs);
      jmat = 0.;
   }

   adjJ.SetSize(dim);
   shape1.SetSize(ndofs1);
   dshape1.SetSize(ndofs1, dim);
   dshape1_ps.SetSize(ndofs1, dim);
   nor.SetSize(dim);
   // nL1.SetSize(dim);
   nM1.SetSize(dim);
   dshape1_dnM.SetSize(ndofs1);

   if (ndofs2)
   {
      shape2.SetSize(ndofs2);
      dshape2.SetSize(ndofs2, dim);
      dshape2_ps.SetSize(ndofs2, dim);
      // nL2.SetSize(dim);
      nM2.SetSize(dim);
      dshape2_dnM.SetSize(ndofs2);
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * max(el1.GetOrder(), ndofs2 ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   for (int pind = 0; pind < ir->GetNPoints(); ++pind)
   {
      const IntegrationPoint &ip = ir->IntPoint(pind);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el1.CalcDShape(eip1, dshape1);

      CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      Mult(dshape1, adjJ, dshape1_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      double w, wLM;
      if (ndofs2)
      {
         el2.CalcShape(eip2, shape2);
         el2.CalcDShape(eip2, dshape2);
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
         Mult(dshape2, adjJ, dshape2_ps);

         w = ip.weight/2;
         const double w2 = w / Trans.Elem2->Weight();
         // const double wL2 = w2 * lambda->Eval(*Trans.Elem2, eip2);
         const double wM2 = w2 * mu->Eval(*Trans.Elem2, eip2);
         // nL2.Set(wL2, nor);
         nM2.Set(wM2, nor);
         // wLM = (wL2 + 2.0*wM2);
         wLM = wM2;
         dshape2_ps.Mult(nM2, dshape2_dnM);
      }
      else
      {
         w = ip.weight;
         wLM = 0.0;
      }

      {
         const double w1 = w / Trans.Elem1->Weight();
         // const double wL1 = w1 * lambda->Eval(*Trans.Elem1, eip1);
         const double wM1 = w1 * mu->Eval(*Trans.Elem1, eip1);
         // nL1.Set(wL1, nor);
         nM1.Set(wM1, nor);
         // wLM += (wL1 + 2.0*wM1);
         wLM += wM1;
         dshape1_ps.Mult(nM1, dshape1_dnM);
      }

      const double jmatcoef = kappa * (nor*nor) * wLM;

      // (1,1) block
      AssembleBlock(
         dim, ndofs1, ndofs1, 0, 0, jmatcoef,
         shape1, shape1, dshape1_dnM, elmat, jmat);

      if (ndofs2 == 0) { continue; }

      // In both elmat and jmat, shape2 appears only with a minus sign.
      shape2.Neg();

      // (1,2) block
      AssembleBlock(
         dim, ndofs1, ndofs2, 0, dim*ndofs1, jmatcoef,
         shape1, shape2, dshape2_dnM, elmat, jmat);
      // (2,1) block
      AssembleBlock(
         dim, ndofs2, ndofs1, dim*ndofs1, 0, jmatcoef,
         shape2, shape1, dshape1_dnM, elmat, jmat);
      // (2,2) block
      AssembleBlock(
         dim, ndofs2, ndofs2, dim*ndofs1, dim*ndofs1, jmatcoef,
         shape2, shape2, dshape2_dnM, elmat, jmat);
   }

   // elmat := -elmat + alpha*elmat^t + jmat
   if (kappa_is_nonzero)
   {
      for (int i = 0; i < nvdofs; ++i)
      {
         for (int j = 0; j < i; ++j)
         {
            double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
            elmat(i,j) = alpha*aji - aij + mij;
            elmat(j,i) = alpha*aij - aji + mij;
         }
         elmat(i,i) = (alpha - 1.)*elmat(i,i) + jmat(i,i);
      }
   }
   else
   {
      for (int i = 0; i < nvdofs; ++i)
      {
         for (int j = 0; j < i; ++j)
         {
            double aij = elmat(i,j), aji = elmat(j,i);
            elmat(i,j) = alpha*aji - aij;
            elmat(j,i) = alpha*aij - aji;
         }
         elmat(i,i) *= (alpha - 1.);
      }
   }
}

// static method
void DGVectorDiffusionIntegrator::AssembleBlock(
   const int dim, const int row_ndofs, const int col_ndofs,
   const int row_offset, const int col_offset, const double jmatcoef,
   const Vector &row_shape, const Vector &col_shape, const Vector &col_dshape_dnM,
   DenseMatrix &elmat, DenseMatrix &jmat)
{
   for (int d = 0; d < dim; ++d)
   {
      int j = col_offset + d * col_ndofs;
      for (int jdof = 0; jdof < col_ndofs; ++jdof, ++j)
      {
         int i = row_offset + d * row_ndofs;
         const double t2 = col_dshape_dnM(jdof);
         for (int idof = 0; idof < row_ndofs; ++idof, ++i)
            elmat(i, j) += row_shape(idof) * t2;
      }
   }

   if (jmatcoef == 0.0) { return; }

   for (int d = 0; d < dim; ++d)
   {
      const int jo = col_offset + d*col_ndofs;
      const int io = row_offset + d*row_ndofs;
      for (int jdof = 0, j = jo; jdof < col_ndofs; ++jdof, ++j)
      {
         const double sj = jmatcoef * col_shape(jdof);
         for (int i = max(io,j), idof = i - io; idof < row_ndofs; ++idof, ++i)
         {
            jmat(i, j) += row_shape(idof) * sj;
         }
      }
   }
}

void DGVectorDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

   const int dim = el.GetDim();
   const int ndofs = el.GetDof();
   const int nvdofs = dim*ndofs;

   elvect.SetSize(nvdofs);
   elvect = 0.0;

   adjJ.SetSize(dim);
   shape.SetSize(ndofs);
   dshape.SetSize(ndofs, dim);
   dshape_ps.SetSize(ndofs, dim);
   nor.SetSize(dim);
   dshape_dn.SetSize(ndofs);
   dshape_du.SetSize(ndofs);
   u_dir.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*el.GetOrder(); // <-----
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   for (int pi = 0; pi < ir->GetNPoints(); ++pi)
   {
      const IntegrationPoint &ip = ir->IntPoint(pi);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      // Evaluate the Dirichlet b.c. using the face transformation.
      uD.Eval(u_dir, Tr, ip);

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      Mult(dshape, adjJ, dshape_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }

      double wL, wM, jcoef;
      {
         const double w = ip.weight / Tr.Elem1->Weight();
         // wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         // jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         jcoef = kappa * wM * (nor*nor);
         dshape_ps.Mult(nor, dshape_dn);
         dshape_ps.Mult(u_dir, dshape_du);
      }

      // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      //   + kappa < h^{-1} (lambda + 2 mu) uD, v >

      // i = idof + ndofs * im
      // v_phi(i,d) = delta(im,d) phi(idof)
      // div(v_phi(i)) = dphi(idof,im)
      // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
      //
      // term 1:
      //   alpha < uD, lambda div(v_phi(i)) n >
      //   alpha lambda div(v_phi(i)) (uD.n) =
      //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
      //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
      //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
      // term 2:
      //   < alpha uD, mu grad(v_phi(i)).n > =
      //   alpha mu uD^T grad(v_phi(i)) n =
      //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
      //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
      //   alpha * wM * u_dir(im) * dshape_dn(idof)
      // term 3:
      //   < alpha uD, mu (grad(v_phi(i)))^T n > =
      //   alpha mu n^T grad(v_phi(i)) uD =
      //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
      //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
      //   alpha * wM * nor(im) * dshape_du(idof)
      // term j:
      //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
      //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
      //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
      //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
      //      [ 1/h = |nor|/det(J1) ]
      //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
      //   jcoef * u_dir(im) * shape(idof)

      wM *= alpha;
      // const double t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const double t2 = wM * u_dir(im);
         // const double t3 = wM * nor(im);
         const double tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t2*dshape_dn(idof) + tj*shape(idof));
         }
      }
   }
}

void DGBoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGBoundaryNormalLFIntegrator::AssembleRHSElementVect");
}

void DGBoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim = el.GetDim();
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2 * el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      Q.Eval(Qvec, *Tr.Elem1, eip);

      el.CalcShape(eip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}