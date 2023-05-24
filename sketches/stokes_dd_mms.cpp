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
#include "dg_mixed_bilin.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"
#include "interfaceinteg.hpp"
#include "topology_handler.hpp"

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
void fvec_natural(const Vector & x, Vector & y);
void dudx_ex(const Vector & x, Vector & y);

// Copied MultiBlockSolver::AssembleInterfaceMatrix, only for the sketch purpose.
void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
                              FiniteElementSpace *fes1,
                              FiniteElementSpace *fes2,
                              TopologyHandler *topol_handler,
                              InterfaceNonlinearFormIntegrator *interface_integ,
                              Array<InterfaceInfo> *interface_infos,
                              Array2D<SparseMatrix*> &mats);

void AssembleInterfaceMatrix(
   Mesh *mesh1, Mesh *mesh2, FiniteElementSpace *trial_fes1, 
   FiniteElementSpace *trial_fes2, FiniteElementSpace *test_fes1,
   FiniteElementSpace *test_fes2, TopologyHandler *topol_handler,
   InterfaceNonlinearFormIntegrator *interface_integ,
   Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats);

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
   bool pres_dbc = false;
   bool use_dg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&refine, "-r", "--refine",
                  "Number of refinements.");
   args.AddOption(&pres_dbc, "-pd", "--pdirichlet", "-no-pd", "--no-pdirichlet",
                  "Use pressure dirichlet condition.");
   args.AddOption(&use_dg, "-dg", "--use-dg", "-no-dg", "--no-use-dg",
                  "Use discontinuous Galerkin scheme.");
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
   Mesh *mesh = new Mesh(mesh_file);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   for (int l = 0; l < refine; l++)
   {
      mesh->UniformRefinement();
   }

   TopologyData topol_data;
   SubMeshTopologyHandler *topol_handler = new SubMeshTopologyHandler(mesh);
   Array<Mesh*> meshes;
   topol_handler->ExportInfo(meshes, topol_data);
   
   // Receive topology info
   const int numSub = topol_data.numSub;
   const int dim = topol_data.dim;
   Array<int> global_bdr_attributes = *(topol_data.global_bdr_attributes);

   // number of variable and dimension of each variable.
   const int num_var = 2;  // u + p
   Array<int> vdim(num_var);
   vdim[0] = dim;
   vdim[1] = 1;

   double sigma = -1.0;
   // DG terms are employed for velocity space, which is order+1. resulting kappa becomes (order+2)^2.
   double kappa = (order + 2) * (order + 2);

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *dg_coll(new DG_FECollection(order+1, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *pdg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *ph1_coll(new H1_FECollection(order, dim));

   // This needs to be Array for the multi-component case.
   Array<FiniteElementSpace *> ufes, pfes;
   ufes.SetSize(numSub);
   pfes.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
      if (use_dg)
      {
         ufes[m] = new FiniteElementSpace(meshes[m], dg_coll, vdim[0]);
         pfes[m] = new FiniteElementSpace(meshes[m], pdg_coll, vdim[1]);
      }
      else
      {
         ufes[m] = new FiniteElementSpace(meshes[m], h1_coll, vdim[0]);
         pfes[m] = new FiniteElementSpace(meshes[m], ph1_coll, vdim[1]);
      }

   Array<int> var_offsets(num_var + 1); // need one more index.
   Array<int> domain_offsets(num_var * numSub + 1); // need one more index.
   Array<int> block_offsets(vdim.Sum() * numSub + 1); // do we need this?
   Array<int> num_vdofs(num_var * numSub);
   block_offsets[0] = 0;
   domain_offsets[0] = 0;
   var_offsets = 0;
   for (int v = 0, block_idx = 1, domain_idx=1; v < num_var; v++)
      for (int m = 0; m < numSub; m++, domain_idx++)
      {
         FiniteElementSpace *fes = (v == 0) ? ufes[m] : pfes[m];
         domain_offsets[domain_idx] = fes->GetVSize();
         for (int d = 0; d < vdim[v]; d++, block_idx++)
         {
            block_offsets[block_idx] = fes->GetNDofs();
         }
         var_offsets[v+1] += fes->GetVSize();
      }
   block_offsets.PartialSum();
   domain_offsets.GetSubArray(1, num_var * numSub, num_vdofs);
   domain_offsets.PartialSum();
   var_offsets.PartialSum();

   Array<int> u_offsets, p_offsets, u_vdofs, p_vdofs;
   domain_offsets.GetSubArray(0, numSub+1, u_offsets);
   domain_offsets.GetSubArray(numSub, numSub+1, p_offsets);
   num_vdofs.GetSubArray(0, numSub, u_vdofs);
   num_vdofs.GetSubArray(numSub, numSub, p_vdofs);
   int tmp = p_offsets[0];
   for (int k = 0; k < p_offsets.Size(); k++) p_offsets[k] -= tmp;

   printf("block_offsets\n");
   std::cout << "***********************************************************\n";
   for (int d = 1; d < block_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, block_offsets[d] - block_offsets[d-1]);
   printf("dim(q) = %d\n", block_offsets.Last());
   std::cout << "***********************************************************\n";

   printf("domain_offsets\n");
   std::cout << "***********************************************************\n";
   for (int d = 1; d < domain_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, domain_offsets[d] - domain_offsets[d-1]);
   printf("dim(q) = %d\n", domain_offsets.Last());
   std::cout << "***********************************************************\n";

   printf("num_vdofs\n");
   std::cout << "***********************************************************\n";
   for (int d = 0; d < num_vdofs.Size(); d++)
      printf("dim(%d) = %d\n", d, num_vdofs[d]);
   std::cout << "***********************************************************\n";

   printf("var_offsets\n");
   std::cout << "***********************************************************\n";
   for (int d = 1; d < var_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, var_offsets[d] - var_offsets[d-1]);
   std::cout << "***********************************************************\n";

   printf("u_offsets\n");
   std::cout << "***********************************************************\n";
   for (int d = 1; d < u_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, u_offsets[d] - u_offsets[d-1]);
   std::cout << "***********************************************************\n";

   printf("p_offsets\n");
   std::cout << "***********************************************************\n";
   for (int d = 1; d < p_offsets.Size(); d++)
      printf("dim(%d) = %d\n", d, p_offsets[d] - p_offsets[d-1]);
   std::cout << "***********************************************************\n";

   printf("u_vdofs\n");
   std::cout << "***********************************************************\n";
   for (int d = 0; d < u_vdofs.Size(); d++)
      printf("dim(%d) = %d\n", d, u_vdofs[d]);
   std::cout << "***********************************************************\n";

   printf("p_vdofs\n");
   std::cout << "***********************************************************\n";
   for (int d = 0; d < p_vdofs.Size(); d++)
      printf("dim(%d) = %d\n", d, p_vdofs[d]);
   std::cout << "***********************************************************\n";

   int max_bdr_attr = -1;
   for (int m = 0; m < numSub; m++)
   {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
   }

   Array<int> u_ess_attr(max_bdr_attr);
   Array<int> p_ess_attr(max_bdr_attr);
   // this array of integer essentially acts as the array of boolean:
   // If value is 0, then it is not Dirichlet.
   // If value is 1, then it is Dirichlet.
   u_ess_attr = 0;
   p_ess_attr = 0;
   for (int k = 0; k < global_bdr_attributes.Size(); k++)
   {
      int bdr_attr = global_bdr_attributes[k];
      if (pres_dbc && (bdr_attr == 2))
         p_ess_attr[bdr_attr-1] = 1;
      else
         u_ess_attr[bdr_attr-1] = 1;
   }

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0), minus_one(-1.0), one(1.0), zero(0.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient dudxcoeff(dim, dudx_ex);
   VectorFunctionCoefficient minus_fcoeff(dim, fFun, &minus_one);
   // VectorFunctionCoefficient grad_pcoeff(dim, grad_pFun_ex);
   VectorFunctionCoefficient mlap_ucoeff(dim, mlap_uFun_ex);
   // FunctionCoefficient mlap_pcoeff(mlap_pFun_ex);
   FunctionCoefficient fnatcoeff(f_natural);
   VectorFunctionCoefficient fvecnatcoeff(dim*dim, fvec_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);
   Array<GridFunction *> u_ex(numSub), p_ex(numSub);
   for (int m = 0; m < numSub; m++)
   {
      u_ex[m] = new GridFunction(ufes[m]);
      p_ex[m] = new GridFunction(pfes[m]);

      u_ex[m]->ProjectCoefficient(ucoeff);
      p_ex[m]->ProjectCoefficient(pcoeff);
   }

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
//    MemoryType mt = device.GetMemoryType();
   BlockVector x(domain_offsets), rhs(domain_offsets);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   Array<GridFunction *> u(numSub), p(numSub);
   double p_const = 0.0;
   int ps = 0;
   for (int m = 0, pidx = numSub; m < numSub; m++, pidx++)
   {
      u[m] = new GridFunction;
      p[m] = new GridFunction;

      u[m]->MakeRef(ufes[m], x.GetBlock(m), 0);
      p[m]->MakeRef(pfes[m], x.GetBlock(pidx), 0);

      (*u[m]) = 0.0;
      p[m]->ProjectCoefficient(pcoeff);
      ps += p[m]->Size();
      p_const += p[m]->Sum();
      (*p[m]) = 0.0;
   }
   p_const /= static_cast<double>(ps);

   Array<LinearForm *> fform(numSub), gform(numSub);
   for (int m = 0, pidx = numSub; m < numSub; m++, pidx++)
   {
      fform[m] = new LinearForm;
      fform[m]->Update(ufes[m], rhs.GetBlock(m), 0);
      fform[m]->AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff));
      // fform[m]->AddDomainIntegrator(new VectorDomainLFIntegrator(mlap_ucoeff));

      // // Currently, mfem does not have a way to impose general tensor bc.
      // // dg fe space does not support boundary integrators. needs reimplmentation.
      // fform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff), p_ess_attr);
      // fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(dudxcoeff), p_ess_attr);
      if (use_dg)
         fform[m]->AddBdrFaceIntegrator(new BoundaryNormalStressLFIntegrator(fvecnatcoeff), p_ess_attr);
      else  // BdrFaceIntegrator also works for non-dg case. just to show BoundaryIntegrator is available.
         fform[m]->AddBoundaryIntegrator(new BoundaryNormalStressLFIntegrator(fvecnatcoeff), p_ess_attr);
      
      fform[m]->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(ucoeff, k, sigma, kappa), u_ess_attr);

      fform[m]->Assemble();
      fform[m]->SyncAliasMemory(rhs);

      gform[m] = new LinearForm;
      gform[m]->Update(pfes[m], rhs.GetBlock(pidx), 0);
      gform[m]->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
      // dg fe space does not support boundary integrators. needs reimplmentation.
      // Below two operators are essentially the same. Integration order must be set as 2 * order to guarantee the right convergence rate.
      // gform->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(ucoeff, 2, 0), u_ess_attr);
      gform[m]->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(ucoeff), u_ess_attr);
      gform[m]->Assemble();
      gform[m]->SyncAliasMemory(rhs);
   }

   // 9. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k \grad u_h \cdot \grad v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   Array<BilinearForm *> mVarf(numSub);
   Array<MixedBilinearFormDGExtension *> bVarf(numSub);

   for (int m = 0; m < numSub; m++)
   {
      mVarf[m] = new BilinearForm(ufes[m]);
      bVarf[m] = new MixedBilinearFormDGExtension(ufes[m], pfes[m]);

      mVarf[m]->AddDomainIntegrator(new VectorDiffusionIntegrator(k));
      if (use_dg)
         mVarf[m]->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(k, sigma, kappa));
      mVarf[m]->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(k, sigma, kappa), u_ess_attr);
      mVarf[m]->Assemble();
      // mVarf[m]->Finalize();

      bVarf[m]->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
      if (use_dg)
         bVarf[m]->AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);
      bVarf[m]->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, u_ess_attr);
      bVarf[m]->Assemble();
      // bVarf[m]->Finalize();
   }

   BlockMatrix mMat(u_offsets), bMat(p_offsets, u_offsets);
   Array2D<SparseMatrix *> m_mats(numSub, numSub), b_mats(numSub, numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
      {
         if (i == j)
         {
            m_mats(i, j) = &(mVarf[i]->SpMat());
            b_mats(i, j) = &(bVarf[i]->SpMat());
         }
         else
         {
            m_mats(i, j) = new SparseMatrix(u_vdofs[i], u_vdofs[j]);
            b_mats(i, j) = new SparseMatrix(p_vdofs[i], u_vdofs[j]);
         }
      }

   InterfaceNonlinearFormIntegrator *vector_diff = new InterfaceDGVectorDiffusionIntegrator(k, sigma, kappa);
   InterfaceNonlinearFormIntegrator *normal_flux = new InterfaceDGNormalFluxIntegrator;
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);

      Array<int> midx(2);
      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;
      Array2D<SparseMatrix *> m_mats_p(2,2), b_mats_p(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            m_mats_p(i, j) = m_mats(midx[i], midx[j]);
            b_mats_p(i, j) = b_mats(midx[i], midx[j]);
         }

      Mesh *mesh1, *mesh2;
      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      FiniteElementSpace *ufes1, *ufes2, *pfes1, *pfes2;
      ufes1 = ufes[midx[0]];
      ufes2 = ufes[midx[1]];
      pfes1 = pfes[midx[0]];
      pfes2 = pfes[midx[1]];

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);
      AssembleInterfaceMatrix(mesh1, mesh2, ufes1, ufes2, topol_handler, vector_diff, interface_infos, m_mats_p);
      AssembleInterfaceMatrix(mesh1, mesh2, ufes1, ufes2, pfes1, pfes2, topol_handler, normal_flux, interface_infos, b_mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)

   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
      {
         m_mats(i, j)->Finalize();
         b_mats(i, j)->Finalize();
         mMat.SetBlock(i, j, m_mats(i, j));
         bMat.SetBlock(i, j, b_mats(i, j));
      }

   SparseMatrix *M = mMat.CreateMonolithic();
   SparseMatrix *B = bMat.CreateMonolithic();

   Vector u_view, urhs_view;
   u_view.MakeRef(x, 0, var_offsets[1] - var_offsets[0]);
   urhs_view.MakeRef(rhs, 0, var_offsets[1] - var_offsets[0]);
   Vector R1(u_view.Size());
   R1 = 0.0;

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
   solver.SetOperator(*M);
   // solver.SetPreconditioner(darcyPrec);
   solver.SetPrintLevel(0);
   // x = 0.0;
   solver.Mult(urhs_view, R1);
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

//    double err_u = 0.0, norm_u = 0.0;
//    for (int m = 0; m < numSub; m++)
//    {
//       err_u += u[m]->ComputeL2Error(ucoeff, irs);
//       norm_u += ComputeLpNorm(2., ucoeff, *(meshes[m]), irs);
//    }

//    printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
// }

   // B * A^{-1} * F1 - G1
   Vector p_view, prhs_view;
   p_view.MakeRef(x, var_offsets[1], var_offsets[2] - var_offsets[1]);
   prhs_view.MakeRef(rhs, var_offsets[1], var_offsets[2] - var_offsets[1]);

   Vector R2(p_view.Size());
   B->Mult(R1, R2);
   R2 -= prhs_view;

   SchurOperator schur(M, B);
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
      solver2.Mult(R2, p_view);
   else
      ortho.Mult(R2, p_view);
   printf("Pressure is solved.\n");

//    // AU = F - B^T * P;
//    Vector F3(ufes->GetVSize());
//    F3 = 0.0;
//    bVarf->MultTranspose(p, F3);
//    F3 *= -1.0;
//    F3 += (*fform);

//    printf("Solving for velocity\n");
//    solver.Mult(F3, u);
//    printf("Velocity is solved.\n");

   if (!pres_dbc)
      p_view += p_const;

   int order_quad = max(2, 2*(order+1)+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u = 0.0, norm_u = 0.0;
   double err_p = 0.0, norm_p = 0.0;
   for (int m = 0; m < numSub; m++)
   {
      // err_u += u[m]->ComputeL2Error(ucoeff, irs);
      // norm_u += ComputeLpNorm(2., ucoeff, *(meshes[m]), irs);
      err_p += p[m]->ComputeL2Error(pcoeff, irs);
      norm_p += ComputeLpNorm(2., pcoeff, *(meshes[m]), irs);
   }

   // printf("|| u_h - u_ex || / || u_ex || = %.5E\n", err_u / norm_u);
   printf("|| p_h - p_ex || / || p_ex || = %.5E\n", err_p / norm_p);

   // // 15. Save data in the ParaView format
   // for (int m = 0; m < numSub; m++)
   // {
   //    std::string filename = "stokes_mms_paraview" + std::to_string(m);
   //    ParaViewDataCollection paraview_dc(filename.c_str(), meshes[m]);
   //    // paraview_dc.SetPrefixPath("ParaView");
   //    paraview_dc.SetLevelsOfDetail(order+1);
   //    // paraview_dc.SetCycle(0);
   // //    paraview_dc.SetDataFormat(VTKFormat::BINARY);
   // //    paraview_dc.SetHighOrderOutput(true);
   // //    paraview_dc.SetTime(0.0); // set the time
   //    // paraview_dc.RegisterField("velocity", u[m]);
   //    // paraview_dc.RegisterField("u_exact",&u_ex);
   //    // paraview_dc.RegisterField("pressure",&p);
   //    // paraview_dc.RegisterField("p_exact",&p_ex);
   //    paraview_dc.Save();
   // }

   // 17. Free the used memory.
   delete vector_diff;
   delete normal_flux;
   for (int m = 0; m < numSub; m++)
   {
      delete fform[m];
      delete gform[m];
      delete mVarf[m];
      delete bVarf[m];
      delete u[m];
      delete p[m];
      delete u_ex[m];
      delete p_ex[m];
      delete ufes[m];
      delete pfes[m];
   }
   // delete qfes;
   delete dg_coll;
   delete h1_coll;
   delete pdg_coll;
   delete ph1_coll;
   delete l2_coll;
   delete topol_handler;
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

   y(0) = - sin(xi)*sin(yi);
   y(1) = - cos(xi)*cos(yi);
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

   y(0) -= pFun_ex(x);
   y(3) -= pFun_ex(x);
}

void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
                              FiniteElementSpace *fes1,
                              FiniteElementSpace *fes2,
                              TopologyHandler *topol_handler,
                              InterfaceNonlinearFormIntegrator *interface_integ,
                              Array<InterfaceInfo> *interface_infos,
                              Array2D<SparseMatrix*> &mats)
{
   const int skip_zeros = 0;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);
      
      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *fe1, *fe2;
      Array<Array<int> *> vdofs(2);
      vdofs[0] = new Array<int>;
      vdofs[1] = new Array<int>;

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
         fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         fe1 = fes1->GetFE(tr1->Elem1No);
         fe2 = fes2->GetFE(tr2->Elem1No);

         interface_integ->AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);

         for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
               mats(i, j)->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}

void AssembleInterfaceMatrix(
   Mesh *mesh1, Mesh *mesh2, FiniteElementSpace *trial_fes1, 
   FiniteElementSpace *trial_fes2, FiniteElementSpace *test_fes1,
   FiniteElementSpace *test_fes2, TopologyHandler *topol_handler,
   InterfaceNonlinearFormIntegrator *interface_integ,
   Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats)
{
   const int skip_zeros = 0;

   for (int bn = 0; bn < interface_infos->Size(); bn++)
   {
      InterfaceInfo *if_info = &((*interface_infos)[bn]);
      
      Array2D<DenseMatrix*> elemmats;
      FaceElementTransformations *tr1, *tr2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;
      Array<Array<int> *> test_vdofs(2), trial_vdofs(2);
      trial_vdofs[0] = new Array<int>;
      trial_vdofs[1] = new Array<int>;
      test_vdofs[0] = new Array<int>;
      test_vdofs[1] = new Array<int>;

      topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

      if ((tr1 != NULL) && (tr2 != NULL))
      {
         trial_fes1->GetElementVDofs(tr1->Elem1No, *trial_vdofs[0]);
         trial_fes2->GetElementVDofs(tr2->Elem1No, *trial_vdofs[1]);
         test_fes1->GetElementVDofs(tr1->Elem1No, *test_vdofs[0]);
         test_fes2->GetElementVDofs(tr2->Elem1No, *test_vdofs[1]);
         // Both domains will have the adjacent element as Elem1.
         trial_fe1 = trial_fes1->GetFE(tr1->Elem1No);
         trial_fe2 = trial_fes2->GetFE(tr2->Elem1No);
         test_fe1 = test_fes1->GetFE(tr1->Elem1No);
         test_fe2 = test_fes2->GetFE(tr2->Elem1No);

         interface_integ->AssembleInterfaceMatrix(
            *trial_fe1, *trial_fe2, *test_fe1, *test_fe2, *tr1, *tr2, elemmats);

         for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
               mats(i, j)->AddSubMatrix(*test_vdofs[i], *trial_vdofs[j], *elemmats(i,j), skip_zeros);
            }
         }
      }  // if ((tr1 != NULL) && (tr2 != NULL))
   }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
}