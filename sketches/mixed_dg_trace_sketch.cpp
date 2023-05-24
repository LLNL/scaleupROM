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

using namespace std;
using namespace mfem;

/// Abstract base class BilinearFormIntegrator
class BilinearFormIntegratorExtension : public BilinearFormIntegrator
{
protected:
   BilinearFormIntegratorExtension(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { }

public:
   /** Abstract method used for assembling InteriorFaceIntegrators in a
       MixedBilinearFormDGExtension. */
   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat)
   { mfem_error("Abstract method BilinearFormIntegratorExtension::AssembleFaceMatrix!\n"); }

};

class MixedBilinearFormDGExtension : public MixedBilinearForm
{
protected:
   /// interface integrators.
   Array<BilinearFormIntegratorExtension*> interior_face_integs;

   /// Set of boundary face Integrators to be applied.
   Array<BilinearFormIntegratorExtension*> boundary_face_integs;
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
   void AddInteriorFaceIntegrator(BilinearFormIntegratorExtension *bfi);

   /// Adds new boundary Face Integrator. Assumes ownership of @a bfi.
   void AddBdrFaceIntegrator(BilinearFormIntegratorExtension *bfi);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes.

       Assumes ownership of @a bfi. The array @a bdr_marker is stored internally
       as a pointer to the given Array<int> object. */
   void AddBdrFaceIntegrator(BilinearFormIntegratorExtension *bfi,
                             Array<int> &bdr_marker);

   /// Access all integrators added with AddInteriorFaceIntegrator().
   Array<BilinearFormIntegratorExtension*> *GetFBFI() { return &interior_face_integs; }

   /// Access all integrators added with AddBdrFaceIntegrator().
   Array<BilinearFormIntegratorExtension*> *GetBFBFI() { return &boundary_face_integs; }
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

class MixedDGTraceIntegrator : public BilinearFormIntegratorExtension//, DGTraceIntegrator
{
private:
   int dim;

   Vector shape1, shape2;
   Vector divshape;
   DenseMatrix vshape1, vshape2;
   Vector vshape1_n, vshape2_n;

public:
   MixedDGTraceIntegrator() {};
   // /// Construct integrator with rho = 1, b = 0.5*a.
   // MixedDGTraceIntegrator(VectorCoefficient &u_, double a)
   //    : DGTraceIntegrator(u_, a) {};

   // /// Construct integrator with rho = 1.
   // MixedDGTraceIntegrator(VectorCoefficient &u_, double a, double b)
   //    : DGTraceIntegrator(u_, a, b) {};

   // MixedDGTraceIntegrator(Coefficient &rho_, VectorCoefficient &u_,
   //                         double a, double b)
   //    : DGTraceIntegrator(rho_, u_, a, b) {};

   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");

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

   // // 4. Refine the mesh to increase the resolution. In this example we do
   // //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   // //    largest number that gives a final mesh with no more than 10,000
   // //    elements.
   // {
   //    int ref_levels =
   //       (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
   //    for (int l = 0; l < ref_levels; l++)
   //    {
   //       mesh->UniformRefinement();
   //    }
   // }

   // 5. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, dg_coll, dim);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, dg_coll);

   // 7. Define the coefficients, analytical solution, and rhs of the PDE.
   const double nu = 1.0;
   ConstantCoefficient k(nu);
   ConstantCoefficient zero(0.0);

   // VectorFunctionCoefficient fcoeff(dim, fFun);
   // FunctionCoefficient fnatcoeff(f_natural);
   // FunctionCoefficient gcoeff(gFun);

   // VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   // FunctionCoefficient pcoeff(pFun_ex);

   // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   // MemoryType mt = device.GetMemoryType();
   // BlockVector x(block_offsets, mt), rhs(block_offsets, mt);

   MixedBilinearFormDGExtension *bVarf(new MixedBilinearFormDGExtension(R_space, W_space));

   // bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->AddInteriorFaceIntegrator(new MixedDGTraceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();

   // 17. Free the used memory.
   delete bVarf;
   delete W_space;
   delete R_space;
   delete dg_coll;
   delete mesh;

   return 0;
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

void MixedBilinearFormDGExtension::AddInteriorFaceIntegrator(BilinearFormIntegratorExtension * bfi)
{
   interior_face_integs.Append(bfi);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(BilinearFormIntegratorExtension *bfi)
{
   boundary_face_integs.Append(bfi);
   // NULL marker means apply everywhere
   boundary_face_integs_marker.Append(NULL);
}

void MixedBilinearFormDGExtension::AddBdrFaceIntegrator(BilinearFormIntegratorExtension *bfi,
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

void MixedDGTraceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                                const FiniteElement &trial_fe2,
                                                const FiniteElement &test_fe1,
                                                const FiniteElement &test_fe2,
                                                FaceElementTransformations &Trans,
                                                DenseMatrix &elmat)
{
   int trial_dof1, trial_dof2, test_dof1, test_dof2;

   double w;

   dim = trial_fe1.GetDim();
   trial_dof1 = trial_fe1.GetDof();
   test_dof1 = test_fe1.GetDof();
   Vector nor(dim);

   vshape1.SetSize(trial_dof1, dim);
   vshape1_n.SetSize(trial_dof1);
   shape1.SetSize(test_dof1);

   if (Trans.Elem2No >= 0)
   {
      trial_dof2 = trial_fe2.GetDof();
      test_dof2 = test_fe2.GetDof();

      vshape2.SetSize(trial_dof2, dim);
      vshape2_n.SetSize(trial_dof2);
      shape2.SetSize(test_dof2);
   }
   else
   {
      trial_dof2 = 0;
      test_dof2 = 0;
   }

   elmat.SetSize((test_dof1 + test_dof2), (trial_dof1 + trial_dof2));
   elmat = 0.0;

   // TODO: need to revisit this part for proper convergence rate.
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(trial_fe1.GetOrder(), trial_fe2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*trial_fe1.GetOrder();
      }
      if (trial_fe1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }  // if (ir == NULL)

   for (int p = 0; p < ir->GetNPoints(); p++)
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

      trial_fe1.CalcVShape(eip1, vshape1);
      test_fe1.CalcShape(eip1, shape1);
      vshape1.Mult(nor, vshape1_n);

      w = ip.weight;
      for (int i = 0; i < test_dof1; i++)
         for (int j = 0; j < trial_dof1; j++)
         {
            elmat(i, j) += w * shape1(i) * vshape1_n(j);
         }

      if (trial_dof2)
      {
         trial_fe2.CalcVShape(eip2, vshape2);
         test_fe2.CalcShape(eip2, shape2);
         vshape2.Mult(nor, vshape2_n);

         for (int i = 0; i < test_dof2; i++)
            for (int j = 0; j < trial_dof1; j++)
            {
               elmat(test_dof1+i, j) -= w * shape2(i) * vshape1_n(j);
            }

         for (int i = 0; i < test_dof2; i++)
            for (int j = 0; j < trial_dof2; j++)
            {
               elmat(test_dof1+i, trial_dof1+j) += w * shape2(i) * vshape2_n(j);
            }

         for (int i = 0; i < test_dof1; i++)
            for (int j = 0; j < trial_dof2; j++)
            {
               elmat(i, trial_dof1+j) -= w * shape1(i) * vshape2_n(j);
            }
      }  // if (trial_dof2)
   }  // for (int p = 0; p < ir->GetNPoints(); p++)
}