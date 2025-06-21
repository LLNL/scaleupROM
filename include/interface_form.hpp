// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_INTERFACE_FORM_HPP
#define SCALEUPROM_INTERFACE_FORM_HPP

#include "mfem.hpp"
#include "interfaceinteg.hpp"
#include "topology_handler.hpp"

namespace mfem
{

// TODO(kevin): inherits or cherry-pick ROMNonlinearForm for hyper-reduction.
class InterfaceForm
{
protected:
   mutable TimeProfiler timer;

   int numSub = -1;
   int skip_zeros = 1;

   Array<Mesh *> meshes;                  // not owned
   Array<FiniteElementSpace *> fes;       // not owned
   TopologyHandler *topol_handler = NULL; // not owned

   Array<int> block_offsets;  // Size(numSub + 1). each block corresponds to a vector solution in the FiniteElementSpace.

   /// Set of interior face Integrators to be assembled (added).
   Array<InterfaceNonlinearFormIntegrator*> fnfi; // owned

   // For Mult and GetGradient.
   mutable BlockVector x_tmp, y_tmp;

public:
   /// Construct a NonlinearForm on the given FiniteElementSpace, @a f.
   /** As an Operator, the NonlinearForm has input and output size equal to the
      number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   InterfaceForm(Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_, TopologyHandler *topol_);

   /** @brief Destroy the NonlinearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~InterfaceForm();

   /* access functions */
   const Array<int>& GetBlockOffsets() { return block_offsets; }

   /// Adds new Interior Face Integrator.
   virtual void AddInterfaceIntegrator(InterfaceNonlinearFormIntegrator *nlfi)
   {
      fnfi.Append(nlfi);
   }

   /** @brief Access all interface integrators added with
       AddInterfaceIntegrator(). */
   const Array<InterfaceNonlinearFormIntegrator*> &GetIntefaceIntegrators() const
   { return fnfi; }

   void AssembleInterfaceMatrices(Array2D<SparseMatrix *> &mats) const;

   void AssembleInterfaceMatrixAtPort(const int p, Array<FiniteElementSpace *> &fes_comp, Array2D<SparseMatrix *> &mats_p) const;

   virtual void InterfaceAddMult(const Vector &x, Vector &y) const;

   virtual void InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const;

   /*
      this is public only for the sake of testing.
      TODO(kevin): bring it back to protected.
   */
   // NonlinearForm interface operator.
   void AssembleInterfaceVector(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *fes1, FiniteElementSpace *fes2,
      Array<InterfaceInfo> *interface_infos,
      const Vector &x1, const Vector &x2,
      Vector &y1, Vector &y2) const;

protected:

   // BilinearForm interface operator.
   void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *fes1, FiniteElementSpace *fes2,
      Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats) const;

   void AssembleInterfaceGrad(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *fes1, FiniteElementSpace *fes2,
      Array<InterfaceInfo> *interface_infos,
      const Vector &x1, const Vector &x2, Array2D<SparseMatrix*> &mats) const;

};

// TODO(kevin): inherits or cherry-pick ROMNonlinearForm for hyper-reduction.
class MixedInterfaceForm
{
protected:
   int numSub = -1;
   int skip_zeros = 1;

   Array<Mesh *> meshes;   // not owned
   Array<FiniteElementSpace *> trial_fes, test_fes;   // not owned
   TopologyHandler *topol_handler = NULL; // not owned

   Array<int> trial_block_offsets;  // Size(numSub + 1). each block corresponds to a vector solution in trial FiniteElementSpace.
   Array<int> test_block_offsets;  // Size(numSub + 1). each block corresponds to a vector solution in test FiniteElementSpace.

   /// Set of interior face Integrators to be assembled (added).
   Array<InterfaceNonlinearFormIntegrator*> fnfi; // owned
   // Array<Array<SampleInfo> *> fnfi_sample;

   // For Mult and GetGradient.
   mutable BlockVector x_tmp, y_tmp;

public:
   /// Construct a NonlinearForm on the given FiniteElementSpace, @a f.
   /** As an Operator, the NonlinearForm has input and output size equal to the
      number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   MixedInterfaceForm(Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &trial_fes_, 
                      Array<FiniteElementSpace *> &test_fes_, TopologyHandler *topol_);

   /** @brief Destroy the NonlinearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~MixedInterfaceForm();

   /// Adds new Interior Face Integrator.
   void AddInterfaceIntegrator(InterfaceNonlinearFormIntegrator *nlfi)
   {
      fnfi.Append(nlfi);
      // fnfi_sample.Append(NULL);
   }

   /** @brief Access all interface integrators added with
       AddInterfaceIntegrator(). */
   const Array<InterfaceNonlinearFormIntegrator*> &GetIntefaceIntegrators() const
   { return fnfi; }

   void AssembleInterfaceMatrices(Array2D<SparseMatrix *> &mats) const;

   void AssembleInterfaceMatrixAtPort(const int p, Array<FiniteElementSpace *> &trial_fes_comp, 
                                      Array<FiniteElementSpace *> &test_fes_comp, Array2D<SparseMatrix *> &mats_p) const;

   void InterfaceAddMult(const Vector &x, Vector &y) const
   { "MixedInterfaceForm::InterfaceAddMult is not implemented yet!\n"; }

   void InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const
   { "MixedInterfaceForm::InterfaceGetGradient is not implemented yet!\n"; }

protected:

   // MixedBilinearForm interface operator.
   void AssembleInterfaceMatrix(Mesh *mesh1, Mesh *mesh2,
      FiniteElementSpace *trial_fes1, FiniteElementSpace *trial_fes2,
      FiniteElementSpace *test_fes1, FiniteElementSpace *test_fes2, 
      Array<InterfaceInfo> *interface_infos, Array2D<SparseMatrix*> &mats) const;

};

} // namespace mfem

#endif
