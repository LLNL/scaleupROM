// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef SCALEUPROM_DG_MIXED_BILINEAR_HPP
#define SCALEUPROM_DG_MIXED_BILINEAR_HPP

#include "mfem.hpp"

namespace mfem
{

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

   virtual ~MixedBilinearFormDGExtension();
};

class DGNormalFluxIntegrator : public MixedBilinearFormFaceIntegrator
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

   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

} // namespace mfem

#endif
