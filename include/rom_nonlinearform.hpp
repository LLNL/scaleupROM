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

#ifndef SCALEUPROM_ROM_NONLINEARFORM_HPP
#define SCALEUPROM_ROM_NONLINEARFORM_HPP

#include "mfem.hpp"
#include "hyperreduction_integ.hpp"

namespace mfem
{

class ROMNonlinearForm : public NonlinearForm
{
private:
   static const int Nt = 6;
   mutable StopWatch *jac_timers[Nt];

protected:
   /// ROM basis for projection.
   /// Needs to be converted to MFEM DenseMatrix.
   DenseMatrix *basis = NULL;   // not owned

   /// Projection ROM's jacobian matrix is dense most of time.
   mutable DenseMatrix *Grad = NULL;

   /// Set of Domain Integrators to be assembled (added).
   Array<HyperReductionIntegrator*> dnfi; // owned
   // hyper reduction sampling indexes.
   Array<Array<SampleInfo> *> dnfi_sample;

   /// Set of interior face Integrators to be assembled (added).
   Array<HyperReductionIntegrator*> fnfi; // owned
   Array<Array<SampleInfo> *> fnfi_sample;

   /// Set of boundary face Integrators to be assembled (added).
   Array<HyperReductionIntegrator*> bfnfi; // owned
   Array<Array<SampleInfo> *> bfnfi_sample;

   /// @brief Flag for precomputing necessary coefficients for fast computation.
   bool precompute = false;

public:
   /// Construct a NonlinearForm on the given FiniteElementSpace, @a f.
   /** As an Operator, the NonlinearForm has input and output size equal to the
      number of true degrees of freedom, i.e. f->GetTrueVSize(). */
   ROMNonlinearForm(const int num_basis, FiniteElementSpace *f);

   /** @brief Destroy the NonlinearForm including the owned
       NonlinearFormIntegrator%s and gradient Operator. */
   virtual ~ROMNonlinearForm();

   void SetPrecomputeMode(const bool precompute_) { precompute = precompute_; }

   void PrecomputeCoefficients();

   void SetBasis(DenseMatrix &basis_)
   {
      assert(basis_.NumCols() == height);
      assert(basis_.NumRows() == fes->GetTrueVSize());
      basis = &basis_;
   }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(HyperReductionIntegrator *nlfi)
   {
      dnfi.Append(nlfi);
      dnfi_sample.Append(NULL);
   }

   void UpdateDomainIntegratorSampling(const int i, const Array<int> &el, const Array<int> &qp, const Array<double> &qw)
   {
      assert((i >= 0) && (i < dnfi.Size()));
      assert(dnfi.Size() == dnfi_sample.Size());

      dnfi_sample[i] = new Array<SampleInfo>(0);
      for (int s = 0; s < el.Size(); s++)
         dnfi_sample[i]->Append({.el=el[s], .face=-1, .be=-1, .qp=qp[s], .qw=qw[s]});
   }

   /// Access all integrators added with AddDomainIntegrator().
   Array<HyperReductionIntegrator*> *GetDNFI() { return &dnfi; }
   const Array<HyperReductionIntegrator*> *GetDNFI() const { return &dnfi; }

   /// Adds new Interior Face Integrator.
   void AddInteriorFaceIntegrator(HyperReductionIntegrator *nlfi)
   {
      fnfi.Append(nlfi);
      fnfi_sample.Append(NULL);
   }

   void UpdateInteriorFaceIntegratorSampling(const int i, const Array<int> &face, const Array<int> &qp, const Array<double> &qw)
   {
      assert((i >= 0) && (i < fnfi.Size()));
      assert(fnfi.Size() == fnfi_sample.Size());

      fnfi_sample[i] = new Array<SampleInfo>(0);
      for (int s = 0; s < face.Size(); s++)
         fnfi_sample[i]->Append({.el=-1, .face=face[s], .be=-1, .qp=qp[s], .qw=qw[s]});
   }

   /** @brief Access all interior face integrators added with
       AddInteriorFaceIntegrator(). */
   const Array<HyperReductionIntegrator*> &GetInteriorFaceIntegrators() const
   { return fnfi; }

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator(HyperReductionIntegrator *nlfi)
   { 
      bfnfi.Append(nlfi);
      bfnfi_marker.Append(NULL);
      bfnfi_sample.Append(NULL);
   }

   /** @brief Adds new Boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(HyperReductionIntegrator *nfi,
                             Array<int> &bdr_marker)
   {
      bfnfi.Append(nfi);
      bfnfi_marker.Append(&bdr_marker);
      bfnfi_sample.Append(NULL);
   }

   void UpdateBdrFaceIntegratorSampling(const int i, const Array<int> &be, const Array<int> &qp, const Array<double> &qw)
   {
      assert((i >= 0) && (i < bfnfi.Size()));
      assert(bfnfi.Size() == bfnfi_sample.Size());

      bfnfi_sample[i] = new Array<SampleInfo>(0);
      for (int s = 0; s < be.Size(); s++)
         bfnfi_sample[i]->Append({.el=-1, .face=-1, .be=be[s], .qp=qp[s], .qw=qw[s]});
   }

   /** @brief Access all boundary face integrators added with
       AddBdrFaceIntegrator(). */
   const Array<HyperReductionIntegrator*> &GetBdrFaceIntegrators() const
   { return bfnfi; }

   // /// Compute the energy corresponding to the state @a x.
   // /** In general, @a x may have non-homogeneous essential boundary values.

   //     The state @a x must be a "GridFunction size" vector, i.e. its size must
   //     be fes->GetVSize(). */
   // double GetGridFunctionEnergy(const Vector &x) const;

   // /// Compute the energy corresponding to the state @a x.
   // /** In general, @a x may have non-homogeneous essential boundary values.

   //     The state @a x must be a true-dof vector. */
   // virtual double GetEnergy(const Vector &x) const
   // { return GetGridFunctionEnergy(Prolongate(x)); }

   /// Evaluate the action of the NonlinearForm.
   /** The input essential dofs in @a x will, generally, be non-zero. However,
       the output essential dofs in @a y will always be set to zero.

       Both the input and the output vectors, @a x and @a y, must be true-dof
       vectors, i.e. their size must be fes->GetTrueVSize(). */
   virtual void Mult(const Vector &x, Vector &y) const;

   /** @brief Compute the gradient Operator of the NonlinearForm corresponding
       to the state @a x. */
   /** Any previously specified essential boundary conditions will be
       automatically imposed on the gradient operator.

       The returned object is valid until the next call to this method or the
       destruction of this object.

       In general, @a x may have non-homogeneous essential boundary values.

       The state @a x must be a true-dof vector. */
   virtual Operator &GetGradient(const Vector &x) const;

};

} // namespace mfem

#endif
