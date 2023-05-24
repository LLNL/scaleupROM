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

#ifndef SCALEUPROM_DG_BILINEAR_HPP
#define SCALEUPROM_DG_BILINEAR_HPP

#include "mfem.hpp"

namespace mfem
{

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

} // namespace mfem

#endif
