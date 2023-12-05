// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

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
