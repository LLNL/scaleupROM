// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nlelast_integ.hpp"

using namespace std;
namespace mfem
{
/* // Boundary integrator
void DGHyperelasticNLFIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, Vector &elvect){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleFaceGrad(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Tr,
                              const Vector &elfun, DenseMatrix &elmat){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleBlock(
       const int dim, const int row_ndofs, const int col_ndofs,
       const int row_offset, const int col_offset,
       const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
       const Vector &row_shape, const Vector &col_shape,
       const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
       DenseMatrix &elmat, DenseMatrix &jmat){MFEM_ABORT("not implemented")};

void DGHyperelasticNLFIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                              const FiniteElement &el2,
                              FaceElementTransformations &Trans,
                              DenseMatrix &elmat){MFEM_ABORT("not implemented")};
 */
// Domain integrator
void HyperelasticNLFIntegratorHR::AssembleElementMatrix(const FiniteElement &,
                                       ElementTransformation &,
                                       DenseMatrix &){MFEM_ABORT("not implemented")};

void HyperelasticNLFIntegratorHR::AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      const Vector &elfun,
                                      Vector &elvect){MFEM_ABORT("not implemented")};
void HyperelasticNLFIntegratorHR::AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &trans,
                                    const Vector &elfun,
                                    DenseMatrix &elmat){MFEM_ABORT("not implemented")};
 
void HyperelasticNLFIntegratorHR::ComputeElementFlux(const FiniteElement &el,
                                    ElementTransformation &Trans,
                                    Vector &u,
                                    const FiniteElement &fluxelem,
                                    Vector &flux, bool with_coef,
                                    const IntegrationRule *ir){MFEM_ABORT("not implemented")};
 
double HyperelasticNLFIntegratorHR::ComputeFluxEnergy(const FiniteElement &fluxelem,
                                     ElementTransformation &Trans,
                                     Vector &flux, Vector *d_energy){MFEM_ABORT("not implemented");
                                     double a = 0;
                                     return a;};
//RHS
void DGHyperelasticDirichletNLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect){MFEM_ABORT("not implemented")};

void DGHyperelasticDirichletNLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                        FaceElementTransformations &Tr,
                                        Vector &elvect){MFEM_ABORT("not implemented")};

} // namespace mfem
