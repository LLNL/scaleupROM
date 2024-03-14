// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "advdiff_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

AdvDiffSolver::AdvDiffSolver()
   : PoissonSolver()
{
   // ConvectionIntegrator does not support L2 space.
   assert(!full_dg);

   Pe = config.GetOption<double>("adv-diff/peclet_number", 0.0);

   flow_coeffs.SetSize(numSub);
}

AdvDiffSolver::~AdvDiffSolver()
{
   DeletePointers(flow_coeffs);
}

void AdvDiffSolver::BuildDomainOperators()
{
   PoissonSolver::BuildDomainOperators();

   for (int m = 0; m < numSub; m++)
   {
      assert(flow_coeffs[m]);
      as[m]->AddDomainIntegrator(new ConvectionIntegrator(*flow_coeffs[m], Pe));
   }
}

void AdvDiffSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   mfem_error("AdvDiffSolver::BuildCompROMElement is not implemented yet!\n");

   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = fes_comp.Size();
   assert(comp_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      BilinearForm a_comp(fes_comp[c]);

      a_comp.AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         a_comp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));

      // TODO(kevin): ConvectionIntegrator needs to be added

      a_comp.Assemble();
      a_comp.Finalize();

      // Poisson equation has only one solution variable.
      comp_mats[c]->SetSize(1, 1);
      (*comp_mats[c])(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
   }
}

void AdvDiffSolver::SetFlowAtSubdomain(std::function<void(const Vector &, Vector &)> F, const int m)
{
   assert(flow_coeffs.Size() == numSub);
   assert((m == -1) || ((m >= 0) && (m < numSub)));

   if (m == -1)
   {
      for (int k = 0; k < numSub; k++)
         flow_coeffs[k] = new VectorFunctionCoefficient(dim, F);
   }
   else
      flow_coeffs[m] = new VectorFunctionCoefficient(dim, F);
}

void AdvDiffSolver::SetParameterizedProblem(ParameterizedProblem *problem)
{
   PoissonSolver::SetParameterizedProblem(problem);

   // TODO(kevin): add flow field setup.
}

void AdvDiffSolver::SetMUMPSSolver()
{
   assert(globalMat_hypre);
   mumps = new MUMPSSolver();
   mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps->SetOperator(*globalMat_hypre);
}