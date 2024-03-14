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
   Pe = config.GetOption<double>("adv-diff/peclet_number", 0.0);
}

AdvDiffSolver::~AdvDiffSolver()
{
}

void AdvDiffSolver::BuildDomainOperators()
{
   PoissonSolver::BuildDomainOperators();

   for (int m = 0; m < numSub; m++)
   {
      // as[m]->AddDomainIntegrator(new DiffusionIntegrator);
   }
}

void AdvDiffSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
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

      a_comp.Assemble();
      a_comp.Finalize();

      // Poisson equation has only one solution variable.
      comp_mats[c]->SetSize(1, 1);
      (*comp_mats[c])(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
   }
}

void AdvDiffSolver::SetMUMPSSolver()
{
   assert(globalMat_hypre);
   mumps = new MUMPSSolver();
   mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
   mumps->SetOperator(*globalMat_hypre);
}