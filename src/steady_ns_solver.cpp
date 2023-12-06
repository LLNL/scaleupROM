// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "steady_ns_solver.hpp"
#include "hyperreduction_integ.hpp"
// #include "input_parser.hpp"
// #include "hdf5_utils.hpp"
// #include "linalg_utils.hpp"
// #include "dg_bilinear.hpp"
// #include "dg_linear.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

/*
   SteadyNSOperator
*/

SteadyNSOperator::SteadyNSOperator(BlockMatrix *linearOp_, Array<NonlinearForm *> &hs_, Array<int> &u_offsets_, const bool direct_solve_)
   : Operator(linearOp_->Height(), linearOp_->Width()), linearOp(linearOp_), hs(hs_), u_offsets(u_offsets_), direct_solve(direct_solve_),
     M(&(linearOp_->GetBlock(0, 0))), Bt(&(linearOp_->GetBlock(0, 1))), B(&(linearOp_->GetBlock(1, 0)))
{
   vblock_offsets.SetSize(3);
   vblock_offsets[0] = 0;
   vblock_offsets[1] = u_offsets.Last();
   vblock_offsets[2] = height;

   // TODO: this needs to be changed for parallel implementation.
   sys_glob_size = height;
   sys_row_starts[0] = 0;
   sys_row_starts[1] = height;

   Hop = new BlockOperator(u_offsets);
   for (int m = 0; m < hs_.Size(); m++)
   {
      assert(hs_[m]);
      Hop->SetDiagonalBlock(m, hs_[m]);
   }
}

SteadyNSOperator::~SteadyNSOperator()
{
   delete Hop;
   delete system_jac;
   DeletePointers(hs_mats);
   delete hs_jac;
   delete uu_mono;
   delete mono_jac;
   delete jac_hypre;
}

void SteadyNSOperator::Mult(const Vector &x, Vector &y) const
{
   assert(linearOp);
   Vector x_u(x.GetData(), u_offsets.Last()), y_u(y.GetData(), u_offsets.Last());

   y = 0.0;

   Hop->Mult(x_u, y_u);
   linearOp->AddMult(x, y);
}

Operator& SteadyNSOperator::GetGradient(const Vector &x) const
{
   DeletePointers(hs_mats);
   delete hs_jac;
   delete uu_mono;
   delete system_jac;
   delete mono_jac;
   delete jac_hypre;
   
   const Vector x_u(x.GetData(), u_offsets.Last());

   hs_jac = new BlockMatrix(u_offsets);
   hs_mats.SetSize(hs.Size());
   for (int i = 0; i < hs.Size(); i++)
   {
      hs_mats[i] = dynamic_cast<SparseMatrix *>(&hs[i]->GetGradient(x_u));

      hs_jac->SetBlock(i, i, hs_mats[i]);
   }
   SparseMatrix *hs_jac_mono = hs_jac->CreateMonolithic();
   uu_mono = Add(*M, *hs_jac_mono);
   delete hs_jac_mono;

   assert(B && Bt);

   system_jac = new BlockMatrix(vblock_offsets);
   system_jac->SetBlock(0,0, uu_mono);
   system_jac->SetBlock(0,1, Bt);
   system_jac->SetBlock(1,0, B);

   mono_jac = system_jac->CreateMonolithic();
   if (direct_solve)
   {
      jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, sys_glob_size, sys_row_starts, mono_jac);
      return *jac_hypre;
   }  
   else
      return *mono_jac;
}

/*
   SteadyNSSolver
*/

SteadyNSSolver::SteadyNSSolver()
   : StokesSolver(), zeta_coeff(zeta)
{
   // StokesSolver reads viscosity from stokes/nu.
   nu = config.GetOption<double>("navier-stokes/nu", 1.0);
   delete nu_coeff;
   nu_coeff = new ConstantCoefficient(nu);

   ir_nl = &(IntRules.Get(ufes[0]->GetFE(0)->GetGeomType(), (int)(ceil(1.5 * (2 * ufes[0]->GetMaxElementOrder() - 1)))));
}

SteadyNSSolver::~SteadyNSSolver()
{
   DeletePointers(hs);
   delete mumps;
   delete J_gmres;
   delete newton_solver;
}

void SteadyNSSolver::BuildOperators()
{
   BuildRHSOperators();

   BuildDomainOperators();
}

void SteadyNSSolver::BuildDomainOperators()
{
   StokesSolver::BuildDomainOperators();

   hs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      hs[m] = new NonlinearForm(ufes[m]);

      auto nl_integ = new VectorConvectionTrilinearFormIntegrator(zeta_coeff);
      nl_integ->SetIntRule(ir_nl);
      hs[m]->AddDomainIntegrator(nl_integ);
      // if (full_dg)
      //    hs[m]->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));
   }
}

void SteadyNSSolver::Assemble()
{
   StokesSolver::AssembleRHS();

   StokesSolver::AssembleOperator();

   // nonlinear operator?
}

void SteadyNSSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   StokesSolver::BuildCompROMElement(fes_comp);

   const int num_comp = topol_handler->GetNumComponents();
   comp_tensors.SetSize(num_comp);
   comp_tensors = NULL;

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      const int nvdofs = fes_comp[fidx]->GetVSize();

      const int num_basis_c = rom_handler->GetNumBasis(c);
      comp_tensors[c] = new DenseTensor(num_basis_c, num_basis_c, num_basis_c);
      Vector tmp(nvdofs);
      DenseMatrix tmp_jk(num_basis_c, num_basis_c);

      // DenseTensor is column major and i is the fastest index. 
      // For fast iteration, we set k to be the test function index.
      for (int i = 0; i < num_basis_c; i++)
      {
         Vector *u_i = rom_handler->GetBasisVector(c, i, nvdofs);
         GridFunction ui_gf(fes_comp[fidx], u_i->GetData());
         VectorGridFunctionCoefficient ui_coeff(&ui_gf);

         NonlinearForm h_comp(fes_comp[fidx]);
         auto nl_integ_tmp = new VectorConvectionTrilinearFormIntegrator(zeta_coeff, &ui_coeff);
         nl_integ_tmp->SetIntRule(ir_nl);
         h_comp.AddDomainIntegrator(nl_integ_tmp);
         // if (full_dg)
         //    h_comp.AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));

         for (int j = 0; j < num_basis_c; j++)
         {
            Vector *u_j = rom_handler->GetBasisVector(c, j, nvdofs);
            tmp = 0.0;
            h_comp.Mult(*u_j, tmp);
            
            for (int k = 0; k < num_basis_c; k++)
            {
               Vector *u_k = rom_handler->GetBasisVector(c, k, nvdofs);
               (*comp_tensors[c])(i, j, k) = (*u_k) * tmp;
            }  // for (int k = 0; k < num_basis_c; k++)
         }  // for (int j = 0; j < num_basis_c; j++)
      }  // for (int i = 0; i < num_basis_c; i++)
   }  // for (int c = 0; c < num_comp; c++)
}

void SteadyNSSolver::SaveCompBdrROMElement(hid_t &file_id)
{

}

void SteadyNSSolver::LoadCompBdrROMElement(hid_t &file_id)
{

}

void SteadyNSSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 100);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   int jac_maxIter = config.GetOption<int>("solver/jacobian/max_iter", 10000);
   double jac_rtol = config.GetOption<double>("solver/jacobian/relative_tolerance", 1.e-15);
   double jac_atol = config.GetOption<double>("solver/jacobian/absolute_tolerance", 1.e-15);
   int jac_print_level = config.GetOption<int>("solver/jacobian/print_level", -1);

   // same size as var_offsets, but sorted by variables first (then by subdomain).
   Array<int> offsets_byvar(num_var * numSub + 1);
   offsets_byvar = 0;
   for (int k = 0; k < numSub; k++)
   {
      offsets_byvar[k+1] = u_offsets[k+1];
      offsets_byvar[k+1 + numSub] = p_offsets[k+1] + u_offsets.Last();
   }

   // sort out solution/rhs by variables.
   BlockVector rhs_byvar(offsets_byvar);
   BlockVector sol_byvar(offsets_byvar);
   SortByVariables(*RHS, rhs_byvar);
   sol_byvar = 0.0;

   SteadyNSOperator oper(systemOp, hs, u_offsets, direct_solve);

   if (direct_solve)
   {
      mumps = new MUMPSSolver();
      mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps->SetPrintLevel(jac_print_level);
      J_solver = mumps;
   }
   else
   {
      J_gmres = new GMRESSolver;
      J_gmres->SetAbsTol(jac_atol);
      J_gmres->SetRelTol(jac_rtol);
      J_gmres->SetMaxIter(jac_maxIter);
      J_gmres->SetPrintLevel(jac_print_level);
      J_solver = J_gmres;
   }

   newton_solver = new NewtonSolver;
   newton_solver->SetSolver(*J_solver);
   newton_solver->SetOperator(oper);
   newton_solver->SetPrintLevel(print_level); // print Newton iterations
   newton_solver->SetRelTol(rtol);
   newton_solver->SetAbsTol(atol);
   newton_solver->SetMaxIter(maxIter);

   newton_solver->Mult(rhs_byvar, sol_byvar);

   // orthogonalize the pressure.
   if (!pres_dbc)
   {
      Vector pres_view(sol_byvar, vblock_offsets[1], vblock_offsets[2] - vblock_offsets[1]);

      // TODO(kevin): parallelization.
      double tmp = pres_view.Sum() / pres_view.Size();
      pres_view -= tmp;
   }

   SortBySubdomains(sol_byvar, *U);
}

void SteadyNSSolver::ProjectOperatorOnReducedBasis()
{
   StokesSolver::ProjectOperatorOnReducedBasis();

}