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

   hs_mats.SetSize(hs.Size());
   hs_mats = NULL;
}

SteadyNSOperator::~SteadyNSOperator()
{
   delete Hop;
   delete system_jac;
   // NonlinearForm owns the gradient operator.
   // DeletePointers(hs_mats);
   delete hs_jac;
   delete uu_mono;
   delete mono_jac;
   delete jac_hypre;
}

void SteadyNSOperator::Mult(const Vector &x, Vector &y) const
{
   assert(linearOp);
   x_u.MakeRef(const_cast<Vector &>(x), 0, u_offsets.Last());
   y_u.MakeRef(y, 0, u_offsets.Last());

   y = 0.0;

   Hop->Mult(x_u, y_u);
   linearOp->AddMult(x, y);
}

Operator& SteadyNSOperator::GetGradient(const Vector &x) const
{
   // NonlinearForm owns the gradient operator.
   // DeletePointers(hs_mats);
   delete hs_jac;
   delete uu_mono;
   delete system_jac;
   delete mono_jac;
   delete jac_hypre;

   hs_jac = new BlockMatrix(u_offsets);
   hs_mats.SetSize(hs.Size());
   for (int i = 0; i < hs.Size(); i++)
   {
      x_u.MakeRef(const_cast<Vector &>(x), u_offsets[i], u_offsets[i+1] - u_offsets[i]);
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
   SteadyNSTensorROM
*/

SteadyNSTensorROM::SteadyNSTensorROM(
   SparseMatrix *linearOp_, Array<DenseTensor *> &hs_, const Array<int> &block_offsets_, const bool direct_solve_)
   : Operator(linearOp_->Height(), linearOp_->Width()), linearOp(linearOp_), hs(hs_),
     block_offsets(block_offsets_), direct_solve(direct_solve_)
{
   // TODO: this needs to be changed for parallel implementation.
   sys_glob_size = height;
   sys_row_starts[0] = 0;
   sys_row_starts[1] = height;

   block_idxs.SetSize(hs.Size());
   for (int m = 0; m < hs.Size(); m++)
   {
      block_idxs[m] = new Array<int>(block_offsets[m+1] - block_offsets[m]);
      for (int k = 0, idx = block_offsets[m]; k < block_idxs[m]->Size(); k++, idx++)
         (*block_idxs[m])[k] = idx;
   }
}

SteadyNSTensorROM::~SteadyNSTensorROM()
{
   DeletePointers(block_idxs);
}

void SteadyNSTensorROM::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   linearOp->Mult(x, y);

   for (int m = 0; m < hs.Size(); m++)
   {
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[m], block_offsets[m+1] - block_offsets[m]);
      y_comp.MakeRef(y, block_offsets[m], block_offsets[m+1] - block_offsets[m]);

      TensorAddScaledContract(*hs[m], 1.0, x_comp, x_comp, y_comp);
   }
}

Operator& SteadyNSTensorROM::GetGradient(const Vector &x) const
{
   delete jac_mono;
   delete jac_hypre;
   jac_mono = new SparseMatrix(*linearOp);
   DenseMatrix jac_comp;

   for (int m = 0; m < hs.Size(); m++)
   {
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[m], block_offsets[m+1] - block_offsets[m]);

      jac_comp.SetSize(block_offsets[m+1] - block_offsets[m]);
      jac_comp = 0.0;
      TensorAddMultTranspose(*hs[m], x_comp, 0, jac_comp);
      TensorAddMultTranspose(*hs[m], x_comp, 1, jac_comp);

      jac_mono->AddSubMatrix(*block_idxs[m], *block_idxs[m], jac_comp);
   }
   jac_mono->Finalize();
   
   if (direct_solve)
   {
      jac_hypre = new HypreParMatrix(MPI_COMM_WORLD, sys_glob_size, sys_row_starts, jac_mono);
      return *jac_hypre;
   }
   else
      return *jac_mono;
}

/*
   SteadyNSSolver
*/

SteadyNSSolver::SteadyNSSolver()
   : StokesSolver(), zeta_coeff(zeta)
{
   // StokesSolver reads viscosity from stokes/nu.
   nu = config.GetOption<double>("stokes/nu", 1.0);
   delete nu_coeff;
   nu_coeff = new ConstantCoefficient(nu);

   ir_nl = &(IntRules.Get(ufes[0]->GetFE(0)->GetGeomType(), (int)(ceil(1.5 * (2 * ufes[0]->GetMaxElementOrder() - 1)))));
}

SteadyNSSolver::~SteadyNSSolver()
{
   DeletePointers(hs);
   // mumps is deleted by StokesSolver.
   // delete mumps;
   delete J_gmres;
   delete newton_solver;

   if (use_rom)
   {
      DeletePointers(comp_tensors);
      if (rom_handler->SaveOperator() != ROMBuildingLevel::COMPONENT)
         DeletePointers(subdomain_tensors);
   }
}

void SteadyNSSolver::InitVariables()
{
   StokesSolver::InitVariables();
   if (use_rom)
   {
      rom_handler->SetNonlinearMode(true);
      subdomain_tensors.SetSize(numSub);
      subdomain_tensors = NULL;
   }
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

void SteadyNSSolver::LoadROMOperatorFromFile(const std::string input_prefix)
{
   assert(rom_handler->SaveOperator() == ROMBuildingLevel::GLOBAL);

   rom_handler->LoadOperatorFromFile(input_prefix);
   
   subdomain_tensors.SetSize(numSub);
   subdomain_tensors = NULL;
   {
      std::string filename = rom_handler->GetOperatorPrefix() + ".h5";
      assert(FileExists(filename));

      hid_t file_id;
      herr_t errf = 0;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      assert(file_id >= 0);

      {
         hid_t grp_id;
         grp_id = H5Gopen2(file_id, "ROM_tensors", H5P_DEFAULT);
         assert(grp_id >= 0);

         for (int m = 0; m < numSub; m++)
         {
            subdomain_tensors[m] = new DenseTensor;
            hdf5_utils::ReadDataset(grp_id, "subdomain" + std::to_string(m), *subdomain_tensors[m]);
         }

         errf = H5Gclose(grp_id);
         assert(errf >= 0);
      }

      errf = H5Fclose(file_id);
      assert(errf >= 0);
   }
}

void SteadyNSSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   StokesSolver::BuildCompROMElement(fes_comp);

   DenseMatrix *basis = NULL;
   const int num_comp = topol_handler->GetNumComponents();
   comp_tensors.SetSize(num_comp);
   comp_tensors = NULL;

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      rom_handler->GetBasis(c, basis);
      comp_tensors[c] = GetReducedTensor(basis, fes_comp[fidx]);
   }  // for (int c = 0; c < num_comp; c++)
}

void SteadyNSSolver::SaveCompBdrROMElement(hid_t &file_id)
{
   MultiBlockSolver::SaveCompBdrROMElement(file_id);

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_tensors.Size() == num_comp);

   hid_t grp_id;
   herr_t errf;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   for (int c = 0; c < num_comp; c++)
   {
      assert(comp_tensors[c]);

      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, std::to_string(c).c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      hdf5_utils::WriteDataset(comp_grp_id, "tensor", *comp_tensors[c]);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void SteadyNSSolver::LoadCompBdrROMElement(hid_t &file_id)
{
   MultiBlockSolver::LoadCompBdrROMElement(file_id);

   const int num_comp = topol_handler->GetNumComponents();
   comp_tensors.SetSize(num_comp);
   comp_tensors = NULL;

   hid_t grp_id;
   herr_t errf;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   for (int c = 0; c < num_comp; c++)
   {
      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, std::to_string(c).c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      comp_tensors[c] = new DenseTensor;
      hdf5_utils::ReadDataset(comp_grp_id, "tensor", *comp_tensors[c]);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   subdomain_tensors.SetSize(numSub);
   subdomain_tensors = NULL;
   for (int m = 0; m < numSub; m++)
      subdomain_tensors[m] = comp_tensors[rom_handler->GetBasisIndexForSubdomain(m)];
}

void SteadyNSSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 100);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-10);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-10);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   int jac_maxIter = config.GetOption<int>("solver/jacobian/max_iter", 10000);
   double jac_rtol = config.GetOption<double>("solver/jacobian/relative_tolerance", 1.e-10);
   double jac_atol = config.GetOption<double>("solver/jacobian/absolute_tolerance", 1.e-10);
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

   subdomain_tensors.SetSize(numSub);
   subdomain_tensors = NULL;

   DenseMatrix *basis = NULL;
   for (int m = 0; m < numSub; m++)
   {
      rom_handler->GetBasisOnSubdomain(m, basis);
      subdomain_tensors[m] = GetReducedTensor(basis, ufes[m]);
   }

   if (rom_handler->SaveOperator() == ROMBuildingLevel::GLOBAL)
   {
      std::string filename = rom_handler->GetOperatorPrefix() + ".h5";
      assert(FileExists(filename));

      hid_t file_id;
      herr_t errf = 0;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      assert(file_id >= 0);

      hid_t grp_id;
      grp_id = H5Gcreate(file_id, "ROM_tensors", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(grp_id >= 0);

      for (int m = 0; m < numSub; m++)
         hdf5_utils::WriteDataset(grp_id, "subdomain" + std::to_string(m), *subdomain_tensors[m]);

      errf = H5Gclose(grp_id);
      assert(errf >= 0);

      errf = H5Fclose(file_id);
      assert(errf >= 0);
   }
}

void SteadyNSSolver::SolveROM()
{
   assert(subdomain_tensors.Size() == numSub);
   for (int m = 0; m < numSub; m++) assert(subdomain_tensors[m]);

   BlockVector U_domain(U->GetData(), domain_offsets); // View vector for U.
   // NOTE(kevin): currently assumes direct solve.
   SteadyNSTensorROM rom_oper(rom_handler->GetOperator(), subdomain_tensors, *(rom_handler->GetBlockOffsets()));
   rom_handler->NonlinearSolve(rom_oper, &U_domain);
}

DenseTensor* SteadyNSSolver::GetReducedTensor(DenseMatrix *basis, FiniteElementSpace *fespace)
{
   assert(basis && fespace);
   const int nvdofs = fespace->GetTrueVSize();
   const int num_basis = basis->NumCols();
   assert(basis->NumRows() >= nvdofs);

   DenseTensor *tensor = new DenseTensor(num_basis, num_basis, num_basis);

   Vector tmp(nvdofs), u_i, u_j, u_k;
   DenseMatrix tmp_jk(num_basis, num_basis);

   // DenseTensor is column major and i is the fastest index. 
   // For fast iteration, we set k to be the test function index.
   for (int i = 0; i < num_basis; i++)
   {
      u_i.SetDataAndSize(basis->GetColumn(i), nvdofs);
      GridFunction ui_gf(fespace, u_i.GetData());
      VectorGridFunctionCoefficient ui_coeff(&ui_gf);

      NonlinearForm h_comp(fespace);
      auto nl_integ_tmp = new VectorConvectionTrilinearFormIntegrator(zeta_coeff, &ui_coeff);
      nl_integ_tmp->SetIntRule(ir_nl);
      h_comp.AddDomainIntegrator(nl_integ_tmp);
      // if (full_dg)
      //    h_comp.AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));

      for (int j = 0; j < num_basis; j++)
      {
         u_j.SetDataAndSize(basis->GetColumn(j), nvdofs);
         tmp = 0.0;
         h_comp.Mult(u_j, tmp);
         
         for (int k = 0; k < num_basis; k++)
         {
            u_k.SetDataAndSize(basis->GetColumn(k), nvdofs);
            (*tensor)(i, j, k) = u_k * tmp;
         }  // for (int k = 0; k < num_basis_c; k++)
      }  // for (int j = 0; j < num_basis_c; j++)
   }  // for (int i = 0; i < num_basis_c; i++)

   return tensor;
}