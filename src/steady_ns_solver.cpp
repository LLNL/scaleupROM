// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "steady_ns_solver.hpp"
#include "hyperreduction_integ.hpp"
#include "nonlinear_integ.hpp"
// #include "input_parser.hpp"
// #include "hdf5_utils.hpp"
// #include "linalg_utils.hpp"
// #include "dg_bilinear.hpp"
#include "dg_linear.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

/*
   SteadyNSOperator
*/

SteadyNSOperator::SteadyNSOperator(
   BlockMatrix *linearOp_, Array<NonlinearForm *> &hs_, InterfaceForm *nl_itf_,
   Array<int> &u_offsets_, const bool direct_solve_)
   : Operator(linearOp_->Height(), linearOp_->Width()), linearOp(linearOp_), hs(hs_), nl_itf(nl_itf_),
     u_offsets(u_offsets_), direct_solve(direct_solve_),
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

   hs_mats.SetSize(hs.Size(), hs.Size());
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
   if (nl_itf) nl_itf->InterfaceAddMult(x_u, y_u);
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
   for (int i = 0; i < hs.Size(); i++)
      for (int j = 0; j < hs.Size(); j++)
      {
         if (i == j)
         {
            x_u.MakeRef(const_cast<Vector &>(x), u_offsets[i], u_offsets[i+1] - u_offsets[i]);
            hs_mats(i, j) = dynamic_cast<SparseMatrix *>(&hs[i]->GetGradient(x_u));
         }
         else
         {
            delete hs_mats(i, j);
            hs_mats(i, j) = new SparseMatrix(u_offsets[i+1] - u_offsets[i], u_offsets[j+1] - u_offsets[j]);
         }
      }

   x_u.MakeRef(const_cast<Vector &>(x), 0, u_offsets.Last());
   if (nl_itf) nl_itf->InterfaceGetGradient(x_u, hs_mats);

   for (int i = 0; i < hs.Size(); i++)
      for (int j = 0; j < hs.Size(); j++)
      {
         hs_mats(i, j)->Finalize();
         hs_jac->SetBlock(i, j, hs_mats(i, j));
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
      jac_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, mono_jac);
      return *jac_hypre;
   }  
   else
      return *mono_jac;
}

/*
   SteadyNSROM
*/

SteadyNSROM::SteadyNSROM(
   SparseMatrix *linearOp_, const int numSub_, const Array<int> &block_offsets_, const bool direct_solve_)
   : Operator(linearOp_->Height(), linearOp_->Width()), linearOp(linearOp_),
     numSub(numSub_), block_offsets(block_offsets_), direct_solve(direct_solve_)
{
   separate_variable = (block_offsets.Size() == num_var * numSub + 1);
   assert(separate_variable || (block_offsets.Size() == numSub + 1));

   // TODO: this needs to be changed for parallel implementation.
   sys_glob_size = height;
   sys_row_starts[0] = 0;
   sys_row_starts[1] = height;

   block_idxs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      int midx = (separate_variable) ? m * num_var : m;
      block_idxs[m] = new Array<int>(block_offsets[midx+1] - block_offsets[midx]);
      for (int k = 0, idx = block_offsets[midx]; k < block_idxs[m]->Size(); k++, idx++)
         (*block_idxs[m])[k] = idx;
   }
}

SteadyNSROM::~SteadyNSROM()
{
   DeletePointers(block_idxs);
}

/*
   SteadyNSTensorROM
*/

void SteadyNSTensorROM::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   linearOp->Mult(x, y);

   for (int m = 0; m < numSub; m++)
   {
      int midx = (separate_variable) ? m * num_var : m;
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);
      y_comp.MakeRef(y, block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);

      TensorAddScaledContract(*hs[m], 1.0, x_comp, x_comp, y_comp);
   }
}

Operator& SteadyNSTensorROM::GetGradient(const Vector &x) const
{
   delete jac_mono;
   delete jac_hypre;
   jac_mono = new SparseMatrix(*linearOp);
   DenseMatrix jac_comp;

   for (int m = 0; m < numSub; m++)
   {
      int midx = (separate_variable) ? m * num_var : m;
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);

      jac_comp.SetSize(block_offsets[midx+1] - block_offsets[midx]);
      jac_comp = 0.0;
      TensorAddMultTranspose(*hs[m], x_comp, 0, jac_comp);
      TensorAddMultTranspose(*hs[m], x_comp, 1, jac_comp);

      jac_mono->AddSubMatrix(*block_idxs[m], *block_idxs[m], jac_comp);
   }
   jac_mono->Finalize();
   
   if (direct_solve)
   {
      jac_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, jac_mono);
      return *jac_hypre;
   }
   else
      return *jac_mono;
}

/*
   SteadyNSTensorROM
*/

void SteadyNSEQPROM::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   linearOp->Mult(x, y);

   for (int m = 0; m < numSub; m++)
   {
      int midx = (separate_variable) ? m * num_var : m;
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);
      y_comp.MakeRef(y, block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);

      hs[m]->AddMult(x_comp, y_comp);
   }
}

Operator& SteadyNSEQPROM::GetGradient(const Vector &x) const
{
   delete jac_mono;
   delete jac_hypre;
   jac_mono = new SparseMatrix(*linearOp);
   DenseMatrix *jac_comp;

   for (int m = 0; m < numSub; m++)
   {
      int midx = (separate_variable) ? m * num_var : m;
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[midx], block_offsets[midx+1] - block_offsets[midx]);

      // NOTE(kevin): jac_comp is owned by hs[m]. No need of deleting it.
      jac_comp = dynamic_cast<DenseMatrix *>(&hs[m]->GetGradient(x_comp));
      jac_mono->AddSubMatrix(*block_idxs[m], *block_idxs[m], *jac_comp);
   }
   jac_mono->Finalize();
   
   if (direct_solve)
   {
      jac_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, jac_mono);
      return *jac_hypre;
   }
   else
      return *jac_mono;
}

/*
   SteadyNSSolver
*/

SteadyNSSolver::SteadyNSSolver()
   : StokesSolver()
{
   nonlinear_mode = true;

   // StokesSolver reads viscosity from stokes/nu.
   nu = config.GetOption<double>("stokes/nu", 1.0);
   delete nu_coeff;
   nu_coeff = new ConstantCoefficient(nu);

   zeta = config.GetOption<double>("navier-stokes/zeta", 1.0);
   zeta_coeff = new ConstantCoefficient(zeta);
   minus_zeta = new ConstantCoefficient(-zeta);
   minus_half_zeta = new ConstantCoefficient(-0.5 * zeta);

   std::string oper_str = config.GetOption<std::string>("navier-stokes/operator-type", "base");
   if (oper_str == "base")       oper_type = OperType::BASE;
   else if (oper_str == "temam") oper_type = OperType::TEMAM;
   else
      mfem_error("SteadyNSSolver: unknown operator type!\n");

   ir_nl = &(IntRules.Get(ufes[0]->GetFE(0)->GetGeomType(), (int)(ceil(1.5 * (2 * ufes[0]->GetMaxElementOrder() - 1)))));

   /* SteadyNSSolver requires all the meshes to have the same element type. */
   int num_comp = topol_handler->GetNumComponents();
   const Element::Type type0 = topol_handler->GetMesh(0)->GetElementType(0);
   for (int c = 0; c < num_comp; c++)
      if (type0 != topol_handler->GetMesh(c)->GetElementType(0))
         mfem_error("SteadyNSSolver requires all meshes to have the same element type!\n");
}

SteadyNSSolver::~SteadyNSSolver()
{
   delete zeta_coeff;
   delete minus_zeta;
   delete minus_half_zeta;
   delete nl_itf;
   DeletePointers(hs);
   // mumps is deleted by StokesSolver.
   // delete mumps;
   delete J_gmres;
   delete newton_solver;

   if (use_rom)
   {
      if (rom_handler->GetNonlinearHandling() == NonlinearHandling::TENSOR)
      {
         DeletePointers(comp_tensors);
         if (rom_handler->GetBuildingLevel() != ROMBuildingLevel::COMPONENT)
            DeletePointers(subdomain_tensors);
      }
      else if (rom_handler->GetNonlinearHandling() == NonlinearHandling::EQP)
         DeletePointers(comp_eqps);
   }
}

void SteadyNSSolver::InitVariables()
{
   StokesSolver::InitVariables();
   if (use_rom)
   {
      rom_handler->SetNonlinearMode(nonlinear_mode);
      subdomain_tensors.SetSize(numSub);
      subdomain_tensors = NULL;
      subdomain_eqps.SetSize(numSub);
      subdomain_eqps = NULL;
   }
}

void SteadyNSSolver::BuildDomainOperators()
{
   StokesSolver::BuildDomainOperators();

   hs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      hs[m] = new NonlinearForm(ufes[m]);

      switch (oper_type)
      {
         case OperType::BASE:
         {
            auto nl_integ = new VectorConvectionTrilinearFormIntegrator(*zeta_coeff);
            nl_integ->SetIntRule(ir_nl);
            hs[m]->AddDomainIntegrator(nl_integ);
         }
         break;
         case OperType::TEMAM:
         {
            auto nl_integ = new TemamTrilinearFormIntegrator(*zeta_coeff);
            nl_integ->SetIntRule(ir_nl);
            hs[m]->AddDomainIntegrator(nl_integ);
            if (full_dg)
            {
               auto nl_face = new DGTemamFluxIntegrator(*minus_zeta);
               // nl_face->SetIntRule(ir_nl);
               hs[m]->AddInteriorFaceIntegrator(nl_face);
            }
         }
         break;
         default:
            mfem_error("SteadyNSSolver: unknown operator type!\n");
         break;
      }
   }

   if (oper_type == OperType::TEMAM)
   {
      nl_itf = new InterfaceForm(meshes, ufes, topol_handler);
      nl_itf->AddIntefaceIntegrator(new InterfaceDGTemamFluxIntegrator(*minus_zeta));
      // nl_interface->SetIntRule(ir_nl);
   }
}

void SteadyNSSolver::SetupRHSBCOperators()
{
   StokesSolver::SetupRHSBCOperators();

   if (oper_type != OperType::TEMAM) return;

   for (int m = 0; m < numSub; m++)
   {
      assert(fs[m] && gs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (!BCExistsOnBdr(b)) continue;
         // TODO: Non-homogeneous Neumann stress bc
         if (bdr_type[b] == BoundaryType::NEUMANN)
            continue;

         fs[m]->AddBdrFaceIntegrator(new DGBdrTemamLFIntegrator(*ud_coeffs[b], minus_half_zeta), *bdr_markers[b]);
      }
   }
}

void SteadyNSSolver::SetupDomainBCOperators()
{
   StokesSolver::SetupDomainBCOperators();

   if (oper_type != OperType::TEMAM) return;

   assert(hs.Size() == numSub);
   for (int m = 0; m < numSub; m++)
   {
      assert(hs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         
         // homogeneous Neumann boundary condition
         if (!BCExistsOnBdr(b) && (bdr_type[b] == BoundaryType::NEUMANN))
            hs[m]->AddBdrFaceIntegrator(new DGTemamFluxIntegrator(*zeta_coeff), *bdr_markers[b]);
      }
   }
   
}

void SteadyNSSolver::AssembleOperator()
{
   StokesSolver::AssembleOperatorBase();

   if (direct_solve)
      StokesSolver::SetupMUMPSSolver(false);
   else
      // pressure mass matrix for preconditioner.
      StokesSolver::SetupPressureMassMatrix();
}

void SteadyNSSolver::SaveROMOperator(const std::string input_prefix)
{
   MultiBlockSolver::SaveROMOperator(input_prefix);

   if (rom_handler->GetNonlinearHandling() != NonlinearHandling::TENSOR)
      return;

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

void SteadyNSSolver::LoadROMOperatorFromFile(const std::string input_prefix)
{
   assert(rom_handler->GetBuildingLevel() == ROMBuildingLevel::GLOBAL);

   rom_handler->LoadOperatorFromFile(input_prefix);

   if (rom_handler->GetNonlinearHandling() != NonlinearHandling::TENSOR)
      return;
   
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

bool SteadyNSSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 100);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-10);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-10);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   int jac_maxIter = config.GetOption<int>("solver/jacobian/max_iter", 10000);
   double jac_rtol = config.GetOption<double>("solver/jacobian/relative_tolerance", 1.e-10);
   double jac_atol = config.GetOption<double>("solver/jacobian/absolute_tolerance", 1.e-10);
   int jac_print_level = config.GetOption<int>("solver/jacobian/print_level", -1);

   bool lbfgs = config.GetOption<bool>("solver/use_lbfgs", false);
   bool use_restart = config.GetOption<bool>("solver/use_restart", false);
   std::string restart_file;
   if (use_restart)
      restart_file = config.GetRequiredOption<std::string>("solver/restart_file");

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
   if (use_restart)
   {
      LoadSolution(restart_file);
      SortByVariables(*U, sol_byvar);
   }
   else
      sol_byvar = 0.0;

   SteadyNSOperator oper(systemOp, hs, nl_itf, u_offsets, direct_solve);

   if (direct_solve)
   {
      mumps = new MUMPSSolver(MPI_COMM_SELF);
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

   if (lbfgs)
      newton_solver = new LBFGSSolver;
   else
      newton_solver = new NewtonSolver;
   newton_solver->SetSolver(*J_solver);
   newton_solver->SetOperator(oper);
   newton_solver->SetPrintLevel(print_level); // print Newton iterations
   newton_solver->SetRelTol(rtol);
   newton_solver->SetAbsTol(atol);
   newton_solver->SetMaxIter(maxIter);

   newton_solver->Mult(rhs_byvar, sol_byvar);
   bool converged = newton_solver->GetConverged();

   // orthogonalize the pressure.
   if (!pres_dbc)
   {
      Vector pres_view(sol_byvar, vblock_offsets[1], vblock_offsets[2] - vblock_offsets[1]);

      // TODO(kevin): parallelization.
      double tmp = pres_view.Sum() / pres_view.Size();
      pres_view -= tmp;
   }

   SortBySubdomains(sol_byvar, *U);

   return converged;
}

void SteadyNSSolver::ProjectOperatorOnReducedBasis()
{
   StokesSolver::ProjectOperatorOnReducedBasis();

   if (rom_handler->GetNonlinearHandling() != NonlinearHandling::TENSOR)
      return;

   subdomain_tensors.SetSize(numSub);
   subdomain_tensors = NULL;

   DenseMatrix *basis = NULL;
   for (int m = 0; m < numSub; m++)
   {
      int idx = (separate_variable_basis) ? m * num_var : m;
      rom_handler->GetDomainBasis(idx, basis);
      subdomain_tensors[m] = GetReducedTensor(basis, ufes[m]);
   }
}

void SteadyNSSolver::SolveROM()
{
   // View vector for U.
   BlockVector *U_domain = NULL;
   if (separate_variable_basis)
      U_domain = new BlockVector(U->GetData(), var_offsets);
   else
      U_domain = new BlockVector(U->GetData(), domain_offsets);

   bool use_restart = config.GetOption<bool>("solver/use_restart", false);
   std::string restart_file;
   if (use_restart)
   {
      restart_file = config.GetRequiredOption<std::string>("solver/restart_file");
      LoadSolution(restart_file);
   }

   // NOTE(kevin): currently assumes direct solve.
   SteadyNSROM *rom_oper = NULL;
   if (rom_handler->GetNonlinearHandling() == NonlinearHandling::TENSOR)
   {
      assert(subdomain_tensors.Size() == numSub);
      for (int m = 0; m < numSub; m++) assert(subdomain_tensors[m]);
      rom_oper = new SteadyNSTensorROM(rom_handler->GetOperator(), subdomain_tensors, *(rom_handler->GetBlockOffsets()));
   }
   else if (rom_handler->GetNonlinearHandling() == NonlinearHandling::EQP)
   {
      assert(subdomain_eqps.Size() == numSub);
      for (int m = 0; m < numSub; m++) assert(subdomain_eqps[m]);
      rom_oper = new SteadyNSEQPROM(rom_handler->GetOperator(), subdomain_eqps, *(rom_handler->GetBlockOffsets()));
   }
   
   rom_handler->NonlinearSolve(*rom_oper, U_domain);

   delete rom_oper;
}

void SteadyNSSolver::AllocateROMTensorElems()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);

   const int num_comp = topol_handler->GetNumComponents();
   comp_tensors.SetSize(num_comp);
   comp_tensors = NULL;
}

void SteadyNSSolver::BuildROMTensorElems()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   // Component domain system
   const int num_comp = topol_handler->GetNumComponents();
   Array<FiniteElementSpace *> fes_comp;
   GetComponentFESpaces(fes_comp);

   DenseMatrix *basis = NULL;
   assert(comp_tensors.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      const int cidx = (separate_variable_basis) ? fidx : c;
      rom_handler->GetReferenceBasis(cidx, basis);
      comp_tensors[c] = GetReducedTensor(basis, fes_comp[fidx]);
   }  // for (int c = 0; c < num_comp; c++)

   for (int k = 0 ; k < fes_comp.Size(); k++) delete fes_comp[k];
}

void SteadyNSSolver::SaveROMTensorElems(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);
   assert(FileExists(filename));

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
   assert(file_id >= 0);

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_tensors.Size() == num_comp);

   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      assert(comp_tensors[c]);
      dset_name = topol_handler->GetComponentName(c);

      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, dset_name.c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      hdf5_utils::WriteDataset(comp_grp_id, "tensor", *comp_tensors[c]);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

void SteadyNSSolver::LoadROMTensorElems(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_tensors.Size() == num_comp);

   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
   assert(grp_id >= 0);

   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      dset_name = topol_handler->GetComponentName(c);

      hid_t comp_grp_id;
      comp_grp_id = H5Gopen2(grp_id, dset_name.c_str(), H5P_DEFAULT);
      assert(comp_grp_id >= 0);

      comp_tensors[c] = new DenseTensor;
      hdf5_utils::ReadDataset(comp_grp_id, "tensor", *comp_tensors[c]);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);

   errf = H5Fclose(file_id);
   assert(errf >= 0);
}

void SteadyNSSolver::AssembleROMTensorOper()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_tensors.Size() == num_comp);

   subdomain_tensors.SetSize(numSub);
   subdomain_tensors = NULL;
   for (int m = 0; m < numSub; m++)
      subdomain_tensors[m] = comp_tensors[rom_handler->GetRefIndexForSubdomain(m)];
}

void SteadyNSSolver::TrainEQP(SampleGenerator *sample_generator)
{
   assert(sample_generator);
   assert(rom_handler);
   assert(rom_handler->BasisLoaded());

   double eqp_tol = config.GetOption<double>("model_reduction/eqp/relative_tolerance", 1.0e-2);

   SetupEQPOperators();

   /* EQP NNLS for each reference ROM component */
   const int num_comp = rom_handler->GetNumROMRefComps();
   std::string basis_tag;
   for (int c = 0; c < num_comp; c++)
   {
      int idx = (separate_variable_basis) ? c * num_var : c;
      basis_tag = rom_handler->GetRefBasisTag(idx);

      const CAROM::Matrix *snapshots = sample_generator->LookUpSnapshot(basis_tag);
      comp_eqps[c]->TrainEQP(*snapshots, eqp_tol);
   }
}

void SteadyNSSolver::SaveEQP()
{
   std::string filename = rom_handler->GetBasisPrefix();
   filename += ".eqp.h5";

   /*
      TODO(kevin): this is a boilerplate for parallel POD/EQP training.
      Full parallelization will save EQ points/weights in a parallel way.
   */
   if (rank == 0)
   {

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   const int num_comp = rom_handler->GetNumROMRefComps();
   assert(comp_eqps.Size() == num_comp);
   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      assert(comp_eqps[c]);
      dset_name = GetBasisTagForComponent(c, train_mode, topol_handler);
      // only one integrator exists in each nonlinear form.
      comp_eqps[c]->SaveEQPForIntegrator(IntegratorType::DOMAIN, 0, file_id, dset_name);
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   }
   MPI_Barrier(MPI_COMM_WORLD);
   return;
}

void SteadyNSSolver::LoadEQP()
{
   assert(rom_handler->BasisLoaded());

   SetupEQPOperators();

   const int num_comp = rom_handler->GetNumROMRefComps();

   std::string filename = rom_handler->GetBasisPrefix();
   filename += ".eqp.h5";

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   std::string dset_name;
   for (int c = 0; c < num_comp; c++)
   {
      assert(comp_eqps[c]);
      dset_name = GetBasisTagForComponent(c, train_mode, topol_handler);
      // only one integrator exists in each nonlinear form.
      comp_eqps[c]->LoadEQPForIntegrator(IntegratorType::DOMAIN, 0, file_id, dset_name);

      if (comp_eqps[c]->PrecomputeMode())
         comp_eqps[c]->PrecomputeCoefficients();
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);
}

DenseTensor* SteadyNSSolver::GetReducedTensor(DenseMatrix *basis, FiniteElementSpace *fespace)
{
   assert(basis && fespace);
   const int nvdofs = fespace->GetTrueVSize();
   const int num_basis = basis->NumCols();
   assert(basis->NumRows() >= nvdofs);

   if (oper_type == OperType::TEMAM)
      mfem_error("SteadyNSSolver: Temam Operator is not implemented for ROM yet!\n");

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
      auto nl_integ_tmp = new VectorConvectionTrilinearFormIntegrator(*zeta_coeff, &ui_coeff);
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

void SteadyNSSolver::SetupEQPOperators()
{
   assert(rom_handler);
   assert(rom_handler->BasisLoaded());

   bool precompute = config.GetOption<bool>("model_reduction/eqp/precompute", false);

   const int num_comp = rom_handler->GetNumROMRefComps();
   comp_eqps.SetSize(num_comp);
   comp_fes.SetSize(num_comp);
   comp_eqps = NULL;
   comp_fes = NULL;

   for (int c = 0; c < num_comp; c++)
   {
      int midx = -1;
      for (int m = 0; m < numSub; m++)
         if (rom_handler->GetRefIndexForSubdomain(m) == c)
         {
            midx = m;
            break;
         }
      assert((midx >= 0) && (midx < numSub));

      comp_fes[c] = ufes[midx];
   }

   DenseMatrix *basis;
   for (int c = 0; c < num_comp; c++)
   {
      int idx = (separate_variable_basis) ? c * num_var : c;
      rom_handler->GetReferenceBasis(idx, basis);

      auto nl_integ_tmp = new VectorConvectionTrilinearFormIntegrator(*zeta_coeff);
      nl_integ_tmp->SetIntRule(ir_nl);

      comp_eqps[c] = new ROMNonlinearForm(basis->NumCols(), comp_fes[c]);
      comp_eqps[c]->AddDomainIntegrator(nl_integ_tmp);
      comp_eqps[c]->SetBasis(*basis);
      comp_eqps[c]->SetPrecomputeMode(precompute);
   }

   subdomain_eqps.SetSize(numSub);
   subdomain_eqps = NULL;
   for (int m = 0; m < numSub; m++)
      subdomain_eqps[m] = comp_eqps[rom_handler->GetRefIndexForSubdomain(m)];
}
