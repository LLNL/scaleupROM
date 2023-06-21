// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Bilinear Form Integrators

#include "poisson_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"

using namespace std;
using namespace mfem;

PoissonSolver::PoissonSolver()
   : MultiBlockSolver()
{
   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));
 
   // solution dimension is determined by initialization.
   udim = 1;
   num_var = 1;
   vdim.SetSize(num_var);
   vdim = 1;

   // Set up FE collection/spaces.
   fec.SetSize(num_var);
   if (full_dg)
   {
      fec = new DG_FECollection(order, dim);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }

   fes.SetSize(numSub);
   for (int m = 0; m < numSub; m++) {
      fes[m] = new FiniteElementSpace(meshes[m], fec[0], udim);
   }

   var_names.resize(num_var);
   var_names[0] = "solution";
}

PoissonSolver::~PoissonSolver()
{
   delete interface_integ;

   for (int k = 0; k < bs.Size(); k++) delete bs[k];
   for (int k = 0; k < as.Size(); k++) delete as[k];

   for (int k = 0; k < bdr_coeffs.Size(); k++)
      delete bdr_coeffs[k];
      
   for (int k = 0; k < rhs_coeffs.Size(); k++)
      delete rhs_coeffs[k];
}

void PoissonSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();

   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;
}

void PoissonSolver::AddBCFunction(std::function<double(const Vector &)> F, const int battr)
{
   assert(bdr_coeffs.Size() > 0);

   if (battr > 0)
   {
      int idx = global_bdr_attributes.Find(battr);
      if (idx < 0)
      {
         std::string msg = "battr " + std::to_string(battr) + " is not in global boundary attributes. skipping this boundary condition.\n";
         mfem_warning(msg.c_str());
         return;
      }
      bdr_coeffs[idx] = new FunctionCoefficient(F);
   }
   else
      for (int k = 0; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new FunctionCoefficient(F);
}

void PoissonSolver::AddBCFunction(const double &F, const int battr)
{
   assert(bdr_coeffs.Size() > 0);

   if (battr > 0)
   {
      int idx = global_bdr_attributes.Find(battr);
      if (idx < 0)
      {
         std::string msg = "battr " + std::to_string(battr) + " is not in global boundary attributes. skipping this boundary condition.\n";
         mfem_warning(msg.c_str());
         return;
      }
      bdr_coeffs[idx] = new ConstantCoefficient(F);
   }
   else
      for (int k = 0; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new ConstantCoefficient(F);
}

void PoissonSolver::InitVariables()
{
   // number of blocks = solution dimension * number of subdomain;
   block_offsets.SetSize(udim * numSub + 1);
   var_offsets.SetSize(numSub + 1);
   num_vdofs.SetSize(numSub);
   block_offsets[0] = 0;
   var_offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
   {
      var_offsets[i + 1] = fes[i]->GetTrueVSize();
      num_vdofs[i] = fes[i]->GetTrueVSize();
      for (int d = 0; d < udim; d++)
      {
         block_offsets[d + i * udim + 1] = fes[i]->GetNDofs();
      }
   }
   block_offsets.PartialSum();
   var_offsets.PartialSum();
   domain_offsets = var_offsets;

   SetupBCVariables();

   // Set up solution/rhs variables/
   U = new BlockVector(var_offsets);
   RHS = new BlockVector(var_offsets);
   /* 
      Note: for compatibility with ROM, it's better to split with domain_offsets.
      For vector-component operations, can set up a view BlockVector like below:

         BlockVector *U_blocks = new BlockVector(U->GetData(), block_offsets);

      U_blocks does not own the data.
      These are system-specific, therefore not defining it now.
   */

   us.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      us[m] = new GridFunction(fes[m], U->GetBlock(m), 0);
      (*us[m]) = 0.0;

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      us[m]->SetTrueVector();
   }

   rhs_coeffs.SetSize(0);

   if (use_rom) MultiBlockSolver::InitROMHandler();
}

void PoissonSolver::BuildOperators()
{
   BuildRHSOperators();

   BuildDomainOperators();
}

void PoissonSolver::BuildRHSOperators()
{
   SanityCheckOnCoeffs();

   bs.SetSize(numSub);

   // These are heavily system-dependent.
   // Based on scalar/vector system, different integrators/coefficients will be used.
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m).GetData());
      for (int r = 0; r < rhs_coeffs.Size(); r++)
         bs[m]->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeffs[r]));
   }
}

void PoissonSolver::BuildDomainOperators()
{
   SanityCheckOnCoeffs();

   as.SetSize(numSub);

   for (int m = 0; m < numSub; m++)
   {
      as[m] = new BilinearForm(fes[m]);
      as[m]->AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         as[m]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   }

   interface_integ = new InterfaceDGDiffusionIntegrator(sigma, kappa);
}

bool PoissonSolver::BCExistsOnBdr(const int &global_battr_idx)
{
   assert((global_battr_idx >= 0) && (global_battr_idx < global_bdr_attributes.Size()));
   assert(bdr_coeffs.Size() == global_bdr_attributes.Size());
   return (bdr_coeffs[global_battr_idx]);
}

void PoissonSolver::SetupBCOperators()
{
   SetupRHSBCOperators();

   SetupDomainBCOperators();
}

void PoissonSolver::SetupRHSBCOperators()
{
   SanityCheckOnCoeffs();

   assert(bs.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(bs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (!BCExistsOnBdr(b)) continue;

         bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdr_coeffs[b], sigma, kappa), *bdr_markers[b]);
      }
   }
}

void PoissonSolver::SetupDomainBCOperators()
{
   SanityCheckOnCoeffs();

   assert(as.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(as[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (!BCExistsOnBdr(b)) continue;

         as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), *bdr_markers[b]);
      }
   }
}

void PoissonSolver::Assemble()
{
   AssembleRHS();
   AssembleOperator();
}

void PoissonSolver::AssembleRHS()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      bs[m]->Assemble();
   }

   for (int m = 0; m < numSub; m++)
      // Do we really need SyncAliasMemory?
      bs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
}

void PoissonSolver::AssembleOperator()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      as[m]->Assemble();
   }

   mats.SetSize(numSub, numSub);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i == j) {
            mats(i, i) = &(as[i]->SpMat());
         } else {
            mats(i, j) = new SparseMatrix(fes[i]->GetTrueVSize(), fes[j]->GetTrueVSize());
         }
      }
   }
   AssembleInterfaceMatrixes();

   for (int m = 0; m < numSub; m++)
      as[m]->Finalize();

   // globalMat = new BlockOperator(block_offsets);
   // NOTE: currently, domain-decomposed system will have a significantly different sparsity pattern.
   // This is especially true for vector solution, where ordering of component is changed.
   // This is quite inevitable, but is it desirable?
   globalMat = new BlockMatrix(var_offsets);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i != j) mats(i, j)->Finalize();

         globalMat->SetBlock(i, j, mats(i, j));
      }
   }

   if (use_amg)
      globalMat_mono = globalMat->CreateMonolithic();
}

void PoissonSolver::AssembleInterfaceMatrixes()
{
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);

      Array<int> midx(2);
      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;
      Array2D<SparseMatrix *> mats_p(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) mats_p(i, j) = mats(midx[i], midx[j]);

      Mesh *mesh1, *mesh2;
      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      FiniteElementSpace *fes1, *fes2;
      fes1 = fes[midx[0]];
      fes2 = fes[midx[1]];

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);
      AssembleInterfaceMatrix(mesh1, mesh2, fes1, fes2, interface_integ, interface_infos, mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void PoissonSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   const TrainMode train_mode = rom_handler->GetTrainMode();
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

      comp_mats[c] = rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()));
   }
}

void PoissonSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = fes_comp.Size();
   assert(bdr_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      assert(bdr_mats[c]->Size() == comp->bdr_attributes.Size());
      Array<SparseMatrix *> *bdr_mats_c = bdr_mats[c];

      for (int b = 0; b < comp->bdr_attributes.Size(); b++)
      {
         Array<int> bdr_marker(comp->bdr_attributes.Max());
         bdr_marker = 0;
         bdr_marker[comp->bdr_attributes[b] - 1] = 1;
         BilinearForm a_comp(fes_comp[c]);
         a_comp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), bdr_marker);

         a_comp.Assemble();
         a_comp.Finalize();

         (*bdr_mats_c)[b] = rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()));
      }
   }
}

void PoissonSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   assert(port_mats.Size() == num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
   {
      assert(port_mats[p]->NumRows() == 2);
      assert(port_mats[p]->NumCols() == 2);

      int c1, c2;
      topol_handler->GetComponentPair(p, c1, c2);
      Mesh *comp1 = topol_handler->GetComponentMesh(c1);
      Mesh *comp2 = topol_handler->GetComponentMesh(c2);

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      Mesh mesh1(*comp1);
      Mesh mesh2(*comp2);

      Array<int> c_idx(2);
      c_idx[0] = c1;
      c_idx[1] = c2;
      Array2D<SparseMatrix *> spmats(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            spmats(i, j) = new SparseMatrix(fes_comp[c_idx[i]]->GetTrueVSize(), fes_comp[c_idx[j]]->GetTrueVSize());

      Array<InterfaceInfo> *if_infos = topol_handler->GetRefInterfaceInfos(p);

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      AssembleInterfaceMatrix(&mesh1, &mesh2, fes_comp[c1], fes_comp[c2], interface_integ, if_infos, spmats);

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) spmats(i, j)->Finalize();

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            (*port_mats[p])(i, j) = rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], spmats(i,j));

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) delete spmats(i, j);
   }  // for (int p = 0; p < num_ref_ports; p++)
}

void PoissonSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   // TODO: need to change when the actual parallelization is implemented.
   CGSolver *solver = NULL;
   HypreParMatrix *parGlobalMat = NULL;
   HypreBoomerAMG *M = NULL;
   BlockDiagonalPreconditioner *globalPrec = NULL;
   
   // HypreBoomerAMG makes a meaningful difference in computation time.
   if (use_amg)
   {
      // Initializating HypreParMatrix needs the monolithic sparse matrix.
      assert(globalMat_mono != NULL);

      solver = new CGSolver(MPI_COMM_WORLD);
      
      // TODO: need to change when the actual parallelization is implemented.
      HYPRE_BigInt glob_size = block_offsets.Last();
      HYPRE_BigInt row_starts[2] = {0, block_offsets.Last()};
      
      parGlobalMat = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, globalMat_mono);
      M = new HypreBoomerAMG(*parGlobalMat);
      M->SetPrintLevel(print_level);
      solver->SetPreconditioner(*M);

      solver->SetOperator(*parGlobalMat);
   }
   else
   {
      solver = new CGSolver();
      
      if (config.GetOption<bool>("solver/block_diagonal_preconditioner", true))
      {
         globalPrec = new BlockDiagonalPreconditioner(var_offsets);
         solver->SetPreconditioner(*globalPrec);
      }
      solver->SetOperator(*globalMat);
   }
   solver->SetAbsTol(atol);
   solver->SetRelTol(rtol);
   solver->SetMaxIter(maxIter);
   solver->SetPrintLevel(print_level);

   *U = 0.0;
   // The time for the setup above is much smaller than this Mult().
   // StopWatch test;
   // test.Start();
   solver->Mult(*RHS, *U);
   // test.Stop();
   // printf("test: %f seconds.\n", test.RealTime());

   // delete the created objects.
   if (use_amg)
   {
      delete M;
      delete parGlobalMat;
   }
   else
   {
      if (globalPrec != NULL) delete globalPrec;
   }
   delete solver;
}

void PoissonSolver::ProjectOperatorOnReducedBasis()
{ 
   Array2D<Operator *> tmp(mats.NumRows(), mats.NumCols());
   for (int i = 0; i < tmp.NumRows(); i++)
      for (int j = 0; j < tmp.NumCols(); j++)
         tmp(i, j) = mats(i, j);
         
   rom_handler->ProjectOperatorOnReducedBasis(tmp);
}

void PoissonSolver::SanityCheckOnCoeffs()
{
   if (rhs_coeffs.Size() == 0)
      MFEM_WARNING("There is no right-hand side coeffcient assigned! Make sure to set rhs coefficients before BuildOperator.\n");

   if (bdr_coeffs.Size() == 0)
      MFEM_WARNING("There is no bc coeffcient assigned! Make sure to set bc coefficients before SetupBCOperator.\n");

   bool all_null = true;
   for (int i = 0; i < rhs_coeffs.Size(); i++)
      if (rhs_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All rhs coefficents are NULL! Make sure to set rhs coefficients before BuildOperator.\n");

   all_null = true;
   for (int i = 0; i < bdr_coeffs.Size(); i++)
      if (bdr_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All bc coefficients are NULL, meaning there is no Dirichlet BC. Make sure to set bc coefficients before SetupBCOperator.\n");
}

void PoissonSolver::SetParameterizedProblem(ParameterizedProblem *problem)
{
   // clean up rhs for parametrized problem.
   if (rhs_coeffs.Size() > 0)
   {
      for (int k = 0; k < rhs_coeffs.Size(); k++) delete rhs_coeffs[k];
      rhs_coeffs.SetSize(0);
   }
   // clean up boundary functions for parametrized problem.
   bdr_coeffs = NULL;

   for (int b = 0; b < problem->battr.Size(); b++)
   {
      switch (problem->bdr_type[b])
      {
         case PoissonProblem::BoundaryType::DIRICHLET:
         { 
            assert(problem->scalar_bdr_ptr[b]);
            AddBCFunction(*(problem->scalar_bdr_ptr[b]), problem->battr[b]);
            break;
         }
         case PoissonProblem::BoundaryType::NEUMANN: break;
         default:
         case PoissonProblem::BoundaryType::ZERO:
         { AddBCFunction(0.0, problem->battr[b]); break; }
      }
   }

   if (problem->scalar_rhs_ptr != NULL)
      AddRHSFunction(*(problem->scalar_rhs_ptr));
   else
      AddRHSFunction(0.0);
}