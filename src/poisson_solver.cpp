// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "poisson_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

PoissonSolver::PoissonSolver()
   : MultiBlockSolver()
{
   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));
 
   var_names = GetVariableNames();
   num_var = var_names.size();

   // solution dimension is determined by initialization.
   udim = 1;
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
}

PoissonSolver::~PoissonSolver()
{
   delete a_itf;

   DeletePointers(bs);
   DeletePointers(as);
   DeletePointers(bdr_coeffs);
   DeletePointers(rhs_coeffs);

   delete globalMat_mono;
   delete globalMat;
   delete globalMat_hypre;
   delete mumps;
}

void PoissonSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();

   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;
}

void PoissonSolver::AddBCFunction(std::function<double(const Vector &, double)> F, const int battr)
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

   a_itf = new InterfaceForm(meshes, fes, topol_handler);
   a_itf->AddInterfaceIntegrator(new InterfaceDGDiffusionIntegrator(sigma, kappa));
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
         if (bdr_type[b] == BoundaryType::NEUMANN)
            continue;

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
         if (bdr_type[b] == BoundaryType::NEUMANN)
            continue;

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
   AssembleInterfaceMatrices();

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

   if (use_amg || direct_solve)
   {
      globalMat_mono = globalMat->CreateMonolithic();

      // TODO: need to change when the actual parallelization is implemented.
      sys_glob_size = globalMat_mono->NumRows();
      sys_row_starts[0] = 0;
      sys_row_starts[1] = globalMat_mono->NumRows();
      globalMat_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, globalMat_mono);

      if (direct_solve) SetMUMPSSolver();
   }
}

void PoissonSolver::AssembleInterfaceMatrices()
{
   assert(a_itf);
   a_itf->AssembleInterfaceMatrices(mats);
}

void PoissonSolver::BuildCompROMLinElems()
{
   assert(rom_handler->BasisLoaded());
   assert(rom_elems);

   for (int c = 0; c < topol_handler->GetNumComponents(); c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      BilinearForm a_comp(comp_fes[c]);

      a_comp.AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         a_comp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));

      a_comp.Assemble();
      a_comp.Finalize();

      // Poisson equation has only one solution variable.
      rom_elems->comp[c]->SetSize(1, 1);
      (*rom_elems->comp[c])(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
   }
}

void PoissonSolver::BuildBdrROMLinElems()
{
   assert(rom_handler->BasisLoaded());
   assert(rom_elems);

   for (int c = 0; c < topol_handler->GetNumComponents(); c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      assert(rom_elems->bdr[c]->Size() == comp->bdr_attributes.Size());

      MatrixBlocks *bdr_mat;
      for (int b = 0; b < comp->bdr_attributes.Size(); b++)
      {
         Array<int> bdr_marker(comp->bdr_attributes.Max());
         bdr_marker = 0;
         bdr_marker[comp->bdr_attributes[b] - 1] = 1;
         BilinearForm a_comp(comp_fes[c]);
         a_comp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), bdr_marker);

         a_comp.Assemble();
         a_comp.Finalize();

         bdr_mat = (*rom_elems->bdr[c])[b];
         bdr_mat->SetSize(1, 1);
         (*bdr_mat)(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
      }
   }
}

void PoissonSolver::BuildItfaceROMLinElems()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(rom_handler->BasisLoaded());
   assert(rom_elems);

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   for (int p = 0; p < num_ref_ports; p++)
   {
      assert(rom_elems->port[p]->nrows == 2);
      assert(rom_elems->port[p]->ncols == 2);

      int c1, c2;
      topol_handler->GetComponentPair(p, c1, c2);

      Array<int> c_idx(2);
      c_idx[0] = c1;
      c_idx[1] = c2;

      Array2D<SparseMatrix *> spmats(2,2);
      spmats = NULL;

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      a_itf->AssembleInterfaceMatrixAtPort(p, comp_fes, spmats);

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            (*rom_elems->port[p])(i, j) = rom_handler->ProjectToRefBasis(c_idx[i], c_idx[j], spmats(i,j));

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) delete spmats(i, j);
   }  // for (int p = 0; p < num_ref_ports; p++)
}

bool PoissonSolver::Solve(SampleGenerator *sample_generator)
{
   // If using direct solver, returns always true.
   bool converged = true;

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   // TODO: need to change when the actual parallelization is implemented.
   if (direct_solve)
   {
      assert(mumps);
      mumps->SetPrintLevel(print_level);
      mumps->Mult(*RHS, *U);
   }
   else
   {
      CGSolver *solver = NULL;
      HypreBoomerAMG *M = NULL;
      BlockDiagonalPreconditioner *globalPrec = NULL;
      
      // HypreBoomerAMG makes a meaningful difference in computation time.
      if (use_amg)
      {
         // Initializating HypreParMatrix needs the monolithic sparse matrix.
         assert(globalMat_mono != NULL);

         solver = new CGSolver(MPI_COMM_SELF);

         M = new HypreBoomerAMG(*globalMat_hypre);
         M->SetPrintLevel(print_level);

         solver->SetPreconditioner(*M);
         solver->SetOperator(*globalMat_hypre);
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
      converged = solver->GetConverged();

      // delete the created objects.
      if (use_amg)
      {
         delete M;
      }
      else
      {
         if (globalPrec != NULL) delete globalPrec;
      }
      delete solver;
   }

   /* save solution if sample generator is provided */
   if (converged && sample_generator)
      SaveSnapshots(sample_generator);

   return converged;
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
   /* set up boundary types */
   MultiBlockSolver::SetParameterizedProblem(problem);

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
         case BoundaryType::DIRICHLET:
         { 
            assert(problem->scalar_bdr_ptr[b]);
            AddBCFunction(*(problem->scalar_bdr_ptr[b]), problem->battr[b]);
            break;
         }
         case BoundaryType::NEUMANN: break;
         default:
         case BoundaryType::ZERO:
         { AddBCFunction(0.0, problem->battr[b]); break; }
      }
   }

   if (problem->scalar_rhs_ptr != NULL)
      AddRHSFunction(*(problem->scalar_rhs_ptr));
   else
      AddRHSFunction(0.0);
}

void PoissonSolver::SetMUMPSSolver()
{
   assert(globalMat_hypre);
   mumps = new MUMPSSolver(MPI_COMM_SELF);
   mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
   mumps->SetOperator(*globalMat_hypre);
}