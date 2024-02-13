// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "linelast_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

LinElastSolver::LinElastSolver()
    : MultiBlockSolver()
{
   alpha = config.GetOption<double>("discretization/interface/alpha", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));

   var_names = GetVariableNames();
   num_var = var_names.size();

   // solution dimension is determined by initialization.
   udim = dim;
   vdim.SetSize(num_var);
   vdim = dim;

   // Set up FE collection/spaces.
   fec.SetSize(num_var);
   if (full_dg)
   {
      fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }

   fes.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      fes[m] = new FiniteElementSpace(meshes[m], fec[0], udim);
   }

   init_x = new VectorFunctionCoefficient(dim, InitDisplacement);
}

LinElastSolver::~LinElastSolver()
{
   delete a_itf;

   DeletePointers(bs);
   DeletePointers(as);
   DeletePointers(lambda_cs);
   DeletePointers(mu_cs);

   delete globalMat_mono;
   delete globalMat;
   delete globalMat_hypre;
   delete mumps;
}

void LinElastSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();

   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;
}

void LinElastSolver::SetupMaterialVariables()
{
   int max_bdr_attr = -1; // A bit redundant...
   for (int m = 0; m < numSub; m++)
   {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
   }

   // Set up the Lame constants for the two materials.
   Vector lambda(max_bdr_attr);
   lambda = 1.0;     // Set lambda = 1 for all element attributes.
   lambda(0) = 50.0; // Set lambda = 50 for element attribute 1.
   // PWConstCoefficient lambda_c(lambda);

   Vector mu(max_bdr_attr);
   mu = 1.0;     // Set mu = 1 for all element attributes.
   mu(0) = 50.0; // Set mu = 50 for element attribute 1.
   // PWConstCoefficient mu_c(mu);

   lambda_cs.SetSize(numSub);
   mu_cs.SetSize(numSub);

   for (int m = 0; m < numSub; m++)
   {
      // lambda_cs[m] = &lambda_c;
      // mu_cs[m] = &mu_c;
      lambda_cs[m] = new PWConstCoefficient(lambda);
      mu_cs[m] = new PWConstCoefficient(mu);
   }
}

void LinElastSolver::InitVariables()
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

   MultiBlockSolver::SetupBCVariables();
   SetupMaterialVariables();

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
      us[m]->ProjectCoefficient(*init_x);

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      us[m]->SetTrueVector();
   }

   // if (use_rom)  //Off for now
   //   MultiBlockSolver::InitROMHandler();
}

void LinElastSolver::BuildOperators()
{
   BuildRHSOperators();
   BuildDomainOperators();
}

void LinElastSolver::BuildRHSOperators()
{
   bs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m).GetData());
   }
}

void LinElastSolver::SetupRHSBCOperators()
{
   assert(bs.Size() == numSub);
   for (int m = 0; m < numSub; m++)
   {
      assert(bs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++)
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0)
            continue;
         if (!BCExistsOnBdr(b))
            continue;

         // bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdr_coeffs[b], sigma, kappa), *bdr_markers[b]);
         // bs[m]->AddBdrFaceIntegrator(new DGElasticityDirichletLFIntegrator(*init_x, *lambda_cs[b], *mu_cs[b], alpha, kappa), *bdr_markers[b]);
         bs[m]->AddBdrFaceIntegrator(new DGElasticityDirichletLFIntegrator(*bdr_coeffs[b], *lambda_cs[b], *mu_cs[b], alpha, kappa), *bdr_markers[b]);
      }
   }
}

void LinElastSolver::BuildDomainOperators()
{
   // SanityCheckOnCoeffs();
   as.SetSize(numSub);

   for (int m = 0; m < numSub; m++)
   {
      as[m] = new BilinearForm(fes[m]);
      as[m]->AddDomainIntegrator(new ElasticityIntegrator(*(lambda_cs[m]), *(mu_cs[m])));

      if (full_dg)
      {
         as[m]->AddInteriorFaceIntegrator(
             new DGElasticityIntegrator(*(lambda_cs[m]), *(mu_cs[m]), alpha, kappa));

         for (int b = 0; b < global_bdr_attributes.Size(); b++)
         {
            int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
            if (idx < 0)
               continue;
            if (!BCExistsOnBdr(b))
               continue;

            as[m]->AddBdrFaceIntegrator(new DGElasticityDirichletLFIntegrator(*bdr_coeffs[b], *lambda_cs[b], *mu_cs[b], alpha, kappa), *bdr_markers[b]);
         }
      }
      as[m]->Assemble();
      as[m]->Finalize();
   }

   a_itf = new InterfaceForm(meshes, fes, topol_handler);
   a_itf->AddIntefaceIntegrator(new InterfaceDGElasticityIntegrator(lambda_cs[0], mu_cs[0], alpha, kappa));
}

void LinElastSolver::Assemble()
{
   AssembleRHS();
   AssembleOperator();
}

void LinElastSolver::AssembleRHS()
{
   // SanityCheckOnCoeffs();
   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");
   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      cout << bs[m]->GetFLFI()[0] << endl;
      bs[m]->Assemble();
      cout << "linear form norm: " << bs[m]->Norml2() << endl;
   }

   for (int m = 0; m < numSub; m++)
      // Do we really need SyncAliasMemory?
      bs[m]->SyncAliasMemory(*RHS); // Synchronize with block vector RHS. What is different from SyncMemory?
}

void LinElastSolver::AssembleOperator()
{
   // SanityCheckOnCoeffs();
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");
   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      as[m]->Assemble();
      as[m]->Finalize();
      double binorm = as[m]->SpMat().ToDenseMatrix()->FNorm();

      cout << "bilinear form norm: " << binorm << endl;
   }
   mats.SetSize(numSub, numSub);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i == j)
         {
            mats(i, i) = &(as[i]->SpMat());
         }
         else
         {
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
         if (i != j)
            mats(i, j)->Finalize();

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
      globalMat_hypre = new HypreParMatrix(MPI_COMM_WORLD, sys_glob_size, sys_row_starts, globalMat_mono);

      mumps = new MUMPSSolver();
      mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
      mumps->SetOperator(*globalMat_hypre);
   }
}

void LinElastSolver::AssembleInterfaceMatrixes()
{
   assert(a_itf);
   a_itf->AssembleInterfaceMatrixes(mats);
}

bool LinElastSolver::Solve()
{
   // If using direct solver, returns always true.
   bool converged = true;

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);
   for (size_t i = 0; i < U->NumBlocks(); i++)
   {
      cout << "Unorm " << i << ": " << U->GetBlock(i).Norml2() << endl;
   }
   cout << "RHSnorm: " << RHS->Norml2() << endl;
   // TODO: need to change when the actual parallelization is implemented.
   cout << "direct_solve is: " << direct_solve << endl;
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

         solver = new CGSolver(MPI_COMM_WORLD);

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
         if (globalPrec != NULL)
            delete globalPrec;
      }
      delete solver;
   }

   return converged;
}

void PrintVector(string filename, Vector &vec)
{
   std::ofstream outfile(filename);
   double tol = 1e-7;
   double val = 0.0;
   for (size_t i = 0; i < vec.Size(); i++)
   {
      val = vec[i];
      if (abs(val) < tol)
      {
         val = 0.0;
      }

      outfile << setprecision(8) << val << endl;
   }
   outfile.close();
   cout << "done printing vector" << endl;
}

void PrintMatrix(string filename, DenseMatrix &mat)
{
   std::ofstream outfile(filename);

   double tol = 1e-7;
   double val = 0.0;

   for (size_t i = 0; i < mat.Height(); i++)
   {
      for (size_t j = 0; j < mat.Width(); j++)
      {
         val = mat(i, j);
         if (abs(val) < tol)
         {
            val = 0.0;
         }

         outfile << setprecision(8) << val << " ";
      }
      outfile << endl;
   }
   outfile.close();
   cout << "done printing matrix" << endl;
}

void PrintBlockVector(string filename, BlockVector &bvec)
{
   std::ofstream outfile(filename);

   for (size_t i = 0; i < bvec.GetBlock(0).Size(); i++)
   {
      for (size_t j = 0; j < bvec.NumBlocks(); j++)
      {
         outfile << setprecision(1) << bvec.GetBlock(j)[i] << " ";
      }
      outfile << endl;
   }
   outfile.close();
   cout << "done printing blockvector" << endl;
}

void LinElastSolver::PrintOperators()
{
   PrintMatrix("scaleuprom_a.txt", *(as[0]->SpMat().ToDenseMatrix()));
   PrintVector("scaleuprom_b.txt", *bs[0]);
}

void LinElastSolver::SetupBCVariables() { "LinElastSolver::SetupBCVariables is not implemented yet!\n"; }
void LinElastSolver::AddBCFunction(std::function<double(const Vector &)> F, const int battr) { "LinElastSolver::AddBCFunction is not implemented yet!\n"; }
void LinElastSolver::AddBCFunction(const double &F, const int battr) { "LinElastSolver::AddBCFunction is not implemented yet!\n"; }
bool LinElastSolver::BCExistsOnBdr(const int &global_battr_idx)
{
   std::cout << "LinElastSolver::BCExistsOnBdr is not implemented yet!\n";
   return false;
}

void LinElastSolver::SetupBCOperators()
{
   SetupRHSBCOperators();
   SetupDomainBCOperators();
}

void LinElastSolver::SetupRHSBCOperators() { "LinElastSolver::SetupRHSBCOperators is not implemented yet!\n"; }
void LinElastSolver::SetupDomainBCOperators() { "LinElastSolver::SetupDomainBCOperators is not implemented yet!\n"; }

// Component-wise assembly
void LinElastSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp) { "LinElastSolver::BuildCompROMElement is not implemented yet!\n"; }
void LinElastSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp) { "LinElastSolver::BuildBdrROMElement is not implemented yet!\n"; }
void LinElastSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp) { "LinElastSolver::BuildInterfaceROMElement is not implemented yet!\n"; }

void LinElastSolver::ProjectOperatorOnReducedBasis() { "LinElastSolver::ProjectOperatorOnReducedBasis is not implemented yet!\n"; }

void LinElastSolver::SanityCheckOnCoeffs() { "LinElastSolver::SanityCheckOnCoeffs is not implemented yet!\n"; }

void LinElastSolver::SetParameterizedProblem(ParameterizedProblem *problem) { "LinElastSolver::SetParameterizedProblem is not implemented yet!\n"; }
