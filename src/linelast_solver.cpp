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
}

LinElastSolver::~LinElastSolver()
{
   delete a_itf;

   DeletePointers(bs);
   DeletePointers(as);
   DeletePointers(lambda_c);
   DeletePointers(mu_c);

   delete globalMat_mono;
   delete globalMat;
   delete globalMat_hypre;
   delete mumps;
   delete init_x;
   
}

void LinElastSolver::SetupIC(std::function<void(const Vector &, Vector &)> F)
{
   init_x = new VectorFunctionCoefficient(dim, F);
   for (int m = 0; m < numSub; m++)
   {
      assert(us[m]);
      us[m]->ProjectCoefficient(*init_x);
   }
}

void LinElastSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();
   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;

   type_idx.SetSize(numBdr);
   type_idx = LinElastProblem::BoundaryType::DIRICHLET;

   lambda_c.SetSize(numSub);
   lambda_c = NULL;

   mu_c.SetSize(numSub);
   mu_c = NULL;

   for (size_t i = 0; i < numSub; i++)
   {
      lambda_c[i] = new ConstantCoefficient(1.0);
      mu_c[i] = new ConstantCoefficient(1.0);
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

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      us[m]->SetTrueVector();
   }
   if (use_rom)
     MultiBlockSolver::InitROMHandler();
}

void LinElastSolver::BuildOperators()
{
   BuildRHSOperators();
   BuildDomainOperators();
}

bool LinElastSolver::BCExistsOnBdr(const int &global_battr_idx)
{
   assert((global_battr_idx >= 0) && (global_battr_idx < global_bdr_attributes.Size()));
   assert(bdr_coeffs.Size() == global_bdr_attributes.Size());
   return (bdr_coeffs[global_battr_idx]);
}

void LinElastSolver::BuildRHSOperators()
{
   bs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m).GetData());
      for (int r = 0; r < rhs_coeffs.Size(); r++)
      {
         bs[m]->AddDomainIntegrator(new VectorDomainLFIntegrator(*rhs_coeffs[r]));
      }
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

         if (type_idx[b] == LinElastProblem::BoundaryType::NEUMANN)
         {
            bs[m]->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(*bdr_coeffs[b]), *bdr_markers[b]);
         }  

         if (!BCExistsOnBdr(b))
            continue;

         if (type_idx[b] == LinElastProblem::BoundaryType::DIRICHLET) // Default behavior
         { 
            bs[m]->AddBdrFaceIntegrator(new DGElasticityDirichletLFIntegrator(*bdr_coeffs[b], *lambda_c[m], *mu_c[m], alpha, kappa), *bdr_markers[b]);
         }
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
      as[m]->AddDomainIntegrator(new ElasticityIntegrator(*(lambda_c[m]), *(mu_c[m])));

      if (full_dg)
      {
         as[m]->AddInteriorFaceIntegrator(
             new DGElasticityIntegrator(*(lambda_c[m]), *(mu_c[m]), alpha, kappa));
      }
   }

   a_itf = new InterfaceForm(meshes, fes, topol_handler); // TODO: Is this reasonable?
   a_itf->AddIntefaceIntegrator(new InterfaceDGElasticityIntegrator(lambda_c[0], mu_c[0], alpha, kappa));
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
      bs[m]->Assemble();
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

void LinElastSolver::AssembleInterfaceMatrices()
{
   assert(a_itf);
   a_itf->AssembleInterfaceMatrices(mats);
}

bool LinElastSolver::Solve()
{
   // If using direct solver, returns always true.
   bool converged = true;

   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

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

void LinElastSolver::AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr)
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
      bdr_coeffs[idx] = new VectorFunctionCoefficient(dim, F);
   }
   else
      for (int k = 0; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new VectorFunctionCoefficient(dim, F);
}

void LinElastSolver::AddRHSFunction(std::function<void(const Vector &, Vector &)> F)
{
   rhs_coeffs.Append(new VectorFunctionCoefficient(dim, F));
}

void LinElastSolver::SetupBCOperators()
{
   SetupRHSBCOperators();
   SetupDomainBCOperators();
}

void LinElastSolver::SetupDomainBCOperators()
{
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");
   if (full_dg)
   {
      for (int m = 0; m < numSub; m++)
      {
         for (int b = 0; b < global_bdr_attributes.Size(); b++)
         {
            int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
            if (idx < 0)
               continue;
            if (!BCExistsOnBdr(b))
               continue;
            as[m]->AddBdrFaceIntegrator(new DGElasticityIntegrator(*(lambda_c[m]), *(mu_c[m]), alpha, kappa), *(bdr_markers[b]));
         }
      }
   }
}

void LinElastSolver::SetParameterizedProblem(ParameterizedProblem *problem)
{
   /* set up boundary types */
   MultiBlockSolver::SetParameterizedProblem(problem);

   // Set materials
   lambda_c.SetSize(numSub);
   lambda_c = NULL;

   mu_c.SetSize(numSub);
   mu_c = NULL;

   Vector _x(1);

   for (size_t i = 0; i < numSub; i++)
   {
      double lambda_i = (problem->general_scalar_ptr[0])(_x);
      lambda_c[i] = new ConstantCoefficient(lambda_i);

      double mu_i = (problem->general_scalar_ptr[1])(_x);
      mu_c[i] = new ConstantCoefficient(mu_i);
   }

   // Set BCs, the switch on BC type is done inside SetupRHSBCOperators
   for (int b = 0; b < problem->battr.Size(); b++)
   {
      switch (problem->bdr_type[b])
      {
         case ParameterizedProblem::BoundaryType::DIRICHLET:
         case ParameterizedProblem::BoundaryType::NEUMANN:
            assert(problem->vector_bdr_ptr[b]);
            AddBCFunction(*(problem->vector_bdr_ptr[b]), problem->battr[b]);
            break;
         default:
            break;
      }
   }

   // Set RHS
   if (problem->vector_rhs_ptr != NULL){
      AddRHSFunction(*(problem->vector_rhs_ptr));
   }

   // Add initial condition
   if (problem->general_vector_ptr[0] != NULL)
   {
      SetupIC(*(problem->general_vector_ptr[0]));
   }
}

void LinElastSolver::ProjectOperatorOnReducedBasis()
{ 
   Array2D<Operator *> tmp(mats.NumRows(), mats.NumCols());
   for (int i = 0; i < tmp.NumRows(); i++)
      for (int j = 0; j < tmp.NumCols(); j++)
         tmp(i, j) = mats(i, j);
         
   rom_handler->ProjectOperatorOnReducedBasis(tmp);
}

// Component-wise assembly
void LinElastSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = fes_comp.Size();
   assert(comp_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      BilinearForm a_comp(fes_comp[c]);

      a_comp.AddDomainIntegrator(new ElasticityIntegrator(*(lambda_c[c]), *(mu_c[c])));
      if (full_dg)
         a_comp.AddInteriorFaceIntegrator(new DGElasticityIntegrator(*(lambda_c[c]), *(mu_c[c]), alpha, kappa));

      a_comp.Assemble();
      a_comp.Finalize();

      // Elasticity equation has only one solution variable.
      comp_mats[c]->SetSize(1, 1);
      (*comp_mats[c])(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
   }
}

void LinElastSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = fes_comp.Size();
   assert(bdr_mats.Size() == num_comp);

   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      assert(bdr_mats[c]->Size() == comp->bdr_attributes.Size());

      MatrixBlocks *bdr_mat;
      for (int b = 0; b < comp->bdr_attributes.Size(); b++)
      {
         Array<int> bdr_marker(comp->bdr_attributes.Max());
         bdr_marker = 0;
         bdr_marker[comp->bdr_attributes[b] - 1] = 1;
         BilinearForm a_comp(fes_comp[c]);
         a_comp.AddBdrFaceIntegrator(new DGElasticityIntegrator(*(lambda_c[c]), *(mu_c[c]), alpha, kappa), bdr_marker);
         
         a_comp.Assemble();
         a_comp.Finalize();

         bdr_mat = (*bdr_mats[c])[b];
         bdr_mat->SetSize(1, 1);
         (*bdr_mat)(0, 0) = rom_handler->ProjectToRefBasis(c, c, &(a_comp.SpMat()));
      }
   }
}

void LinElastSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   assert(port_mats.Size() == num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
   {
      assert(port_mats[p]->nrows == 2);
      assert(port_mats[p]->ncols == 2);

      int c1, c2;
      topol_handler->GetComponentPair(p, c1, c2);

      Array<int> c_idx(2);
      c_idx[0] = c1;
      c_idx[1] = c2;

      Array2D<SparseMatrix *> spmats(2,2);
      spmats = NULL;

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      a_itf->AssembleInterfaceMatrixAtPort(p, fes_comp, spmats);

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            (*port_mats[p])(i, j) = rom_handler->ProjectToRefBasis(c_idx[i], c_idx[j], spmats(i,j));

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) delete spmats(i, j);
   }  // for (int p = 0; p < num_ref_ports; p++)
}
void LinElastSolver::SanityCheckOnCoeffs() { "LinElastSolver::SanityCheckOnCoeffs is not implemented yet!\n"; }
