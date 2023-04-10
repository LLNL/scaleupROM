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

#include "multiblock_solver.hpp"
#include "linalg_utils.hpp"
#include "component_topology_handler.hpp"
// #include <cmath>
// #include <algorithm>

using namespace std;
using namespace mfem;

MultiBlockSolver::MultiBlockSolver()
{
   ParseInputs();

   TopologyData topol_data;
   switch (topol_mode)
   {
      case SUBMESH:
      {
         topol_handler = new SubMeshTopologyHandler();
         break;
      }
      case COMPONENT:
      {
         topol_handler = new ComponentTopologyHandler();
         break;
      }
      default:
      {
         mfem_error("Unknown topology handler mode!\n");
         break;
      }
   }
   topol_handler->ExportInfo(meshes, topol_data);
   
   // Receive topology info
   numSub = topol_data.numSub;
   dim = topol_data.dim;
   global_bdr_attributes = *(topol_data.global_bdr_attributes);
 
   // Set up FE collection/spaces.
   if (full_dg)
   {
      fec = new DG_FECollection(order, dim);
   }
   else
   {
      fec = new H1_FECollection(order, dim);
   }
   
   // solution dimension is determined by initialization.
   udim = 1;
   fes.SetSize(numSub);
   for (int m = 0; m < numSub; m++) {
      fes[m] = new FiniteElementSpace(&(*meshes[m]), fec, udim);
   }

}

MultiBlockSolver::~MultiBlockSolver()
{
   delete U;
   delete RHS;
   delete interface_integ;

   if (save_visual)
      for (int k = 0; k < paraviewColls.Size(); k++) delete paraviewColls[k];

   for (int k = 0; k < bs.Size(); k++) delete bs[k];
   for (int k = 0; k < as.Size(); k++) delete as[k];
   for (int k = 0; k < us.Size(); k++) delete us[k];
   for (int k = 0; k < fes.Size(); k++) delete fes[k];
   for (int k = 0; k < ess_attrs.Size(); k++) delete ess_attrs[k];
   for (int k = 0; k < ess_tdof_lists.Size(); k++) delete ess_tdof_lists[k];

   delete fec;

   for (int k = 0; k < bdr_markers.Size(); k++)
      delete bdr_markers[k];


   for (int k = 0; k < bdr_coeffs.Size(); k++)
      delete bdr_coeffs[k];
      
   for (int k = 0; k < rhs_coeffs.Size(); k++)
      delete rhs_coeffs[k];

   delete rom_handler;
   delete topol_handler;
}

void MultiBlockSolver::ParseInputs()
{
   std::string topol_str = config.GetOption<std::string>("mesh/type", "submesh");
   if (topol_str == "submesh")
   {
      topol_mode = SUBMESH;
   }
   else if (topol_str == "component-wise")
   {
      topol_mode = COMPONENT;
   }
   else
   {
      printf("%s\n", topol_str.c_str());
      mfem_error("Unknown topology handler mode!\n");
   }

   order = config.GetOption<int>("discretization/order", 1);
   full_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);
   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));

   // solver option;
   use_amg = config.GetOption<bool>("solver/use_amg", true);

   save_visual = config.GetOption<bool>("visualization/enabled", false);
   if (save_visual)
   {
      // Default file path if no input file name is provided.
      visual_dir = config.GetOption<std::string>("visualization/file_path/directory", ".");
      visual_prefix = config.GetOption<std::string>("visualization/file_path/prefix", "paraview_output");
   }

   // rom inputs.
   use_rom = config.GetOption<bool>("main/use_rom", false);
}

void MultiBlockSolver::SetupBCVariables()
{
   int numBdr = global_bdr_attributes.Size();
   MFEM_ASSERT(numBdr == bdr_coeffs_in.Size(), "MultiBlockSolver::SetupBoundaryConditions\n");

   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;

   // Boundary conditions are weakly constrained.
   ess_attrs.SetSize(numSub);
   ess_tdof_lists.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      ess_attrs[m] = new Array<int>(meshes[m]->bdr_attributes.Max());
      if (strong_bc) (*ess_attrs[m]) = 0;

      ess_tdof_lists[m] = new Array<int>;
      fes[m]->GetEssentialTrueDofs((*ess_attrs[m]), (*ess_tdof_lists[m]));
   }

   // Set up boundary markers.
   int max_bdr_attr = -1;
   for (int m = 0; m < numSub; m++)
   {
      max_bdr_attr = max(max_bdr_attr, meshes[m]->bdr_attributes.Max());
   }

   bdr_markers.SetSize(max_bdr_attr);
   for (int k = 0; k < max_bdr_attr; k++) {
      bdr_markers[k] = new Array<int>(max_bdr_attr);
      (*bdr_markers[k]) = 0;
      (*bdr_markers[k])[k] = 1;
   }
}

void MultiBlockSolver::AddBCFunction(std::function<double(const Vector &)> F, const int battr)
{
   MFEM_ASSERT(bdr_coeffs.Size() > 0, "MultiBlockSolver::AddBCFunction\n");

   int idx = (battr > 0) ? battr - 1 : 0;
   bdr_coeffs[idx] = new FunctionCoefficient(F);

   if (battr < 0)
      for (int k = 1; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new FunctionCoefficient(F);
}

void MultiBlockSolver::AddBCFunction(const double &F, const int battr)
{
   MFEM_ASSERT(bdr_coeffs.Size() > 0, "MultiBlockSolver::AddBCFunction\n");

   int idx = (battr > 0) ? battr - 1 : 0;
   bdr_coeffs[idx] = new ConstantCoefficient(F);

   if (battr < 0)
      for (int k = 1; k < bdr_coeffs.Size(); k++)
         bdr_coeffs[k] = new ConstantCoefficient(F);
}

void MultiBlockSolver::InitVariables()
{
   // number of blocks = solution dimension * number of subdomain;
   block_offsets.SetSize(udim * numSub + 1);
   domain_offsets.SetSize(numSub + 1);
   num_vdofs.SetSize(numSub);
   block_offsets[0] = 0;
   domain_offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
   {
      domain_offsets[i + 1] = fes[i]->GetTrueVSize();
      num_vdofs[i] = fes[i]->GetTrueVSize();
      for (int d = 0; d < udim; d++)
      {
         block_offsets[d + i * udim + 1] = fes[i]->GetNDofs();
      }
   }
   block_offsets.PartialSum();
   domain_offsets.PartialSum();

   SetupBCVariables();

   // Set up solution/rhs variables/
   U = new BlockVector(domain_offsets);
   RHS = new BlockVector(domain_offsets);
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

   if (use_rom) InitROMHandler();
}

void MultiBlockSolver::BuildOperators()
{
   SanityCheckOnCoeffs();

   bs.SetSize(numSub);
   as.SetSize(numSub);

   double sigma = -1.0;
   double kappa = (order + 1.0) * (order + 1.0);

   // These are heavily system-dependent.
   // Based on scalar/vector system, different integrators/coefficients will be used.
   for (int m = 0; m < numSub; m++)
   {
      bs[m] = new LinearForm(fes[m], RHS->GetBlock(m).GetData());
      for (int r = 0; r < rhs_coeffs.Size(); r++)
         bs[m]->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeffs[r]));

      as[m] = new BilinearForm(fes[m]);
      as[m]->AddDomainIntegrator(new DiffusionIntegrator);
      if (full_dg)
         as[m]->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));
   }

   interface_integ = new InterfaceDGDiffusionIntegrator(sigma, kappa);
}

void MultiBlockSolver::SetupBCOperators()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m] && bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (bdr_coeffs[b] == NULL) continue;

         bs[m]->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(*bdr_coeffs[b], sigma, kappa), *bdr_markers[b]);
         as[m]->AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), *bdr_markers[b]);
      }
   }
}

void MultiBlockSolver::Assemble()
{
   SanityCheckOnCoeffs();

   MFEM_ASSERT(bs.Size() == numSub, "LinearForm bs != numSub.\n");
   MFEM_ASSERT(as.Size() == numSub, "BilinearForm bs != numSub.\n");

   for (int m = 0; m < numSub; m++)
   {
      MFEM_ASSERT(as[m] && bs[m], "LinearForm or BilinearForm pointer of a subdomain is not associated!\n");

      bs[m]->Assemble();
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
   AssembleInterfaceMatrix();

   for (int m = 0; m < numSub; m++)
   {
      // Do we really need SyncAliasMemory?
      bs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
      as[m]->Finalize();
   }

   // globalMat = new BlockOperator(block_offsets);
   // NOTE: currently, domain-decomposed system will have a significantly different sparsity pattern.
   // This is especially true for vector solution, where ordering of component is changed.
   // This is quite inevitable, but is it desirable?
   globalMat = new BlockMatrix(domain_offsets);
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

void MultiBlockSolver::AssembleInterfaceMatrix()
{
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);

      Array<int> midx(2);
      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);
      for (int bn = 0; bn < interface_infos->Size(); bn++)
      {
         InterfaceInfo *if_info = &((*interface_infos)[bn]);
         Mesh *mesh1, *mesh2;
         FiniteElementSpace *fes1, *fes2;
         Array2D<DenseMatrix*> elemmats;
         FaceElementTransformations *tr1, *tr2;
         const FiniteElement *fe1, *fe2;
         Array<Array<int> *> vdofs(2);
         vdofs[0] = new Array<int>;
         vdofs[1] = new Array<int>;

         mesh1 = &(*meshes[midx[0]]);
         mesh2 = &(*meshes[midx[1]]);
         fes1 = fes[midx[0]];
         fes2 = fes[midx[1]];

         topol_handler->GetInterfaceTransformations(mesh1, mesh2, if_info, tr1, tr2);

         if ((tr1 != NULL) && (tr2 != NULL))
         {
            fes1->GetElementVDofs(tr1->Elem1No, *vdofs[0]);
            fes2->GetElementVDofs(tr2->Elem1No, *vdofs[1]);
            // Both domains will have the adjacent element as Elem1.
            fe1 = fes1->GetFE(tr1->Elem1No);
            fe2 = fes2->GetFE(tr2->Elem1No);

            interface_integ->AssembleInterfaceMatrix(*fe1, *fe2, *tr1, *tr2, elemmats);

            for (int i = 0; i < 2; i++) {
               for (int j = 0; j < 2; j++) {
                  mats(midx[i], midx[j])->AddSubMatrix(*vdofs[i], *vdofs[j], *elemmats(i,j), skip_zeros);
               }
            }
         }  // if ((tr1 != NULL) && (tr2 != NULL))
      }  // for (int bn = 0; bn < interface_infos.Size(); bn++)
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void MultiBlockSolver::Solve()
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

      solver->SetOperator(*parGlobalMat);
      M = new HypreBoomerAMG(*parGlobalMat);
      solver->SetPreconditioner(*M);
   }
   else
   {
      solver = new CGSolver();
      solver->SetOperator(*globalMat);
      if (config.GetOption<bool>("solver/block_diagonal_preconditioner", true))
      {
         globalPrec = new BlockDiagonalPreconditioner(domain_offsets);
         solver->SetPreconditioner(*globalPrec);
      }
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
}

void MultiBlockSolver::InitVisualization(const std::string& output_path)
{
   if (!save_visual) return;

   std::string file_prefix;
   if (output_path != "")
      file_prefix = output_path;
   else
   {
      assert(visual_prefix != "");
      assert(visual_dir != "");
      file_prefix = visual_dir + "/" + visual_prefix;
   }

   unified_paraview = config.GetOption<bool>("visualization/unified_paraview", true);

   if (unified_paraview)
      InitUnifiedParaview(file_prefix);
   else
      InitIndividualParaview(file_prefix);
}

void MultiBlockSolver::InitIndividualParaview(const std::string& file_prefix)
{
   paraviewColls.SetSize(numSub);

   for (int m = 0; m < numSub; m++) {
      ostringstream oss;
      // Each subdomain needs to be save separately.
      oss << file_prefix << "_" << std::to_string(m);

      paraviewColls[m] = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
      paraviewColls[m]->SetLevelsOfDetail(order);
      paraviewColls[m]->SetHighOrderOutput(true);
      paraviewColls[m]->SetPrecision(8);

      paraviewColls[m]->RegisterField("solution", us[m]);
      paraviewColls[m]->SetOwnData(false);
   }
}

void MultiBlockSolver::InitUnifiedParaview(const std::string& file_prefix)
{
   // TODO: For truly bottom-up case, when the parent mesh does not exist?
   mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
   paraviewColls.SetSize(1);

   // grid function initialization for visual.
   // TODO: for vector solution.
   Mesh *pmesh = topol_handler->GetGlobalMesh();
   global_fes = new FiniteElementSpace(pmesh, fec, udim);
   global_us_visual = new GridFunction(global_fes);

   paraviewColls[0] = new ParaViewDataCollection(file_prefix.c_str(), pmesh);
   paraviewColls[0]->SetLevelsOfDetail(order);
   paraviewColls[0]->SetHighOrderOutput(true);
   paraviewColls[0]->SetPrecision(8);

   paraviewColls[0]->RegisterField("solution", global_us_visual);
   paraviewColls[0]->SetOwnData(false);
}

void MultiBlockSolver::SaveVisualization()
{
   if (!save_visual) return;

   if (unified_paraview)
   {
      mfem_warning("Paraview is unified. Any overlapped interface dof data will not be shown.\n");
      topol_handler->TransferToGlobal(us, global_us_visual);
   }

   for (int m = 0; m < paraviewColls.Size(); m++)
      paraviewColls[m]->Save();
};

void MultiBlockSolver::InitROMHandler()
{
   std::string rom_handler_str = config.GetOption<std::string>("model_reduction/rom_handler_type", "base");

   if (rom_handler_str == "base")
   {
      rom_handler = new ROMHandler(numSub, udim, num_vdofs);
   }
   else if (rom_handler_str == "mfem")
   {
      rom_handler = new MFEMROMHandler(numSub, udim, num_vdofs);
   }
   else
   {
      mfem_error("Unknown ROM handler type!\n");
   }
}

double MultiBlockSolver::CompareSolution()
{
   // Copy the rom solution.
   BlockVector romU(domain_offsets);
   romU = *U;
   Array<GridFunction *> rom_us;
   Array<VectorGridFunctionCoefficient *> rom_u_coeffs;
   ConstantCoefficient zero(0.0);
   rom_us.SetSize(numSub);
   rom_u_coeffs.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      rom_us[m] = new GridFunction(fes[m], romU.GetBlock(m), 0);

      // BC's are weakly constrained and there is no essential dofs.
      // Does this make any difference?
      rom_us[m]->SetTrueVector();

      rom_u_coeffs[m] = new VectorGridFunctionCoefficient(rom_us[m]);
   }

   // TODO: right now we solve the full-order system to compare the solution.
   // Need to implement loading the fom solution file?
   StopWatch solveTimer;
   solveTimer.Start();
   Solve();
   solveTimer.Stop();
   printf("FOM-solve time: %f seconds.\n", solveTimer.RealTime());

   // Compare the solution.
   double norm = 0.0;
   double error = 0.0;
   for (int m = 0; m < numSub; m++)
   {
      norm += us[m]->ComputeLpError(2, zero);
      error += us[m]->ComputeLpError(2, *rom_u_coeffs[m]);
   }
   error /= norm;
   printf("Relative error: %.5E\n", error);

   for (int m = 0; m < numSub; m++)
   {
      delete rom_us[m];
      delete rom_u_coeffs[m];
   }

   return error;
}

void MultiBlockSolver::SanityCheckOnCoeffs()
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

   all_null = false;
   for (int i = 0; i < bdr_coeffs.Size(); i++)
      if (bdr_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All bc coefficients are NULL, meaning there is no Dirichlet BC. Make sure to set bc coefficients before SetupBCOperator.\n");
}
