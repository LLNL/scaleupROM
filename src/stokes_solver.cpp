// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "stokes_solver.hpp"
#include "input_parser.hpp"
#include "hdf5_utils.hpp"
#include "linalg_utils.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

StokesSolver::StokesSolver()
   : MultiBlockSolver(), minus_one(-1.0)
{
   nu = config.GetOption<double>("stokes/nu", 1.0);
   nu_coeff = new ConstantCoefficient(nu);

   porder = order;
   uorder = porder + 1;

   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (uorder + 1) * (uorder + 1));
   
   // solution dimension is determined by initialization.
   udim = dim + 1;
   num_var = 2;
   vdim.SetSize(num_var);
   vdim[0] = dim;
   vdim[1] = 1;

   fec.SetSize(num_var);
   // Set up FE collection/spaces.
   if (full_dg)
   {
      fec[0] = new DG_FECollection(uorder, dim);
      fec[1] = new DG_FECollection(porder, dim);
   }
   else
   {
      fec[0] = new H1_FECollection(uorder, dim);
      fec[1] = new H1_FECollection(porder, dim);
   }

   fes.SetSize(num_var * numSub);
   ufes.SetSize(numSub);
   pfes.SetSize(numSub);
   for (int m = 0; m < numSub; m++) {
      ufes[m] = new FiniteElementSpace(meshes[m], fec[0], dim);
      pfes[m] = new FiniteElementSpace(meshes[m], fec[1]);
      // NOTE: ownership is in fes, not ufes and pfes!
      fes[m * num_var] = ufes[m];
      fes[m * num_var + 1] = pfes[m];
   }

   var_names.resize(num_var);
   var_names[0] = "vel";
   var_names[1] = "pres";
}

StokesSolver::~StokesSolver()
{
   delete nu_coeff;
   delete vec_diff;
   delete norm_flux;

   DeletePointers(fs);
   DeletePointers(gs);
   DeletePointers(ms);
   DeletePointers(bs);
   DeletePointers(pms);

   DeletePointers(ud_coeffs);
   DeletePointers(sn_coeffs);

   delete mMat;
   delete bMat;
   delete pmMat;
   delete M;
   delete B;
   delete pM;
   delete systemOp;
   delete Bt;
   delete systemOp_mono;
   delete systemOp_hypre;
   delete mumps;
}

void StokesSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();

   ud_coeffs.SetSize(numBdr);
   sn_coeffs.SetSize(numBdr);
   ud_coeffs = NULL;
   sn_coeffs = NULL;
}

void StokesSolver::AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr)
{
   assert(ud_coeffs.Size() > 0);

   if (battr > 0)
   {
      int idx = global_bdr_attributes.Find(battr);
      if (idx < 0)
      {
         std::string msg = "battr " + std::to_string(battr) + " is not in global boundary attributes. skipping this boundary condition.\n";
         mfem_warning(msg.c_str());
         return;
      }
      ud_coeffs[idx] = new VectorFunctionCoefficient(vdim[0], F);
   }
   else
      for (int k = 0; k < ud_coeffs.Size(); k++)
         ud_coeffs[k] = new VectorFunctionCoefficient(vdim[0], F);

   DeterminePressureDirichlet();
}

void StokesSolver::AddBCFunction(const Vector &F, const int battr)
{
   assert(ud_coeffs.Size() > 0);

   if (battr > 0)
   {
      int idx = global_bdr_attributes.Find(battr);
      if (idx < 0)
      {
         std::string msg = "battr " + std::to_string(battr) + " is not in global boundary attributes. skipping this boundary condition.\n";
         mfem_warning(msg.c_str());
         return;
      }
      ud_coeffs[idx] = new VectorConstantCoefficient(F);
   }
   else
      for (int k = 0; k < ud_coeffs.Size(); k++)
         ud_coeffs[k] = new VectorConstantCoefficient(F);

   DeterminePressureDirichlet();
}

void StokesSolver::InitVariables()
{
   // number of blocks = solution dimension * number of subdomain;
   block_offsets.SetSize(udim * numSub + 1);
   var_offsets.SetSize(num_var * numSub + 1);
   num_vdofs.SetSize(numSub);
   u_offsets.SetSize(numSub + 1);
   p_offsets.SetSize(numSub + 1);

   block_offsets[0] = 0;
   var_offsets[0] = 0;
   u_offsets[0] = 0;
   p_offsets[0] = 0;

   domain_offsets.SetSize(numSub + 1);
   domain_offsets = 0;

   for (int m = 0, block_idx = 1, var_idx=1; m < numSub; m++)
   {
      for (int v = 0; v < num_var; v++, var_idx++)
      {
         FiniteElementSpace *fes = (v == 0) ? ufes[m] : pfes[m];
         var_offsets[var_idx] = fes->GetVSize();
         for (int d = 0; d < vdim[v]; d++, block_idx++)
         {
            block_offsets[block_idx] = fes->GetNDofs();
         }
         domain_offsets[m+1] += fes->GetVSize();
      }
      u_offsets[m + 1] = ufes[m]->GetVSize();
      p_offsets[m + 1] = pfes[m]->GetVSize();
   }
   block_offsets.PartialSum();
   domain_offsets.GetSubArray(1, numSub, num_vdofs);
   var_offsets.PartialSum();
   domain_offsets.PartialSum();
   u_offsets.PartialSum();
   p_offsets.PartialSum();

   // offsets for global block system.
   vblock_offsets.SetSize(3); // number of variables + 1
   vblock_offsets[0] = 0;
   vblock_offsets[1] = u_offsets.Last();
   vblock_offsets[2] = u_offsets.Last() + p_offsets.Last();

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

   us.SetSize(num_var * numSub);
   vels.SetSize(numSub);
   ps.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      vels[m] = new GridFunction(ufes[m], U->GetBlock(num_var * m), 0);
      ps[m] = new GridFunction(pfes[m], U->GetBlock(num_var * m + 1), 0);
      (*vels[m]) = 0.0;
      (*ps[m]) = 0.0;

      // NOTE: ownership is on us, not vels and ps!
      us[m * num_var] = vels[m];
      us[m * num_var + 1] = ps[m];
   }

   f_coeffs.SetSize(0);

   if (use_rom) MultiBlockSolver::InitROMHandler();
}

void StokesSolver::DeterminePressureDirichlet()
{
   pres_dbc = false;

   // If any boundary does not have velocity dirichlet bc profile,
   // then it has pressure dirichlet bc (stress neumann bc).
   for (int b = 0; b < global_bdr_attributes.Size(); b++)
      if (ud_coeffs[b] == NULL) { pres_dbc = true; break; }
}

void StokesSolver::BuildOperators()
{
   BuildRHSOperators();

   BuildDomainOperators();
}

void StokesSolver::BuildRHSOperators()
{
   SanityCheckOnCoeffs();

   fs.SetSize(numSub);
   gs.SetSize(numSub);

   // These are heavily system-dependent.
   // Based on scalar/vector system, different integrators/coefficients will be used.
   for (int m = 0; m < numSub; m++)
   {
      fs[m] = new LinearForm;
      fs[m]->Update(ufes[m], RHS->GetBlock(num_var * m), 0);
      for (int r = 0; r < f_coeffs.Size(); r++)
         fs[m]->AddDomainIntegrator(new VectorDomainLFIntegrator(*f_coeffs[r]));

      gs[m] = new LinearForm;
      gs[m]->Update(pfes[m], RHS->GetBlock(num_var * m + 1), 0);
      // we do not consider non-zero divergence.
      // gs[m]->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   }
}

void StokesSolver::BuildDomainOperators()
{
   SanityCheckOnCoeffs();

   ms.SetSize(numSub);
   bs.SetSize(numSub);

   for (int m = 0; m < numSub; m++)
   {
      ms[m] = new BilinearForm(ufes[m]);
      bs[m] = new MixedBilinearFormDGExtension(ufes[m], pfes[m]);

      ms[m]->AddDomainIntegrator(new VectorDiffusionIntegrator(*nu_coeff));
      if (full_dg)
         ms[m]->AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));

      bs[m]->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
      if (full_dg)
         bs[m]->AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);
   }

   vec_diff = new InterfaceDGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa);
   norm_flux = new InterfaceDGNormalFluxIntegrator;

   // pressure mass matrix for preconditioner.
   pms.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      pms[m] = new BilinearForm(pfes[m]);
      pms[m]->AddDomainIntegrator(new MassIntegrator);
   }
}

bool StokesSolver::BCExistsOnBdr(const int &global_battr_idx)
{
   assert((global_battr_idx >= 0) && (global_battr_idx < global_bdr_attributes.Size()));
   assert(ud_coeffs.Size() == global_bdr_attributes.Size());
   return (ud_coeffs[global_battr_idx]);
}

void StokesSolver::SetupBCOperators()
{
   SetupRHSBCOperators();

   SetupDomainBCOperators();
}

void StokesSolver::SetupRHSBCOperators()
{
   SanityCheckOnCoeffs();

   assert(fs.Size() == numSub);
   assert(gs.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(fs[m] && gs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         // TODO: Non-homogeneous Neumann stress bc
         if (!BCExistsOnBdr(b)) continue;

         fs[m]->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(*ud_coeffs[b], *nu_coeff, sigma, kappa), *bdr_markers[b]);

         // TODO: Non-homogeneous Neumann stress bc
         // fs[m]->AddBdrFaceIntegrator(new BoundaryNormalStressLFIntegrator(*sn_coeffs[b]), p_ess_attr);

         if (full_dg)
            gs[m]->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(*ud_coeffs[b]), *bdr_markers[b]);
         else
            gs[m]->AddBoundaryIntegrator(new DGBoundaryNormalLFIntegrator(*ud_coeffs[b]), *bdr_markers[b]);
      }
   }
}

void StokesSolver::SetupDomainBCOperators()
{
   SanityCheckOnCoeffs();

   assert(ms.Size() == numSub);
   assert(bs.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(ms[m] && bs[m]);
      for (int b = 0; b < global_bdr_attributes.Size(); b++) 
      {
         int idx = meshes[m]->bdr_attributes.Find(global_bdr_attributes[b]);
         if (idx < 0) continue;
         if (!BCExistsOnBdr(b)) continue;

         ms[m]->AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa), *bdr_markers[b]);
         bs[m]->AddBdrFaceIntegrator(new DGNormalFluxIntegrator, *bdr_markers[b]);
      }
   }
}

void StokesSolver::Assemble()
{
   AssembleRHS();

   AssembleOperator();
}

void StokesSolver::AssembleRHS()
{
   SanityCheckOnCoeffs();

   assert(fs.Size() == numSub);
   assert(gs.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(fs[m] && gs[m]);
      fs[m]->Assemble();
      gs[m]->Assemble();

      fs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
      gs[m]->SyncAliasMemory(*RHS);  // Synchronize with block vector RHS. What is different from SyncMemory?
   }
}

void StokesSolver::AssembleOperator()
{
   SanityCheckOnCoeffs();

   assert(ms.Size() == numSub);
   assert(bs.Size() == numSub);

   for (int m = 0; m < numSub; m++)
   {
      assert(ms[m] && bs[m]);
      ms[m]->Assemble();
      bs[m]->Assemble();
   }

   m_mats.SetSize(numSub, numSub);
   b_mats.SetSize(numSub, numSub);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
      {
         if (i == j)
         {
            m_mats(i, i) = &(ms[i]->SpMat());
            b_mats(i, i) = &(bs[i]->SpMat());
         }
         else
         {
            m_mats(i, j) = new SparseMatrix(ufes[i]->GetTrueVSize(), ufes[j]->GetTrueVSize());
            b_mats(i, j) = new SparseMatrix(pfes[i]->GetTrueVSize(), ufes[j]->GetTrueVSize());
         }
      }

   AssembleInterfaceMatrixes();

   for (int m = 0; m < numSub; m++)
   {
      ms[m]->Finalize();
      bs[m]->Finalize();
   }

   // globalMat = new BlockOperator(block_offsets);
   // NOTE: currently, domain-decomposed system will have a significantly different sparsity pattern.
   // This is especially true for vector solution, where ordering of component is changed.
   // This is quite inevitable, but is it desirable?
   // globalMat = new BlockMatrix(domain_offsets);
   mMat = new BlockMatrix(u_offsets);
   bMat = new BlockMatrix(p_offsets, u_offsets);
   for (int i = 0; i < numSub; i++)
   {
      for (int j = 0; j < numSub; j++)
      {
         if (i != j)
         {
            m_mats(i, j)->Finalize();
            b_mats(i, j)->Finalize();
         }

         mMat->SetBlock(i, j, m_mats(i, j));
         bMat->SetBlock(i, j, b_mats(i, j));
      }
   }

   // global block matrix.
   M = mMat->CreateMonolithic();
   B = bMat->CreateMonolithic();
   Bt = Transpose(*B);

   systemOp = new BlockMatrix(vblock_offsets);
   systemOp->SetBlock(0,0, M);
   systemOp->SetBlock(0,1, Bt);
   systemOp->SetBlock(1,0, B);

   if (direct_solve)
   {
      systemOp_mono = systemOp->CreateMonolithic();

      // TODO: need to change when the actual parallelization is implemented.
      sys_glob_size = systemOp_mono->NumRows();
      sys_row_starts[0] = 0;
      sys_row_starts[1] = systemOp_mono->NumRows();
      systemOp_hypre = new HypreParMatrix(MPI_COMM_WORLD, sys_glob_size, sys_row_starts, systemOp_mono);

      mumps = new MUMPSSolver();
      mumps->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
      mumps->SetOperator(*systemOp_hypre);
   }
   else
   {
      // pressure mass matrix for preconditioner.
      pmMat = new BlockMatrix(p_offsets);
      for (int m = 0; m < numSub; m++)
      {
         pms[m]->Assemble();
         pms[m]->Finalize();

         pmMat->SetBlock(m, m, &(pms[m]->SpMat()));
      }
      pM = pmMat->CreateMonolithic();
   }
}

void StokesSolver::AssembleInterfaceMatrixes()
{
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);

      Array<int> midx(2);
      midx[0] = pInfo->Mesh1;
      midx[1] = pInfo->Mesh2;
      Array2D<SparseMatrix *> m_mats_p(2,2), b_mats_p(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            m_mats_p(i, j) = m_mats(midx[i], midx[j]);
            b_mats_p(i, j) = b_mats(midx[i], midx[j]);
         }

      Mesh *mesh1, *mesh2;
      mesh1 = meshes[midx[0]];
      mesh2 = meshes[midx[1]];

      FiniteElementSpace *ufes1, *ufes2, *pfes1, *pfes2;
      ufes1 = ufes[midx[0]];
      ufes2 = ufes[midx[1]];
      pfes1 = pfes[midx[0]];
      pfes2 = pfes[midx[1]];

      Array<InterfaceInfo>* const interface_infos = topol_handler->GetInterfaceInfos(p);
      AssembleInterfaceMatrix(mesh1, mesh2, ufes1, ufes2, vec_diff, interface_infos, m_mats_p);
      AssembleInterfaceMatrix(mesh1, mesh2, ufes1, ufes2, pfes1, pfes2, norm_flux, interface_infos, b_mats_p);
   }  // for (int p = 0; p < topol_handler->GetNumPorts(); p++)
}

void StokesSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = topol_handler->GetNumComponents();
   assert(comp_mats.Size() == num_comp);
   assert(fes_comp.Size() == num_comp * num_var);

   assert(nu_coeff);

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      Mesh *comp = topol_handler->GetComponentMesh(c);

      BilinearForm m_comp(fes_comp[fidx]);
      MixedBilinearFormDGExtension b_comp(fes_comp[fidx], fes_comp[fidx+1]);

      m_comp.AddDomainIntegrator(new VectorDiffusionIntegrator(*nu_coeff));
      if (full_dg)
         m_comp.AddInteriorFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa));

      b_comp.AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
      if (full_dg)
         b_comp.AddInteriorFaceIntegrator(new DGNormalFluxIntegrator);

      m_comp.Assemble();
      b_comp.Assemble();
      m_comp.Finalize();      
      b_comp.Finalize();

      SparseMatrix *m_mat = &(m_comp.SpMat());
      SparseMatrix *b_mat = &(b_comp.SpMat());
      SparseMatrix *bt_mat = Transpose(*b_mat);

      Array<int> dummy1, dummy2;
      BlockMatrix *sys_comp = FormBlockMatrix(m_mat, b_mat, bt_mat, dummy1, dummy2);

      comp_mats[c] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);

      delete bt_mat;
      delete sys_comp;
   }
}

void StokesSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = topol_handler->GetNumComponents();
   assert(bdr_mats.Size() == num_comp);
   assert(fes_comp.Size() == num_comp * num_var);

   assert(nu_coeff);

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      Mesh *comp = topol_handler->GetComponentMesh(c);
      assert(bdr_mats[c]->Size() == comp->bdr_attributes.Size());
      Array<SparseMatrix *> *bdr_mats_c = bdr_mats[c];

      for (int b = 0; b < comp->bdr_attributes.Size(); b++)
      {
         Array<int> bdr_marker(comp->bdr_attributes.Max());
         bdr_marker = 0;
         bdr_marker[comp->bdr_attributes[b] - 1] = 1;

         BilinearForm m_comp(fes_comp[fidx]);
         MixedBilinearFormDGExtension b_comp(fes_comp[fidx], fes_comp[fidx+1]);

         m_comp.AddBdrFaceIntegrator(new DGVectorDiffusionIntegrator(*nu_coeff, sigma, kappa), bdr_marker);
         b_comp.AddBdrFaceIntegrator(new DGNormalFluxIntegrator, bdr_marker);

         m_comp.Assemble();
         b_comp.Assemble();
         m_comp.Finalize();      
         b_comp.Finalize();

         SparseMatrix *m_mat = &(m_comp.SpMat());
         SparseMatrix *b_mat = &(b_comp.SpMat());
         SparseMatrix *bt_mat = Transpose(*b_mat);

         Array<int> dummy1, dummy2;
         BlockMatrix *sys_comp = FormBlockMatrix(m_mat, b_mat, bt_mat, dummy1, dummy2);

         (*bdr_mats_c)[b] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);

         delete bt_mat;
         delete sys_comp;
      }
   }
}

void StokesSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   const int num_comp = topol_handler->GetNumComponents();
   assert(fes_comp.Size() == num_comp * num_var);

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

      Array<int> c_idx(2), f_idx(2);
      c_idx[0] = c1; c_idx[1] = c2;
      f_idx[0] = c1 * num_var;
      f_idx[1] = c2 * num_var;

      Array2D<SparseMatrix *> m_mats_p(2,2), b_mats_p(2,2), bt_mats_p(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            m_mats_p(i, j) = new SparseMatrix(fes_comp[f_idx[i]]->GetTrueVSize(), fes_comp[f_idx[j]]->GetTrueVSize());
            b_mats_p(i, j) = new SparseMatrix(fes_comp[f_idx[i]+1]->GetTrueVSize(), fes_comp[f_idx[j]]->GetTrueVSize());
         }

      Array<InterfaceInfo>* const if_infos = topol_handler->GetRefInterfaceInfos(p);

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      AssembleInterfaceMatrix(&mesh1, &mesh2, fes_comp[f_idx[0]], fes_comp[f_idx[1]],
                              vec_diff, if_infos, m_mats_p);
      AssembleInterfaceMatrix(&mesh1, &mesh2, fes_comp[f_idx[0]], fes_comp[f_idx[1]],
                              fes_comp[f_idx[0]+1], fes_comp[f_idx[1]+1],
                              norm_flux, if_infos, b_mats_p);

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            m_mats_p(i, j)->Finalize();
            b_mats_p(i, j)->Finalize();
            // NOTE: the index also should be transposed.
            bt_mats_p(j, i) = Transpose(*b_mats_p(i, j));
         }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            Array<int> dummy1, dummy2;
            BlockMatrix *tmp_mat = FormBlockMatrix(m_mats_p(i,j), b_mats_p(i,j), bt_mats_p(i,j),
                                                   dummy1, dummy2);
            (*port_mats[p])(i, j) = rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], tmp_mat);
            delete tmp_mat;
         }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            delete m_mats_p(i, j);
            delete b_mats_p(i, j);
            delete bt_mats_p(i, j);
         }
   }  // for (int p = 0; p < num_ref_ports; p++)
}

void StokesSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

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

   if (direct_solve)
   {
      assert(mumps);

      mumps->SetPrintLevel(print_level);
      mumps->Mult(rhs_byvar, sol_byvar);
   }
   else
   {
      HypreParMatrix *Mop = NULL;
      HypreBoomerAMG *amg_prec = NULL;
      GSSmoother *p_prec = NULL;
      OrthoSolver *ortho_p_prec = NULL;
      BlockDiagonalPreconditioner *systemPrec = NULL;

      // TODO: need to change when the actual parallelization is implemented.
      HYPRE_BigInt glob_size = M->NumRows();
      HYPRE_BigInt row_starts[2] = {0, M->NumRows()};

      if (use_amg)
      {
         // velocity amg preconditioner
         Mop = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, M);
         amg_prec = new HypreBoomerAMG(*Mop);
         amg_prec->SetPrintLevel(0);
         amg_prec->SetSystemsOptions(vdim[0], true);

         // pressure mass preconditioner
         p_prec = new GSSmoother(*pM);
         if (!pres_dbc)
         {
            ortho_p_prec = new OrthoSolver;
            ortho_p_prec->SetSolver(*p_prec);
            ortho_p_prec->SetOperator(*pM);
         }

         systemPrec = new BlockDiagonalPreconditioner(vblock_offsets);
         systemPrec->SetDiagonalBlock(0, amg_prec);
         if (pres_dbc)
            systemPrec->SetDiagonalBlock(1, p_prec);
         else
            systemPrec->SetDiagonalBlock(1, ortho_p_prec);
      }

      MINRESSolver solver;
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(maxIter);
      solver.SetOperator(*systemOp);
      if (use_amg)
         solver.SetPreconditioner(*systemPrec);
      solver.SetPrintLevel(print_level);
      solver.Mult(rhs_byvar, sol_byvar);

      if (use_amg)
      {
         delete Mop;
         delete amg_prec;
         delete p_prec;
         if (pres_dbc) delete ortho_p_prec;
         delete systemPrec;
      }
   }

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

void StokesSolver::Solve_obsolete()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-15);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-15);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   BlockVector urhs(u_offsets), prhs(p_offsets);
   BlockVector uvec(u_offsets), pvec(p_offsets);
   urhs = 0.0; prhs = 0.0;
   uvec = 0.0; pvec = 0.0;
   // copy each component of the right-hand side.
   GetVariableVector(0, *RHS, urhs);
   GetVariableVector(1, *RHS, prhs);
   // BlockVector R1(u_offsets);
   Vector R1(urhs.Size());
   R1 = 0.0;

// {
//    PrintMatrix(*M, "stokes.M.txt");
//    PrintMatrix(*B, "stokes.B.txt");

//    PrintVector(urhs, "stokes.urhs.txt");
//    PrintVector(prhs, "stokes.prhs.txt");
// }

   HypreParMatrix *Mop = NULL;
   HypreBoomerAMG *amg_prec = NULL;

   // TODO: need to change when the actual parallelization is implemented.
   HYPRE_BigInt glob_size = M->NumRows();
   HYPRE_BigInt row_starts[2] = {0, M->NumRows()};
   if (use_amg)
   {
      Mop = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, M);
      amg_prec = new HypreBoomerAMG(*Mop);
      amg_prec->SetPrintLevel(print_level);
   }

   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   if (use_amg)
   {
      solver.SetOperator(*Mop);
      solver.SetPreconditioner(*amg_prec);
   }
   else
      solver.SetOperator(*M);
   solver.SetPrintLevel(print_level);
   solver.Mult(urhs, R1);
   if (!solver.GetConverged())
      mfem_error("M^{-1} * urhs fails to converge!\n");

// {
//    SetVariableVector(0, R1, *U);
//    return;
// }

   // B * A^{-1} * F1 - G1
   Vector R2(prhs.Size());
   R2 = 0.0;
   B->Mult(R1, R2);
   R2 -= prhs;

   printf("Set up pressure RHS\n");

   SchurOperator *schur;
   if (use_amg)
      schur = new SchurOperator(Mop, B, use_amg);
   else
      schur = new SchurOperator(M, B, use_amg);
   MINRESSolver solver2;
   solver2.SetOperator(*schur);
   solver2.SetPrintLevel(print_level);
   solver2.SetAbsTol(atol);
   solver2.SetRelTol(rtol);
   solver2.SetMaxIter(maxIter);

   OrthoSolver ortho;
   if (!pres_dbc)
   {
      ortho.SetSolver(solver2);
      ortho.SetOperator(*schur);
      printf("OrthoSolver Set up.\n");
   }

   printf("Solving for pressure\n");
   // printf("%d ?= %d ?= %d\n", R2.Size(), p.Size(), ortho.Height());
   if (pres_dbc)
   {
      solver2.Mult(R2, pvec);
      // if (!solver2.GetConverged())
      //    mfem_error("Pressure Solver fails to converge!\n");
   }
   else
      ortho.Mult(R2, pvec);
   printf("Pressure is solved.\n");

   // AU = F - B^T * P;
   Vector F3(urhs.Size());
   F3 = 0.0;
   B->MultTranspose(pvec, F3);
   F3 *= -1.0;
   F3 += urhs;

   printf("Solving for velocity\n");
   solver.Mult(F3, uvec);
   if (!solver.GetConverged())
      mfem_error("Velocity Solver fails to converge!\n");
   printf("Velocity is solved.\n");

   // Copy back to global vector.
   SetVariableVector(0, uvec, *U);
   SetVariableVector(1, pvec, *U);

   delete schur;
   if (use_amg)
   {
      delete Mop;
      delete amg_prec;
   }
}

void StokesSolver::ProjectOperatorOnReducedBasis()
{
   Array2D<Operator *> tmp(numSub, numSub);
   Array2D<SparseMatrix *> bt_mats(numSub, numSub);
   // NOTE: BlockMatrix offsets are indeed used for its Mult() in ProjectOperatorOnReducedBasis below.
   // Offsets should be stored for multi-component case, until ProjectOperatorOnReducedBasis is done.
   Array2D<Array<int> *> ioffsets(numSub, numSub), joffsets(numSub, numSub);
   for (int i = 0; i < tmp.NumRows(); i++)
      for (int j = 0; j < tmp.NumCols(); j++)
      {
         // NOTE: the index also should be transposed.
         bt_mats(i, j) = Transpose(*b_mats(j, i));

         ioffsets(i, j) = new Array<int>;
         joffsets(i, j) = new Array<int>;
         tmp(i, j) = FormBlockMatrix(m_mats(i,j), b_mats(i,j), bt_mats(i,j),
                                    *(ioffsets(i,j)), *(joffsets(i,j)));
      }
         
   rom_handler->ProjectOperatorOnReducedBasis(tmp);

   for (int i = 0; i < bt_mats.NumRows(); i++)
      for (int j = 0; j < bt_mats.NumCols(); j++)
      {
         delete bt_mats(i, j);
         delete tmp(i, j);
         delete ioffsets(i, j);
         delete joffsets(i, j);
      }
}

void StokesSolver::SanityCheckOnCoeffs()
{
   if ((ud_coeffs.Size() == 0) && (sn_coeffs.Size() == 0))
      MFEM_WARNING("There is no bc coeffcient assigned! Make sure to set bc coefficients before SetupBCOperator.\n");

   bool any_null = false;
   for (int i = 0; i < f_coeffs.Size(); i++)
      if (f_coeffs[i] == NULL)
      {
         any_null = true;
         break;
      }
   if (any_null)
      MFEM_WARNING("Forcing coefficents are assigned but some of them are NULL! Make sure to set rhs coefficients before BuildOperator.\n");

   bool all_null = true;
   for (int i = 0; i < ud_coeffs.Size(); i++)
      if (ud_coeffs[i] != NULL)
      {
         all_null = false;
         break;
      }
   if (all_null)
      MFEM_WARNING("All velocity bc coefficients are NULL, meaning there is no Dirichlet BC. Make sure to set bc coefficients before SetupBCOperator.\n");
}

void StokesSolver::SetParameterizedProblem(ParameterizedProblem *problem)
{
   nu = function_factory::stokes_problem::nu;

   // clean up rhs for parametrized problem.
   if (f_coeffs.Size() > 0)
   {
      for (int k = 0; k < f_coeffs.Size(); k++) delete f_coeffs[k];
      f_coeffs.SetSize(0);
   }
   // clean up boundary functions for parametrized problem.
   ud_coeffs = NULL;
   sn_coeffs = NULL;

   // no-slip dirichlet velocity bc.
   Vector zero(vdim[0]);
   zero = 0.0;

   for (int b = 0; b < problem->battr.Size(); b++)
   {
      switch (problem->bdr_type[b])
      {
         case StokesProblem::BoundaryType::DIRICHLET:
         { 
            assert(problem->vector_bdr_ptr[b]);
            AddBCFunction(*(problem->vector_bdr_ptr[b]), problem->battr[b]);
            break;
         }
         case StokesProblem::BoundaryType::NEUMANN: break;
         default:
         case StokesProblem::BoundaryType::ZERO:
         { AddBCFunction(zero, problem->battr[b]); break; }
      }
   }

   if (problem->vector_rhs_ptr != NULL)
      AddRHSFunction(*(problem->vector_rhs_ptr));
   else
      AddRHSFunction(zero);

   // Ensure incompressibility.
   function_factory::stokes_problem::del_u = 0.0;
   function_factory::stokes_problem::x0.SetSize(dim);
   if (!pres_dbc)
   {
      Array<bool> nz_dbcs(numBdr);
      nz_dbcs = true;
      for (int b = 0; b < problem->battr.Size(); b++)
      {
         if (problem->bdr_type[b] == StokesProblem::BoundaryType::ZERO)
         {
            if (problem->battr[b] == -1)
            { nz_dbcs = false; break; }
            else
               nz_dbcs[b] = false;
         }
      }
      SetComplementaryFlux(nz_dbcs);
   }
}

BlockMatrix* StokesSolver::FormBlockMatrix(
   SparseMatrix* const m, SparseMatrix* const b, SparseMatrix* const bt,
   Array<int> &row_offsets, Array<int> &col_offsets)
{
   assert(m && b && bt);

   const int row1 = m->NumRows();
   const int row2 = b->NumRows();
   const int col1 = m->NumCols();
   const int col2 = bt->NumCols();
   assert(b->NumCols() == col1);
   assert(bt->NumRows() == row1);

   row_offsets.SetSize(3);
   row_offsets[0] = 0;
   row_offsets[1] = row1;
   row_offsets[2] = row2;
   row_offsets.PartialSum();

   col_offsets.SetSize(3);
   col_offsets[0] = 0;
   col_offsets[1] = col1;
   col_offsets[2] = col2;
   col_offsets.PartialSum();

   BlockMatrix *sys_comp = new BlockMatrix(row_offsets, col_offsets);
   sys_comp->SetBlock(0, 0, m);
   sys_comp->SetBlock(1, 0, b);
   sys_comp->SetBlock(0, 1, bt);

   return sys_comp;
}

void StokesSolver::SetComplementaryFlux(const Array<bool> nz_dbcs)
{
   // This routine makes sense only for all velocity dirichlet bc.
   assert(nz_dbcs.Size() == numBdr);
   assert(ud_coeffs.Size() == numBdr);
   for (int k = 0; k < numBdr; k++) assert(ud_coeffs[k]);

   FiniteElementSpace *ufesm = NULL;
   ElementTransformation *eltrans = NULL;
   VectorCoefficient *ud = NULL;
   Mesh *mesh = NULL;

   // initializing complementary flux.
   function_factory::stokes_problem::del_u = 0.0;

   // NOTE: corresponding ParameterizedProblem should use
   // function_factory::stokes_problem::flux
   // to actually enforce the incompressibility.
   Vector *x0 = &(function_factory::stokes_problem::x0);
   x0->SetSize(dim);
   (*x0) = 0.0;
   VectorFunctionCoefficient dir_coeff(dim, function_factory::stokes_problem::dir);

   // Determine the center of domain first.
   Vector x1(dim), dx1(dim);
   x1 = 0.0; dx1 = 0.0;
   double area = 0.0;
   ConstantCoefficient one(1.0);
   for (int m = 0; m < numSub; m++)
   {
      mesh = meshes[m];
      ufesm = ufes[m];

      for (int i = 0; i < ufesm -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         const int global_idx = global_bdr_attributes.Find(bdr_attr);
         if (global_idx < 0) { continue; }

         if (nz_dbcs[global_idx])
         {
            eltrans = ufesm -> GetBdrElementTransformation (i);
            area += ComputeBEIntegral(*ufesm->GetBE(i), *eltrans, one);
            ComputeBEIntegral(*ufesm->GetBE(i), *eltrans, dir_coeff, dx1);
            x1 += dx1;
         }
      }  // for (int i = 0; i < ufesm -> GetNBE(); i++)
   }  // for (int m = 0; m < numSub; m++)
   x1 /= area;

   // set the center of domain for direction function.
   (*x0) = x1;

   // Evaluate boundary flux \int u_d \dot n dA.
   double bflux = 0.0, dirflux = 0.0;
   for (int m = 0; m < numSub; m++)
   {
      mesh = meshes[m];
      ufesm = ufes[m];

      for (int i = 0; i < ufesm -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         const int global_idx = global_bdr_attributes.Find(bdr_attr);
         if (global_idx < 0) { continue; }

         if (nz_dbcs[global_idx])
         {
            ud = ud_coeffs[global_idx];
            eltrans = ufesm -> GetBdrElementTransformation (i);
            bflux += ComputeBEFlux(*ufesm->GetBE(i), *eltrans, *ud);
            dirflux += ComputeBEFlux(*ufesm->GetBE(i), *eltrans, dir_coeff);
         }
      }  // for (int i = 0; i < ufesm -> GetNBE(); i++)
   }  // for (int m = 0; m < numSub; m++)

   // Set the flux to ensure incompressibility.
   function_factory::stokes_problem::del_u = bflux / dirflux;

   // Make sure the resulting flux is zero.
   double threshold = 1.0e-12;
   bflux = 0.0;
   for (int m = 0; m < numSub; m++)
   {
      mesh = meshes[m];
      ufesm = ufes[m];

      for (int i = 0; i < ufesm -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         const int global_idx = global_bdr_attributes.Find(bdr_attr);
         if (global_idx < 0) { continue; }

         if (nz_dbcs[global_idx])
         {
            ud = ud_coeffs[global_idx];
            eltrans = ufesm -> GetBdrElementTransformation (i);
            bflux += ComputeBEFlux(*ufesm->GetBE(i), *eltrans, *ud);
         }
      }  // for (int i = 0; i < ufesm -> GetNBE(); i++)
   }  // for (int m = 0; m < numSub; m++)
   if (abs(bflux) > threshold)
   {
      printf("boundary flux: %.5E\n", bflux);
      mfem_error("Current boundary setup cannot ensure incompressibility!\nMake sure BC uses function_factory::stokes_problem::flux.\n");
   }
   
}

double StokesSolver::ComputeBEFlux(
   const FiniteElement &el, ElementTransformation &Tr,
   VectorCoefficient &ud)
{
   // TODO: support full-dg capability.
   if (full_dg)
      mfem_error("StokesSolver::SetComplementaryFlux does not support full dg discretization yet!\n");

   double bflux = 0.0;
   Vector nor(dim), udvec(dim);

   // int intorder = oa * el.GetOrder() + ob;  // <----------
   int intorder = 2 * el.GetOrder();  // <----------
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      if (dim > 1)
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      else
      {
         nor[0] = 1.0;
      }
      ud.Eval(udvec, Tr, ip);

      bflux += ip.weight * (udvec*nor);
   }

   return bflux;
}

double StokesSolver::ComputeBEIntegral(
   const FiniteElement &el, ElementTransformation &Tr, Coefficient &Q)
{
   // TODO: support full-dg capability.
   if (full_dg)
      mfem_error("StokesSolver::SetComplementaryFlux does not support full dg discretization yet!\n");

   double result = 0.0;

   int intorder = 2*el.GetOrder();
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      
      Tr.SetIntPoint (&ip);
      result += Q.Eval(Tr, ip) * Tr.Weight() * ip.weight;
   }

   return result;
}

void StokesSolver::ComputeBEIntegral(
   const FiniteElement &el, ElementTransformation &Tr,
   VectorCoefficient &Q, Vector &result)
{
   // TODO: support full-dg capability.
   if (full_dg)
      mfem_error("StokesSolver::SetComplementaryFlux does not support full dg discretization yet!\n");

   result.SetSize(dim);
   result = 0.0;
   Vector Qvec(dim);

   int intorder = 2*el.GetOrder();
   const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Q.Eval(Qvec, Tr, ip);
      Tr.SetIntPoint (&ip);
      Qvec *= Tr.Weight() * ip.weight;
      result += Qvec;
   }
}