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

#include "stokes_solver.hpp"
#include "input_parser.hpp"
#include "hdf5_utils.hpp"
#include "linalg_utils.hpp"
#include "dg_bilinear.hpp"
#include "dg_linear.hpp"

using namespace std;
using namespace mfem;

void uFun_ex_tmp(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   assert(x.Size() == 2);

   u(0) = cos(xi)*sin(yi);
   u(1) = - sin(xi)*cos(yi);
}

StokesSolver::StokesSolver()
   : MultiBlockSolver(), minus_one(-1.0)
{
   nu = config.GetOption<double>("stokes/nu", 1.0);
   nu_coeff = new ConstantCoefficient(nu);

   porder = order;
   uorder = porder + 1;

   sigma = config.GetOption<double>("discretization/interface/sigma", -1.0);
   kappa = config.GetOption<double>("discretization/interface/kappa", (uorder + 1) * (uorder + 1));
 
   // Set up FE collection/spaces.
   if (full_dg)
   {
      ufec = new DG_FECollection(uorder, dim);
      pfec = new DG_FECollection(porder, dim);
   }
   else
   {
      ufec = new H1_FECollection(uorder, dim);
      pfec = new H1_FECollection(porder, dim);
   }
   
   // solution dimension is determined by initialization.
   udim = dim + 1;
   num_var = 2;
   vdim.SetSize(num_var);
   vdim[0] = dim;
   vdim[1] = 1;

   ufes.SetSize(numSub);
   pfes.SetSize(numSub);
   for (int m = 0; m < numSub; m++) {
      ufes[m] = new FiniteElementSpace(meshes[m], ufec, dim);
      pfes[m] = new FiniteElementSpace(meshes[m], pfec);
   }
}

StokesSolver::~StokesSolver()
{
   delete nu_coeff;
   delete vec_diff, norm_flux;

   for (int k = 0; k < fs.Size(); k++) delete fs[k];
   for (int k = 0; k < gs.Size(); k++) delete gs[k];
   for (int k = 0; k < ms.Size(); k++) delete ms[k];
   for (int k = 0; k < bs.Size(); k++) delete bs[k];
   for (int k = 0; k < ufes.Size(); k++) delete ufes[k];
   for (int k = 0; k < pfes.Size(); k++) delete pfes[k];

   delete ufec, pfec;

   for (int k = 0; k < ud_coeffs.Size(); k++)
      delete ud_coeffs[k];
      
   for (int k = 0; k < sn_coeffs.Size(); k++)
      delete sn_coeffs[k];

   // for (int c = 0; c < comp_mats.Size(); c++)
   //    delete comp_mats[c];

   // for (int c = 0; c < bdr_mats.Size(); c++)
   // {
   //    for (int b = 0; b < bdr_mats[c]->Size(); b++)
   //       delete (*bdr_mats[c])[b];

   //    delete bdr_mats[c];
   // }

   // for (int p = 0; p < port_mats.Size(); p++)
   // {
   //    for (int i = 0; i < port_mats[p]->NumRows(); i++)
   //       for (int j = 0; j < port_mats[p]->NumCols(); j++)
   //          delete (*port_mats[p])(i,j);

   //    delete port_mats[p];
   // }
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

   int idx = (battr > 0) ? battr - 1 : 0;
   ud_coeffs[idx] = new VectorFunctionCoefficient(vdim[0], F);

   if (battr < 0)
      for (int k = 1; k < ud_coeffs.Size(); k++)
         ud_coeffs[k] = new VectorFunctionCoefficient(vdim[0], F);
}

void StokesSolver::AddBCFunction(const Vector &F, const int battr)
{
   assert(ud_coeffs.Size() > 0);

   int idx = (battr > 0) ? battr - 1 : 0;
   ud_coeffs[idx] = new VectorConstantCoefficient(F);

   if (battr < 0)
      for (int k = 1; k < ud_coeffs.Size(); k++)
         ud_coeffs[k] = new VectorConstantCoefficient(F);
}

void StokesSolver::InitVariables()
{
   // number of blocks = solution dimension * number of subdomain;
   block_offsets.SetSize(udim * numSub + 1);
   domain_offsets.SetSize(num_var * numSub + 1);
   num_vdofs.SetSize(num_var * numSub);
   u_offsets.SetSize(numSub + 1);
   p_offsets.SetSize(numSub + 1);

   block_offsets[0] = 0;
   domain_offsets[0] = 0;
   u_offsets[0] = 0;
   p_offsets[0] = 0;

   for (int m = 0, block_idx = 1, domain_idx=1; m < numSub; m++)
   {
      for (int v = 0; v < num_var; v++, domain_idx++)
      {
         FiniteElementSpace *fes = (v == 0) ? ufes[m] : pfes[m];
         domain_offsets[domain_idx] = fes->GetVSize();
         for (int d = 0; d < vdim[v]; d++, block_idx++)
         {
            block_offsets[block_idx] = fes->GetNDofs();
         }
         // var_offsets[v+1] += fes->GetVSize();
      }
      u_offsets[m + 1] = ufes[m]->GetVSize();
      p_offsets[m + 1] = pfes[m]->GetVSize();
   }
   block_offsets.PartialSum();
   domain_offsets.GetSubArray(1, num_var * numSub, num_vdofs);
   domain_offsets.PartialSum();
   u_offsets.PartialSum();
   p_offsets.PartialSum();

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
   ps.SetSize(numSub);
   for (int m = 0; m < numSub; m++)
   {
      us[m] = new GridFunction(ufes[m], U->GetBlock(num_var * m), 0);
      ps[m] = new GridFunction(pfes[m], U->GetBlock(num_var * m + 1), 0);
      (*us[m]) = 0.0;
      (*ps[m]) = 0.0;
   }

   f_coeffs.SetSize(0);

   if (use_rom)
      mfem_error("StokesSolver does not support ROM yet!\n");
   // if (use_rom) MultiBlockSolver::InitROMHandler();
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
}

void StokesSolver::SetupBCOperators()
{
   SetupRHSBCOperators();

   SetupDomainBCOperators();

   // If any boundary does not have velocity dirichlet bc profile,
   // then it has pressure dirichlet bc (stress neumann bc).
   for (int b = 0; b < global_bdr_attributes.Size(); b++)
      if (ud_coeffs[b] == NULL) { pres_dbc = true; break; }
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
         if (ud_coeffs[b] == NULL) continue;

         fs[m]->AddBdrFaceIntegrator(new DGVectorDirichletLFIntegrator(*ud_coeffs[b], *nu_coeff, sigma, kappa), *bdr_markers[b]);

         // TODO: Non-homogeneous Neumann stress bc
         // fs[m]->AddBdrFaceIntegrator(new BoundaryNormalStressLFIntegrator(*sn_coeffs[b]), p_ess_attr);

         gs[m]->AddBdrFaceIntegrator(new DGBoundaryNormalLFIntegrator(*ud_coeffs[b]), *bdr_markers[b]);
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
         if (ud_coeffs[b] == NULL) continue;

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

   M = mMat->CreateMonolithic();
   B = bMat->CreateMonolithic();
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

// void PoissonSolver::AllocateROMElements()
// {
//    assert(topol_mode == TopologyHandlerMode::COMPONENT);
//    const TrainMode train_mode = rom_handler->GetTrainMode();
//    assert(train_mode == UNIVERSAL);

//    const int num_comp = topol_handler->GetNumComponents();
//    const int num_ref_ports = topol_handler->GetNumRefPorts();

//    comp_mats.SetSize(num_comp);
//    bdr_mats.SetSize(num_comp);
//    for (int c = 0; c < num_comp; c++)
//    {
//       comp_mats[c] = new DenseMatrix();

//       Mesh *comp = topol_handler->GetComponentMesh(c);
//       bdr_mats[c] = new Array<DenseMatrix *>(comp->bdr_attributes.Size());
//       for (int b = 0; b < bdr_mats[c]->Size(); b++)
//          (*bdr_mats[c])[b] = new DenseMatrix();
//    }
//    port_mats.SetSize(num_ref_ports);
//    for (int p = 0; p < num_ref_ports; p++)
//    {
//       port_mats[p] = new Array2D<DenseMatrix *>(2,2);

//       for (int i = 0; i < 2; i++)
//          for (int j = 0; j < 2; j++) (*port_mats[p])(i,j) = new DenseMatrix();
//    }
// }

// void PoissonSolver::BuildROMElements()
// {
//    assert(topol_mode == TopologyHandlerMode::COMPONENT);
//    const TrainMode train_mode = rom_handler->GetTrainMode();
//    assert(train_mode == UNIVERSAL);
//    assert(rom_handler->BasisLoaded());

//    // Component domain system
//    const int num_comp = topol_handler->GetNumComponents();
//    Array<FiniteElementSpace *> fes_comp(num_comp);
//    fes_comp = NULL;
//    for (int c = 0; c < num_comp; c++) {
//       Mesh *comp = topol_handler->GetComponentMesh(c);
//       fes_comp[c] = new FiniteElementSpace(comp, fec, udim);
//    }

//    {
//       assert(comp_mats.Size() == num_comp);
//       for (int c = 0; c < num_comp; c++)
//       {
//          Mesh *comp = topol_handler->GetComponentMesh(c);
//          BilinearForm a_comp(fes_comp[c]);

//          a_comp.AddDomainIntegrator(new DiffusionIntegrator);
//          if (full_dg)
//             a_comp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa));

//          a_comp.Assemble();
//          a_comp.Finalize();

//          rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()), comp_mats[c]);
//       }
//    }

//    // Boundary penalty matrixes
//    {
//       assert(bdr_mats.Size() == num_comp);
//       for (int c = 0; c < num_comp; c++)
//       {
//          Mesh *comp = topol_handler->GetComponentMesh(c);
//          assert(bdr_mats[c]->Size() == comp->bdr_attributes.Size());
//          Array<DenseMatrix *> *bdr_mats_c = bdr_mats[c];

//          for (int b = 0; b < comp->bdr_attributes.Size(); b++)
//          {
//             Array<int> bdr_marker(comp->bdr_attributes.Max());
//             bdr_marker = 0;
//             bdr_marker[comp->bdr_attributes[b] - 1] = 1;
//             BilinearForm a_comp(fes_comp[c]);
//             a_comp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), bdr_marker);

//             a_comp.Assemble();
//             a_comp.Finalize();

//             rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()), (*bdr_mats_c)[b]);
//          }
//       }
//    }

//    // Port penalty matrixes
//    const int num_ref_ports = topol_handler->GetNumRefPorts();
//    {
//       assert(port_mats.Size() == num_ref_ports);
//       for (int p = 0; p < num_ref_ports; p++)
//       {
//          assert(port_mats[p]->NumRows() == 2);
//          assert(port_mats[p]->NumCols() == 2);

//          int c1, c2;
//          topol_handler->GetComponentPair(p, c1, c2);
//          Mesh *comp1 = topol_handler->GetComponentMesh(c1);
//          Mesh *comp2 = topol_handler->GetComponentMesh(c2);

//          Mesh mesh1(*comp1);
//          Mesh mesh2(*comp2);

//          Array<int> c_idx(2);
//          c_idx[0] = c1;
//          c_idx[1] = c2;
//          Array2D<SparseMatrix *> spmats(2,2);
//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++)
//                spmats(i, j) = new SparseMatrix(fes_comp[c_idx[i]]->GetTrueVSize(), fes_comp[c_idx[j]]->GetTrueVSize());

//          Array<InterfaceInfo> *if_infos = topol_handler->GetRefInterfaceInfos(p);

//          // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
//          // Need to use two copied instances.
//          AssembleInterfaceMatrix(&mesh1, &mesh2, fes_comp[c1], fes_comp[c2], interface_integ, if_infos, spmats);

//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++) spmats(i, j)->Finalize();

//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++)
//                rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], spmats(i,j), (*port_mats[p])(i, j));

//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++) delete spmats(i, j);
//       }  // for (int p = 0; p < num_ref_ports; p++)
//    }

//    for (int k = 0 ; k < fes_comp.Size(); k++) delete fes_comp[k];
// }

// void PoissonSolver::SaveROMElements(const std::string &filename)
// {
//    assert(topol_mode == TopologyHandlerMode::COMPONENT);
//    const TrainMode train_mode = rom_handler->GetTrainMode();
//    assert(train_mode == UNIVERSAL);

//    hid_t file_id;
//    herr_t errf = 0;
//    file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//    assert(file_id >= 0);

//    {  // components + boundary
//       hid_t grp_id;
//       grp_id = H5Gcreate(file_id, "components", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//       assert(grp_id >= 0);

//       const int num_comp = topol_handler->GetNumComponents();
//       assert(comp_mats.Size() == num_comp);
//       assert(bdr_mats.Size() == num_comp);

//       hdf5_utils::WriteAttribute(grp_id, "number_of_components", num_comp);

//       for (int c = 0; c < num_comp; c++)
//       {
//          hid_t comp_grp_id;
//          comp_grp_id = H5Gcreate(grp_id, std::to_string(c).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//          assert(comp_grp_id >= 0);

//          hdf5_utils::WriteDataset(comp_grp_id, "domain", *(comp_mats[c]));

//          {  // boundary
//             hid_t bdr_grp_id;
//             bdr_grp_id = H5Gcreate(comp_grp_id, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//             assert(bdr_grp_id >= 0);

//             const int num_bdr = bdr_mats[c]->Size();
//             Mesh *comp = topol_handler->GetComponentMesh(c);
//             assert(num_bdr == comp->bdr_attributes.Size());

//             hdf5_utils::WriteAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);
            
//             Array<DenseMatrix *> *bdr_mat_c = bdr_mats[c];
//             for (int b = 0; b < num_bdr; b++)
//                hdf5_utils::WriteDataset(bdr_grp_id, std::to_string(b), *(*bdr_mat_c)[b]);

//             errf = H5Gclose(bdr_grp_id);
//             assert(errf >= 0);
//          }

//          errf = H5Gclose(comp_grp_id);
//          assert(errf >= 0);
//       }  // for (int c = 0; c < num_comp; c++)

//       errf = H5Gclose(grp_id);
//       assert(errf >= 0);
//    }

//    {  // (reference) ports
//       hid_t grp_id;
//       grp_id = H5Gcreate(file_id, "ports", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//       assert(grp_id >= 0);

//       const int num_ref_ports = topol_handler->GetNumRefPorts();
//       assert(port_mats.Size() == num_ref_ports);

//       hdf5_utils::WriteAttribute(grp_id, "number_of_ports", num_ref_ports);
      
//       for (int p = 0; p < num_ref_ports; p++)
//       {
//          assert(port_mats[p]->NumRows() == 2);
//          assert(port_mats[p]->NumCols() == 2);

//          hid_t port_grp_id;
//          port_grp_id = H5Gcreate(grp_id, std::to_string(p).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//          assert(port_grp_id >= 0);

//          Array2D<DenseMatrix *> *port_mat = port_mats[p];
//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++)
//             {
//                std::string dset_name = std::to_string(i) + std::to_string(j);
//                hdf5_utils::WriteDataset(port_grp_id, dset_name, *((*port_mat)(i,j)));
//             }
         
//          errf = H5Gclose(port_grp_id);
//          assert(errf >= 0);
//       }  // for (int p = 0; p < num_ref_ports; p++)

//       errf = H5Gclose(grp_id);
//       assert(errf >= 0);
//    }

//    errf = H5Fclose(file_id);
//    assert(errf >= 0);
//    return;
// }

// void PoissonSolver::LoadROMElements(const std::string &filename)
// {
//    assert(topol_mode == TopologyHandlerMode::COMPONENT);
//    const TrainMode train_mode = rom_handler->GetTrainMode();
//    assert(train_mode == UNIVERSAL);

//    hid_t file_id;
//    herr_t errf = 0;
//    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
//    assert(file_id >= 0);

//    {  // components
//       hid_t grp_id;
//       grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
//       assert(grp_id >= 0);

//       int num_comp;
//       hdf5_utils::ReadAttribute(grp_id, "number_of_components", num_comp);
//       assert(num_comp == topol_handler->GetNumComponents());
//       assert(comp_mats.Size() == num_comp);
//       assert(bdr_mats.Size() == num_comp);

//       for (int c = 0; c < num_comp; c++)
//       {
//          hid_t comp_grp_id;
//          comp_grp_id = H5Gopen2(grp_id, std::to_string(c).c_str(), H5P_DEFAULT);
//          assert(comp_grp_id >= 0);

//          hdf5_utils::ReadDataset(comp_grp_id, "domain", *(comp_mats[c]));

//          {  // boundary
//             hid_t bdr_grp_id;
//             bdr_grp_id = H5Gopen2(comp_grp_id, "boundary", H5P_DEFAULT);
//             assert(bdr_grp_id >= 0);

//             int num_bdr;
//             hdf5_utils::ReadAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);

//             Mesh *comp = topol_handler->GetComponentMesh(c);
//             assert(num_bdr == comp->bdr_attributes.Size());
//             assert(num_bdr = bdr_mats[c]->Size());

//             Array<DenseMatrix *> *bdr_mat_c = bdr_mats[c];
//             for (int b = 0; b < num_bdr; b++)
//                hdf5_utils::ReadDataset(bdr_grp_id, std::to_string(b), *(*bdr_mat_c)[b]);

//             errf = H5Gclose(bdr_grp_id);
//             assert(errf >= 0);
//          }

//          errf = H5Gclose(comp_grp_id);
//          assert(errf >= 0);
//       }  // for (int c = 0; c < num_comp; c++)

//       errf = H5Gclose(grp_id);
//       assert(errf >= 0);
//    }

//    {  // (reference) ports
//       hid_t grp_id;
//       grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
//       assert(grp_id >= 0);

//       int num_ref_ports;
//       hdf5_utils::ReadAttribute(grp_id, "number_of_ports", num_ref_ports);
//       assert(num_ref_ports == topol_handler->GetNumRefPorts());
//       assert(port_mats.Size() == num_ref_ports);

//       for (int p = 0; p < num_ref_ports; p++)
//       {
//          assert(port_mats[p]->NumRows() == 2);
//          assert(port_mats[p]->NumCols() == 2);

//          hid_t port_grp_id;
//          port_grp_id = H5Gopen2(grp_id, std::to_string(p).c_str(), H5P_DEFAULT);
//          assert(port_grp_id >= 0);

//          Array2D<DenseMatrix *> *port_mat = port_mats[p];
//          for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 2; j++)
//             {
//                std::string dset_name = std::to_string(i) + std::to_string(j);
//                hdf5_utils::ReadDataset(port_grp_id, dset_name, *((*port_mat)(i,j)));
//             }
         
//          errf = H5Gclose(port_grp_id);
//          assert(errf >= 0);
//       }  // for (int p = 0; p < num_ref_ports; p++)

//       errf = H5Gclose(grp_id);
//       assert(errf >= 0);
//    }

//    errf = H5Fclose(file_id);
//    assert(errf >= 0);

//    return;
// }

// void PoissonSolver::AssembleROM()
// {
//    assert(topol_mode == TopologyHandlerMode::COMPONENT);
//    const TrainMode train_mode = rom_handler->GetTrainMode();
//    assert(train_mode == UNIVERSAL);

//    const Array<int> rom_block_offsets = rom_handler->GetBlockOffsets();
//    SparseMatrix *romMat = new SparseMatrix(rom_block_offsets.Last(), rom_block_offsets.Last());

//    // component domain matrix.
//    for (int m = 0; m < numSub; m++)
//    {
//       int c_type = topol_handler->GetMeshType(m);
//       int num_basis = rom_handler->GetNumBasis(c_type);

//       Array<int> vdofs(num_basis);
//       for (int k = rom_block_offsets[m]; k < rom_block_offsets[m+1]; k++)
//          vdofs[k - rom_block_offsets[m]] = k;

//       romMat->AddSubMatrix(vdofs, vdofs, *(comp_mats[c_type]));

//       // boundary matrixes of each component.
//       Array<int> *bdr_c2g = topol_handler->GetBdrAttrComponentToGlobalMap(m);
//       Array<DenseMatrix *> *bdr_mat = bdr_mats[c_type];

//       for (int b = 0; b < bdr_c2g->Size(); b++)
//       {
//          int is_global = global_bdr_attributes.Find((*bdr_c2g)[b]);
//          if (is_global < 0) continue;

//          romMat->AddSubMatrix(vdofs, vdofs, *(*bdr_mat)[b]);
//       }
//    }

//    // interface matrixes.
//    for (int p = 0; p < topol_handler->GetNumPorts(); p++)
//    {
//       const PortInfo *pInfo = topol_handler->GetPortInfo(p);
//       const int p_type = topol_handler->GetPortType(p);
//       Array2D<DenseMatrix *> *port_mat = port_mats[p_type];

//       const int m1 = pInfo->Mesh1;
//       const int m2 = pInfo->Mesh2;
//       const int c1 = topol_handler->GetMeshType(m1);
//       const int c2 = topol_handler->GetMeshType(m2);
//       const int num_basis1 = rom_handler->GetNumBasis(c1);
//       const int num_basis2 = rom_handler->GetNumBasis(c2);

//       Array<int> vdofs1(num_basis1), vdofs2(num_basis2);
//       for (int k = rom_block_offsets[m1]; k < rom_block_offsets[m1+1]; k++)
//          vdofs1[k - rom_block_offsets[m1]] = k;
//       for (int k = rom_block_offsets[m2]; k < rom_block_offsets[m2+1]; k++)
//          vdofs2[k - rom_block_offsets[m2]] = k;
//       Array<Array<int> *> vdofs(2);
//       vdofs[0] = &vdofs1;
//       vdofs[1] = &vdofs2;

//       for (int i = 0; i < 2; i++)
//          for (int j = 0; j < 2; j++)
//             romMat->AddSubMatrix(*vdofs[i], *vdofs[j], *((*port_mat)(i, j)));
//    }

//    romMat->Finalize();
//    rom_handler->LoadOperator(romMat);
// }

void StokesSolver::Solve()
{
   int maxIter = config.GetOption<int>("solver/max_iter", 10000);
   double rtol = config.GetOption<double>("solver/relative_tolerance", 1.e-10);
   double atol = config.GetOption<double>("solver/absolute_tolerance", 1.e-10);
   int print_level = config.GetOption<int>("solver/print_level", 0);

   BlockVector urhs(u_offsets), prhs(p_offsets);
   BlockVector uvec(u_offsets), pvec(p_offsets);
   // copy each component of the right-hand side.
   GetVariableVector(0, *RHS, urhs);
   GetVariableVector(1, *RHS, prhs);
   // BlockVector R1(u_offsets);
   Vector R1(urhs.Size());

{
   PrintMatrix(*M, "stokes.M.txt");
   PrintMatrix(*B, "stokes.B.txt");

   PrintVector(urhs, "stokes.urhs.txt");
   PrintVector(prhs, "stokes.prhs.txt");
}

   CGSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(*M);
   solver.SetPrintLevel(print_level);
   solver.Mult(urhs, R1);

// {
//    SetVariableVector(0, R1, *U);
//    return;
// }

   // B * A^{-1} * F1 - G1
   Vector R2(prhs.Size());
   B->Mult(R1, R2);
   R2 -= prhs;

   printf("Set up pressure RHS\n");

   SchurOperator schur(M, B);
   CGSolver solver2;
   solver2.SetOperator(schur);
   solver2.SetPrintLevel(print_level);
   solver2.SetAbsTol(rtol);
   solver2.SetMaxIter(maxIter);

   OrthoSolver ortho;
   if (!pres_dbc)
   {
      ortho.SetSolver(solver2);
      ortho.SetOperator(schur);
      printf("OrthoSolver Set up.\n");
   }

   printf("Solving for pressure\n");
   // printf("%d ?= %d ?= %d\n", R2.Size(), p.Size(), ortho.Height());
   if (pres_dbc)
      solver2.Mult(R2, pvec);
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
   printf("Velocity is solved.\n");

   // Copy back to global vector.
   SetVariableVector(0, uvec, *U);
   SetVariableVector(1, pvec, *U);
}

void StokesSolver::InitIndividualParaview(const std::string& file_prefix)
{
   paraviewColls.SetSize(numSub);

   for (int m = 0; m < numSub; m++) {
      ostringstream oss;
      // Each subdomain needs to be save separately.
      oss << file_prefix << "_" << std::to_string(m);

      paraviewColls[m] = new ParaViewDataCollection(oss.str().c_str(), &(*meshes[m]));
      paraviewColls[m]->SetLevelsOfDetail(uorder);
      paraviewColls[m]->SetHighOrderOutput(true);
      paraviewColls[m]->SetPrecision(8);

      paraviewColls[m]->RegisterField("vel", us[m]);
      paraviewColls[m]->RegisterField("pres", ps[m]);
      paraviewColls[m]->SetOwnData(false);
   }
}

// void PoissonSolver::InitUnifiedParaview(const std::string& file_prefix)
// {
//    // grid function initialization for visual.
//    // TODO: for vector solution.
//    pmesh = topol_handler->GetGlobalMesh();
//    global_fes = new FiniteElementSpace(pmesh, fec, udim);
//    global_us_visual = new GridFunction(global_fes);

//    MultiBlockSolver::InitUnifiedParaview(file_prefix);
// }

// double PoissonSolver::CompareSolution()
// {
//    // Copy the rom solution.
//    BlockVector romU(domain_offsets);
//    romU = *U;
//    Array<GridFunction *> rom_us;
//    Array<VectorGridFunctionCoefficient *> rom_u_coeffs;
//    ConstantCoefficient zero(0.0);
//    rom_us.SetSize(numSub);
//    rom_u_coeffs.SetSize(numSub);
//    for (int m = 0; m < numSub; m++)
//    {
//       rom_us[m] = new GridFunction(fes[m], romU.GetBlock(m), 0);

//       // BC's are weakly constrained and there is no essential dofs.
//       // Does this make any difference?
//       rom_us[m]->SetTrueVector();

//       rom_u_coeffs[m] = new VectorGridFunctionCoefficient(rom_us[m]);
//    }

//    // TODO: right now we solve the full-order system to compare the solution.
//    // Need to implement loading the fom solution file?
//    StopWatch solveTimer;
//    solveTimer.Start();
//    Solve();
//    solveTimer.Stop();
//    printf("FOM-solve time: %f seconds.\n", solveTimer.RealTime());

//    // Compare the solution.
//    double norm = 0.0;
//    double error = 0.0;
//    for (int m = 0; m < numSub; m++)
//    {
//       norm += us[m]->ComputeLpError(2, zero);
//       error += us[m]->ComputeLpError(2, *rom_u_coeffs[m]);
//    }
//    error /= norm;
//    printf("Relative error: %.5E\n", error);

//    for (int m = 0; m < numSub; m++)
//    {
//       delete rom_us[m];
//       delete rom_u_coeffs[m];
//    }

//    return error;
// }

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

// void PoissonSolver::SetParameterizedProblem(ParameterizedProblem *problem)
// {
//    // clean up rhs for parametrized problem.
//    if (rhs_coeffs.Size() > 0)
//    {
//       for (int k = 0; k < rhs_coeffs.Size(); k++) delete rhs_coeffs[k];
//       rhs_coeffs.SetSize(0);
//    }
//    // clean up boundary functions for parametrized problem.
//    bdr_coeffs = NULL;

//    if (problem->scalar_bdr_ptr != NULL)
//       AddBCFunction(*(problem->scalar_bdr_ptr), problem->battr);
//    else
//       AddBCFunction(0.0);

//    if (problem->scalar_rhs_ptr != NULL)
//       AddRHSFunction(*(problem->scalar_rhs_ptr));
//    else
//       AddRHSFunction(0.0);
// }