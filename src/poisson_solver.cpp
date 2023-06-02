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
#include "hdf5_utils.hpp"
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
         if (bdr_coeffs[b] == NULL) continue;

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
         if (bdr_coeffs[b] == NULL) continue;

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

void PoissonSolver::BuildROMElements()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);
   assert(rom_handler->BasisLoaded());

   // Component domain system
   const int num_comp = topol_handler->GetNumComponents();
   Array<FiniteElementSpace *> fes_comp;
   GetComponentFESpaces(fes_comp);

   {
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

         rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()), comp_mats[c]);
      }
   }

   // Boundary penalty matrixes
   {
      assert(bdr_mats.Size() == num_comp);
      for (int c = 0; c < num_comp; c++)
      {
         Mesh *comp = topol_handler->GetComponentMesh(c);
         assert(bdr_mats[c]->Size() == comp->bdr_attributes.Size());
         Array<DenseMatrix *> *bdr_mats_c = bdr_mats[c];

         for (int b = 0; b < comp->bdr_attributes.Size(); b++)
         {
            Array<int> bdr_marker(comp->bdr_attributes.Max());
            bdr_marker = 0;
            bdr_marker[comp->bdr_attributes[b] - 1] = 1;
            BilinearForm a_comp(fes_comp[c]);
            a_comp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(sigma, kappa), bdr_marker);

            a_comp.Assemble();
            a_comp.Finalize();

            rom_handler->ProjectOperatorOnReducedBasis(c, c, &(a_comp.SpMat()), (*bdr_mats_c)[b]);
         }
      }
   }

   // Port penalty matrixes
   const int num_ref_ports = topol_handler->GetNumRefPorts();
   {
      assert(port_mats.Size() == num_ref_ports);
      for (int p = 0; p < num_ref_ports; p++)
      {
         assert(port_mats[p]->NumRows() == 2);
         assert(port_mats[p]->NumCols() == 2);

         int c1, c2;
         topol_handler->GetComponentPair(p, c1, c2);
         Mesh *comp1 = topol_handler->GetComponentMesh(c1);
         Mesh *comp2 = topol_handler->GetComponentMesh(c2);

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
               rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], spmats(i,j), (*port_mats[p])(i, j));

         for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) delete spmats(i, j);
      }  // for (int p = 0; p < num_ref_ports; p++)
   }

   for (int k = 0 ; k < fes_comp.Size(); k++) delete fes_comp[k];
}

void PoissonSolver::SaveROMElements(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   assert(file_id >= 0);

   {  // components + boundary
      hid_t grp_id;
      grp_id = H5Gcreate(file_id, "components", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(grp_id >= 0);

      const int num_comp = topol_handler->GetNumComponents();
      assert(comp_mats.Size() == num_comp);
      assert(bdr_mats.Size() == num_comp);

      hdf5_utils::WriteAttribute(grp_id, "number_of_components", num_comp);

      for (int c = 0; c < num_comp; c++)
      {
         hid_t comp_grp_id;
         comp_grp_id = H5Gcreate(grp_id, std::to_string(c).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
         assert(comp_grp_id >= 0);

         hdf5_utils::WriteDataset(comp_grp_id, "domain", *(comp_mats[c]));

         {  // boundary
            hid_t bdr_grp_id;
            bdr_grp_id = H5Gcreate(comp_grp_id, "boundary", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert(bdr_grp_id >= 0);

            const int num_bdr = bdr_mats[c]->Size();
            Mesh *comp = topol_handler->GetComponentMesh(c);
            assert(num_bdr == comp->bdr_attributes.Size());

            hdf5_utils::WriteAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);
            
            Array<DenseMatrix *> *bdr_mat_c = bdr_mats[c];
            for (int b = 0; b < num_bdr; b++)
               hdf5_utils::WriteDataset(bdr_grp_id, std::to_string(b), *(*bdr_mat_c)[b]);

            errf = H5Gclose(bdr_grp_id);
            assert(errf >= 0);
         }

         errf = H5Gclose(comp_grp_id);
         assert(errf >= 0);
      }  // for (int c = 0; c < num_comp; c++)

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   {  // (reference) ports
      hid_t grp_id;
      grp_id = H5Gcreate(file_id, "ports", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(grp_id >= 0);

      const int num_ref_ports = topol_handler->GetNumRefPorts();
      assert(port_mats.Size() == num_ref_ports);

      hdf5_utils::WriteAttribute(grp_id, "number_of_ports", num_ref_ports);
      
      for (int p = 0; p < num_ref_ports; p++)
      {
         assert(port_mats[p]->NumRows() == 2);
         assert(port_mats[p]->NumCols() == 2);

         hid_t port_grp_id;
         port_grp_id = H5Gcreate(grp_id, std::to_string(p).c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
         assert(port_grp_id >= 0);

         Array2D<DenseMatrix *> *port_mat = port_mats[p];
         for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
               std::string dset_name = std::to_string(i) + std::to_string(j);
               hdf5_utils::WriteDataset(port_grp_id, dset_name, *((*port_mat)(i,j)));
            }
         
         errf = H5Gclose(port_grp_id);
         assert(errf >= 0);
      }  // for (int p = 0; p < num_ref_ports; p++)

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);
   return;
}

void PoissonSolver::LoadROMElements(const std::string &filename)
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   hid_t file_id;
   herr_t errf = 0;
   file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file_id >= 0);

   {  // components
      hid_t grp_id;
      grp_id = H5Gopen2(file_id, "components", H5P_DEFAULT);
      assert(grp_id >= 0);

      int num_comp;
      hdf5_utils::ReadAttribute(grp_id, "number_of_components", num_comp);
      assert(num_comp == topol_handler->GetNumComponents());
      assert(comp_mats.Size() == num_comp);
      assert(bdr_mats.Size() == num_comp);

      for (int c = 0; c < num_comp; c++)
      {
         hid_t comp_grp_id;
         comp_grp_id = H5Gopen2(grp_id, std::to_string(c).c_str(), H5P_DEFAULT);
         assert(comp_grp_id >= 0);

         hdf5_utils::ReadDataset(comp_grp_id, "domain", *(comp_mats[c]));

         {  // boundary
            hid_t bdr_grp_id;
            bdr_grp_id = H5Gopen2(comp_grp_id, "boundary", H5P_DEFAULT);
            assert(bdr_grp_id >= 0);

            int num_bdr;
            hdf5_utils::ReadAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);

            Mesh *comp = topol_handler->GetComponentMesh(c);
            assert(num_bdr == comp->bdr_attributes.Size());
            assert(num_bdr = bdr_mats[c]->Size());

            Array<DenseMatrix *> *bdr_mat_c = bdr_mats[c];
            for (int b = 0; b < num_bdr; b++)
               hdf5_utils::ReadDataset(bdr_grp_id, std::to_string(b), *(*bdr_mat_c)[b]);

            errf = H5Gclose(bdr_grp_id);
            assert(errf >= 0);
         }

         errf = H5Gclose(comp_grp_id);
         assert(errf >= 0);
      }  // for (int c = 0; c < num_comp; c++)

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   {  // (reference) ports
      hid_t grp_id;
      grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
      assert(grp_id >= 0);

      int num_ref_ports;
      hdf5_utils::ReadAttribute(grp_id, "number_of_ports", num_ref_ports);
      assert(num_ref_ports == topol_handler->GetNumRefPorts());
      assert(port_mats.Size() == num_ref_ports);

      for (int p = 0; p < num_ref_ports; p++)
      {
         assert(port_mats[p]->NumRows() == 2);
         assert(port_mats[p]->NumCols() == 2);

         hid_t port_grp_id;
         port_grp_id = H5Gopen2(grp_id, std::to_string(p).c_str(), H5P_DEFAULT);
         assert(port_grp_id >= 0);

         Array2D<DenseMatrix *> *port_mat = port_mats[p];
         for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
               std::string dset_name = std::to_string(i) + std::to_string(j);
               hdf5_utils::ReadDataset(port_grp_id, dset_name, *((*port_mat)(i,j)));
            }
         
         errf = H5Gclose(port_grp_id);
         assert(errf >= 0);
      }  // for (int p = 0; p < num_ref_ports; p++)

      errf = H5Gclose(grp_id);
      assert(errf >= 0);
   }

   errf = H5Fclose(file_id);
   assert(errf >= 0);

   return;
}

void PoissonSolver::AssembleROM()
{
   assert(topol_mode == TopologyHandlerMode::COMPONENT);
   const TrainMode train_mode = rom_handler->GetTrainMode();
   assert(train_mode == UNIVERSAL);

   const Array<int> rom_block_offsets = rom_handler->GetBlockOffsets();
   SparseMatrix *romMat = new SparseMatrix(rom_block_offsets.Last(), rom_block_offsets.Last());

   // component domain matrix.
   for (int m = 0; m < numSub; m++)
   {
      int c_type = topol_handler->GetMeshType(m);
      int num_basis = rom_handler->GetNumBasis(c_type);

      Array<int> vdofs(num_basis);
      for (int k = rom_block_offsets[m]; k < rom_block_offsets[m+1]; k++)
         vdofs[k - rom_block_offsets[m]] = k;

      romMat->AddSubMatrix(vdofs, vdofs, *(comp_mats[c_type]));

      // boundary matrixes of each component.
      Array<int> *bdr_c2g = topol_handler->GetBdrAttrComponentToGlobalMap(m);
      Array<DenseMatrix *> *bdr_mat = bdr_mats[c_type];

      for (int b = 0; b < bdr_c2g->Size(); b++)
      {
         int is_global = global_bdr_attributes.Find((*bdr_c2g)[b]);
         if (is_global < 0) continue;

         romMat->AddSubMatrix(vdofs, vdofs, *(*bdr_mat)[b]);
      }
   }

   // interface matrixes.
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);
      const int p_type = topol_handler->GetPortType(p);
      Array2D<DenseMatrix *> *port_mat = port_mats[p_type];

      const int m1 = pInfo->Mesh1;
      const int m2 = pInfo->Mesh2;
      const int c1 = topol_handler->GetMeshType(m1);
      const int c2 = topol_handler->GetMeshType(m2);
      const int num_basis1 = rom_handler->GetNumBasis(c1);
      const int num_basis2 = rom_handler->GetNumBasis(c2);

      Array<int> vdofs1(num_basis1), vdofs2(num_basis2);
      for (int k = rom_block_offsets[m1]; k < rom_block_offsets[m1+1]; k++)
         vdofs1[k - rom_block_offsets[m1]] = k;
      for (int k = rom_block_offsets[m2]; k < rom_block_offsets[m2+1]; k++)
         vdofs2[k - rom_block_offsets[m2]] = k;
      Array<Array<int> *> vdofs(2);
      vdofs[0] = &vdofs1;
      vdofs[1] = &vdofs2;

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            romMat->AddSubMatrix(*vdofs[i], *vdofs[j], *((*port_mat)(i, j)));
   }

   romMat->Finalize();
   rom_handler->LoadOperator(romMat);
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