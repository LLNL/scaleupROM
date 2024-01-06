// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "steady_ns_solver.hpp"
#include "hyperreduction_integ.hpp"
#include "nonlinear_integ.hpp"
// #include "input_parser.hpp"
// #include "hdf5_utils.hpp"
// #include "linalg_utils.hpp"
#include "dg_bilinear.hpp"
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

   MultComponent(x);
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

void SteadyNSTensorROM::MultComponent(const Vector &x) const
{
   bool operator_loaded = romMat_m && romMat_b && romMat_bt && romMat_m_bdr && romMat_b_bdr && romMat_bt_bdr && romMat_m_itf && romMat_b_itf && romMat_bt_itf;
   if (!operator_loaded) return;

   Vector res_m(x.Size()), res_b(x.Size()), res_bt(x.Size()),
          res_m_bdr(x.Size()), res_b_bdr(x.Size()), res_bt_bdr(x.Size()),
          res_m_itf(x.Size()), res_b_itf(x.Size()), res_bt_itf(x.Size()),
          res_u(x.Size()), res_p(x.Size()), res_u_bdr(x.Size()), res_p_bdr(x.Size()), total(x.Size());
   res_m = 0.0; res_b = 0.0; res_bt = 0.0;
   res_m_bdr = 0.0; res_b_bdr = 0.0; res_bt_bdr = 0.0;
   res_m_itf = 0.0; res_b_itf = 0.0; res_bt_itf = 0.0;
   res_u = 0.0; res_p = 0.0, res_u_bdr = 0.0; res_p_bdr = 0.0, total = 0.0;
   
   romMat_m->Mult(x, res_m);
   romMat_b->Mult(x, res_b);
   romMat_bt->Mult(x, res_bt);
   romMat_m_bdr->Mult(x, res_m_bdr);
   romMat_b_bdr->Mult(x, res_b_bdr);
   romMat_bt_bdr->Mult(x, res_bt_bdr);
   romMat_m_itf->Mult(x, res_m_itf);
   romMat_b_itf->Mult(x, res_b_itf);
   romMat_bt_itf->Mult(x, res_bt_itf);

   for (int m = 0; m < hs.Size(); m++)
   {
      x_comp.MakeRef(const_cast<Vector &>(x), block_offsets[m], block_offsets[m+1] - block_offsets[m]);
      y_comp.MakeRef(res_m, block_offsets[m], block_offsets[m+1] - block_offsets[m]);

      TensorAddScaledContract(*hs[m], 1.0, x_comp, x_comp, y_comp);
   }

   res_u_bdr = res_m_bdr;
   res_u_bdr += res_bt_bdr;
   res_u_bdr -= *f_rom;
   res_p_bdr = res_b_bdr;
   res_p_bdr -= *g_rom;

   res_u = res_u_bdr;
   res_u += res_m; res_u += res_bt;
   res_u += res_m_itf; res_u += res_bt_itf;

   res_p = res_p_bdr;
   res_p += res_b; res_u += res_b_itf;

   total = res_u; total += res_p;
   printf("%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
          "m_dom", "b_dom", "bt_dom", "m_bdr", "b_bdr", "bt_bdr", "m_itf", "b_itf", "bt_itf", "f_rom", "g_rom");
   printf("%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\t%.4E\n",
          sqrt(res_m*res_m), sqrt(res_b*res_b), sqrt(res_bt*res_bt),
          sqrt(res_m_bdr*res_m_bdr), sqrt(res_b_bdr*res_b_bdr), sqrt(res_bt_bdr*res_bt_bdr),
          sqrt(res_m_itf*res_m_itf), sqrt(res_b_itf*res_b_itf), sqrt(res_bt_itf*res_bt_itf),
          sqrt((*f_rom)*(*f_rom)), sqrt((*g_rom)*(*g_rom)));
   printf("%10s\t%10s\t%10s\t%10s\t%10s\n",
          "res_u_bdr", "res_p_bdr", "res_u", "res_p", "total");
   printf("%.4E\t%.4E\t%.4E\t%.4E\t%.4E\n",
          sqrt(res_u_bdr*res_u_bdr), sqrt(res_p_bdr*res_p_bdr),
          sqrt(res_u*res_u), sqrt(res_p*res_p), sqrt(total*total));
}

/*
   SteadyNSSolver
*/

SteadyNSSolver::SteadyNSSolver()
   : StokesSolver()
{
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
      DeletePointers(comp_tensors);
      if (rom_handler->SaveOperator() != ROMBuildingLevel::COMPONENT)
         DeletePointers(subdomain_tensors);
   }

   delete romMat_m;
   delete romMat_b;
   delete romMat_bt;
   delete romMat_m_bdr;
   delete romMat_b_bdr;
   delete romMat_bt_bdr;
   delete romMat_m_itf;
   delete romMat_b_itf;
   delete romMat_bt_itf;
   delete f_rom;
   delete g_rom;
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
         // TODO: Non-homogeneous Neumann stress bc
         if (!BCExistsOnBdr(b)) continue;

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
         if (!BCExistsOnBdr(b))
            hs[m]->AddBdrFaceIntegrator(new DGTemamFluxIntegrator(*zeta_coeff), *bdr_markers[b]);
      }
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

   comp_mats_m.SetSize(num_comp);
   comp_mats_b.SetSize(num_comp);
   comp_mats_bt.SetSize(num_comp);
   comp_mats_m = NULL;
   comp_mats_b = NULL;
   comp_mats_bt = NULL;
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

      SparseMatrix m_zero(m_mat->NumRows(), m_mat->NumCols());
      SparseMatrix b_zero(b_mat->NumRows(), b_mat->NumCols());
      SparseMatrix bt_zero(bt_mat->NumRows(), bt_mat->NumCols());
      m_zero.Finalize();
      b_zero.Finalize();
      bt_zero.Finalize();

      Array<int> dummy1, dummy2;
      BlockMatrix *sys_comp;
      sys_comp = FormBlockMatrix(m_mat, &b_zero, &bt_zero, dummy1, dummy2);
      comp_mats_m[c] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
      delete sys_comp;
      sys_comp = FormBlockMatrix(&m_zero, b_mat, &bt_zero, dummy1, dummy2);
      comp_mats_b[c] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
      delete sys_comp;
      sys_comp = FormBlockMatrix(&m_zero, &b_zero, bt_mat, dummy1, dummy2);
      comp_mats_bt[c] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
      delete sys_comp;

      delete bt_mat;
   }
}

void SteadyNSSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   StokesSolver::BuildBdrROMElement(fes_comp);

   const int num_comp = topol_handler->GetNumComponents();
   bdr_mats_m.SetSize(num_comp);
   bdr_mats_b.SetSize(num_comp);
   bdr_mats_bt.SetSize(num_comp);
   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      bdr_mats_m[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_m[c]) = NULL;
      bdr_mats_b[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_b[c]) = NULL;
      bdr_mats_bt[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_bt[c]) = NULL;
   }

   for (int c = 0; c < num_comp; c++)
   {
      const int fidx = c * num_var;
      Mesh *comp = topol_handler->GetComponentMesh(c);
      Array<SparseMatrix *> *bdr_mats_m_c = bdr_mats_m[c];
      Array<SparseMatrix *> *bdr_mats_b_c = bdr_mats_b[c];
      Array<SparseMatrix *> *bdr_mats_bt_c = bdr_mats_bt[c];

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

         SparseMatrix m_zero(m_mat->NumRows(), m_mat->NumCols());
         SparseMatrix b_zero(b_mat->NumRows(), b_mat->NumCols());
         SparseMatrix bt_zero(bt_mat->NumRows(), bt_mat->NumCols());
         m_zero.Finalize();
         b_zero.Finalize();
         bt_zero.Finalize();

         Array<int> dummy1, dummy2;
         BlockMatrix *sys_comp;
         sys_comp = FormBlockMatrix(m_mat, &b_zero, &bt_zero, dummy1, dummy2);
         (*bdr_mats_m_c)[b] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
         delete sys_comp;
         sys_comp = FormBlockMatrix(&m_zero, b_mat, &bt_zero, dummy1, dummy2);
         (*bdr_mats_b_c)[b] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
         delete sys_comp;
         sys_comp = FormBlockMatrix(&m_zero, &b_zero, bt_mat, dummy1, dummy2);
         (*bdr_mats_bt_c)[b] = rom_handler->ProjectOperatorOnReducedBasis(c, c, sys_comp);
         delete sys_comp;

         delete bt_mat;
      }
   }
}

void SteadyNSSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp)
{
   StokesSolver::BuildInterfaceROMElement(fes_comp);

   const int num_comp = topol_handler->GetNumComponents();
   Array<FiniteElementSpace *> ufes_comp(num_comp), pfes_comp(num_comp);
   for (int c = 0; c < num_comp; c++)
   {
      ufes_comp[c] = fes_comp[c * num_var];
      pfes_comp[c] = fes_comp[c * num_var + 1];
   }

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   port_mats_m.SetSize(num_ref_ports);
   port_mats_b.SetSize(num_ref_ports);
   port_mats_bt.SetSize(num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
   {
      port_mats_m[p] = new Array2D<SparseMatrix *>(2,2);
      port_mats_b[p] = new Array2D<SparseMatrix *>(2,2);
      port_mats_bt[p] = new Array2D<SparseMatrix *>(2,2);
      (*port_mats_m[p]) = NULL;
      (*port_mats_b[p]) = NULL;
      (*port_mats_bt[p]) = NULL;
   }

   for (int p = 0; p < num_ref_ports; p++)
   {
      int c1, c2;
      topol_handler->GetComponentPair(p, c1, c2);

      Array<int> c_idx(2);
      c_idx[0] = c1; c_idx[1] = c2;

      Array2D<SparseMatrix *> m_mats_p(2,2), b_mats_p(2,2), bt_mats_p(2,2);
      m_mats_p = NULL;
      b_mats_p = NULL;
      bt_mats_p = NULL;

      // NOTE: If comp1 == comp2, using comp1 and comp2 directly leads to an incorrect penalty matrix.
      // Need to use two copied instances.
      m_itf->AssembleInterfaceMatrixAtPort(p, ufes_comp, m_mats_p);
      b_itf->AssembleInterfaceMatrixAtPort(p, ufes_comp, pfes_comp, b_mats_p);

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
            // NOTE: the index also should be transposed.
            bt_mats_p(j, i) = Transpose(*b_mats_p(i, j));

      Array2D<SparseMatrix *> m_zero(2,2), b_zero(2,2), bt_zero(2,2);
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            m_zero(i, j) = new SparseMatrix(m_mats_p(i, j)->NumRows(), m_mats_p(i, j)->NumCols());
            b_zero(i, j) = new SparseMatrix(b_mats_p(i, j)->NumRows(), b_mats_p(i, j)->NumCols());
            bt_zero(i, j) = new SparseMatrix(bt_mats_p(i, j)->NumRows(), bt_mats_p(i, j)->NumCols());
            m_zero(i, j)->Finalize();
            b_zero(i, j)->Finalize();
            bt_zero(i, j)->Finalize();
         }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            Array<int> dummy1, dummy2;
            BlockMatrix *tmp_mat;
            tmp_mat = FormBlockMatrix(m_mats_p(i,j), b_zero(i,j), bt_zero(i,j), dummy1, dummy2);
            (*port_mats_m[p])(i, j) = rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], tmp_mat);
            delete tmp_mat;
            tmp_mat = FormBlockMatrix(m_zero(i,j), b_mats_p(i,j), bt_zero(i,j), dummy1, dummy2);
            (*port_mats_b[p])(i, j) = rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], tmp_mat);
            delete tmp_mat;
            tmp_mat = FormBlockMatrix(m_zero(i,j), b_zero(i,j), bt_mats_p(i,j), dummy1, dummy2);
            (*port_mats_bt[p])(i, j) = rom_handler->ProjectOperatorOnReducedBasis(c_idx[i], c_idx[j], tmp_mat);
            delete tmp_mat;
         }

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            delete m_mats_p(i, j);
            delete b_mats_p(i, j);
            delete bt_mats_p(i, j);
            delete m_zero(i, j);
            delete b_zero(i, j);
            delete bt_zero(i, j);
         }
   }  // for (int p = 0; p < num_ref_ports; p++)
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

      hdf5_utils::WriteSparseMatrix(comp_grp_id, "domain_m", comp_mats_m[c]);
      hdf5_utils::WriteSparseMatrix(comp_grp_id, "domain_b", comp_mats_b[c]);
      hdf5_utils::WriteSparseMatrix(comp_grp_id, "domain_bt", comp_mats_bt[c]);

      errf = H5Gclose(comp_grp_id);
      assert(errf >= 0);
   }  // for (int c = 0; c < num_comp; c++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void SteadyNSSolver::SaveBdrROMElement(hid_t &comp_grp_id, const int &comp_idx)
{
   MultiBlockSolver::SaveBdrROMElement(comp_grp_id, comp_idx);
   assert(comp_grp_id >= 0);
   herr_t errf;
   hid_t bdr_grp_id;
   bdr_grp_id = H5Gopen2(comp_grp_id, "boundary", H5P_DEFAULT);
   assert(bdr_grp_id >= 0);

   const int num_bdr = bdr_mats[comp_idx]->Size();
   
   Array<SparseMatrix *> *bdr_mat_m = bdr_mats_m[comp_idx];
   Array<SparseMatrix *> *bdr_mat_b = bdr_mats_b[comp_idx];
   Array<SparseMatrix *> *bdr_mat_bt = bdr_mats_bt[comp_idx];
   for (int b = 0; b < num_bdr; b++)
   {
      hdf5_utils::WriteSparseMatrix(bdr_grp_id, "m" + std::to_string(b), (*bdr_mat_m)[b]);
      hdf5_utils::WriteSparseMatrix(bdr_grp_id, "b" + std::to_string(b), (*bdr_mat_b)[b]);
      hdf5_utils::WriteSparseMatrix(bdr_grp_id, "bt" + std::to_string(b), (*bdr_mat_bt)[b]);
   }

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void SteadyNSSolver::SaveInterfaceROMElement(hid_t &file_id)
{
   MultiBlockSolver::SaveInterfaceROMElement(file_id);
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
   assert(grp_id >= 0);

   const int num_ref_ports = topol_handler->GetNumRefPorts();
   
   for (int p = 0; p < num_ref_ports; p++)
   {
      hid_t port_grp_id;
      port_grp_id = H5Gopen2(grp_id, std::to_string(p).c_str(), H5P_DEFAULT);
      assert(port_grp_id >= 0);

      Array2D<SparseMatrix *> *port_mat_m = port_mats_m[p];
      Array2D<SparseMatrix *> *port_mat_b = port_mats_b[p];
      Array2D<SparseMatrix *> *port_mat_bt = port_mats_bt[p];
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            std::string dset_name = std::to_string(i) + std::to_string(j);
            hdf5_utils::WriteSparseMatrix(port_grp_id, "m" + dset_name, (*port_mat_m)(i,j));
            hdf5_utils::WriteSparseMatrix(port_grp_id, "b" + dset_name, (*port_mat_b)(i,j));
            hdf5_utils::WriteSparseMatrix(port_grp_id, "bt" + dset_name, (*port_mat_bt)(i,j));
         }
      
      errf = H5Gclose(port_grp_id);
      assert(errf >= 0);
   }  // for (int p = 0; p < num_ref_ports; p++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
   return;
}

void SteadyNSSolver::LoadCompBdrROMElement(hid_t &file_id)
{
   MultiBlockSolver::LoadCompBdrROMElement(file_id);

   const int num_comp = topol_handler->GetNumComponents();
   comp_tensors.SetSize(num_comp);
   comp_tensors = NULL;

   comp_mats_m.SetSize(num_comp);
   comp_mats_b.SetSize(num_comp);
   comp_mats_bt.SetSize(num_comp);
   comp_mats_m = NULL;
   comp_mats_b = NULL;
   comp_mats_bt = NULL;

   bdr_mats_m.SetSize(num_comp);
   bdr_mats_b.SetSize(num_comp);
   bdr_mats_bt.SetSize(num_comp);
   for (int c = 0; c < num_comp; c++)
   {
      Mesh *comp = topol_handler->GetComponentMesh(c);
      bdr_mats_m[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_m[c]) = NULL;
      bdr_mats_b[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_b[c]) = NULL;
      bdr_mats_bt[c] = new Array<SparseMatrix *>(comp->bdr_attributes.Size());
      (*bdr_mats_bt[c]) = NULL;
   }

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

      comp_mats_m[c] = hdf5_utils::ReadSparseMatrix(comp_grp_id, "domain_m");
      comp_mats_b[c] = hdf5_utils::ReadSparseMatrix(comp_grp_id, "domain_b");
      comp_mats_bt[c] = hdf5_utils::ReadSparseMatrix(comp_grp_id, "domain_bt");

      LoadBdrROMElement2(comp_grp_id, c);

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

void SteadyNSSolver::LoadBdrROMElement2(hid_t &comp_grp_id, const int &comp_idx)
{
   assert(comp_grp_id >= 0);
   herr_t errf;
   hid_t bdr_grp_id;
   bdr_grp_id = H5Gopen2(comp_grp_id, "boundary", H5P_DEFAULT);
   assert(bdr_grp_id >= 0);

   int num_bdr;
   hdf5_utils::ReadAttribute(bdr_grp_id, "number_of_boundaries", num_bdr);

   Mesh *comp = topol_handler->GetComponentMesh(comp_idx);
   assert(num_bdr == comp->bdr_attributes.Size());
   assert(num_bdr = bdr_mats_m[comp_idx]->Size());
   assert(num_bdr = bdr_mats_b[comp_idx]->Size());
   assert(num_bdr = bdr_mats_bt[comp_idx]->Size());

   Array<SparseMatrix *> *bdr_mat_m = bdr_mats_m[comp_idx];
   Array<SparseMatrix *> *bdr_mat_b = bdr_mats_b[comp_idx];
   Array<SparseMatrix *> *bdr_mat_bt = bdr_mats_bt[comp_idx];
   for (int b = 0; b < num_bdr; b++)
   {
      (*bdr_mat_m)[b] = hdf5_utils::ReadSparseMatrix(bdr_grp_id, "m" + std::to_string(b));
      (*bdr_mat_b)[b] = hdf5_utils::ReadSparseMatrix(bdr_grp_id, "b" + std::to_string(b));
      (*bdr_mat_bt)[b] = hdf5_utils::ReadSparseMatrix(bdr_grp_id, "bt" + std::to_string(b));
   }

   errf = H5Gclose(bdr_grp_id);
   assert(errf >= 0);
   return;
}

void SteadyNSSolver::LoadInterfaceROMElement(hid_t &file_id)
{
   MultiBlockSolver::LoadInterfaceROMElement(file_id);
   assert(file_id >= 0);
   herr_t errf;
   hid_t grp_id;
   grp_id = H5Gopen2(file_id, "ports", H5P_DEFAULT);
   assert(grp_id >= 0);

   int num_ref_ports;
   hdf5_utils::ReadAttribute(grp_id, "number_of_ports", num_ref_ports);
   assert(num_ref_ports == topol_handler->GetNumRefPorts());
   port_mats_m.SetSize(num_ref_ports);
   port_mats_b.SetSize(num_ref_ports);
   port_mats_bt.SetSize(num_ref_ports);
   for (int p = 0; p < num_ref_ports; p++)
   {
      port_mats_m[p] = new Array2D<SparseMatrix *>(2,2);
      port_mats_b[p] = new Array2D<SparseMatrix *>(2,2);
      port_mats_bt[p] = new Array2D<SparseMatrix *>(2,2);
      (*port_mats_m[p]) = NULL;
      (*port_mats_b[p]) = NULL;
      (*port_mats_bt[p]) = NULL;
   }

   for (int p = 0; p < num_ref_ports; p++)
   {
      hid_t port_grp_id;
      port_grp_id = H5Gopen2(grp_id, std::to_string(p).c_str(), H5P_DEFAULT);
      assert(port_grp_id >= 0);

      Array2D<SparseMatrix *> *port_mat_m = port_mats_m[p];
      Array2D<SparseMatrix *> *port_mat_b = port_mats_b[p];
      Array2D<SparseMatrix *> *port_mat_bt = port_mats_bt[p];
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            std::string dset_name = std::to_string(i) + std::to_string(j);
            (*port_mat_m)(i,j) = hdf5_utils::ReadSparseMatrix(port_grp_id, "m" + dset_name);
            (*port_mat_b)(i,j) = hdf5_utils::ReadSparseMatrix(port_grp_id, "b" + dset_name);
            (*port_mat_bt)(i,j) = hdf5_utils::ReadSparseMatrix(port_grp_id, "bt" + dset_name);
         }
      
      errf = H5Gclose(port_grp_id);
      assert(errf >= 0);
   }  // for (int p = 0; p < num_ref_ports; p++)

   errf = H5Gclose(grp_id);
   assert(errf >= 0);
}

void SteadyNSSolver::AssembleROM()
{
   MultiBlockSolver::AssembleROM();

   const Array<int> *rom_block_offsets = rom_handler->GetBlockOffsets();
   romMat_m = new BlockMatrix(*rom_block_offsets);
   romMat_b = new BlockMatrix(*rom_block_offsets);
   romMat_bt = new BlockMatrix(*rom_block_offsets);
   romMat_m_bdr = new BlockMatrix(*rom_block_offsets);
   romMat_b_bdr = new BlockMatrix(*rom_block_offsets);
   romMat_bt_bdr = new BlockMatrix(*rom_block_offsets);
   romMat_m_itf = new BlockMatrix(*rom_block_offsets);
   romMat_b_itf = new BlockMatrix(*rom_block_offsets);
   romMat_bt_itf = new BlockMatrix(*rom_block_offsets);
   for (int i = 0; i < numSub; i++)
      for (int j = 0; j < numSub; j++)
      {
         const int ci = topol_handler->GetMeshType(i);
         const int cj = topol_handler->GetMeshType(j);
         int num_basis_i = rom_handler->GetNumBasis(ci);
         int num_basis_j = rom_handler->GetNumBasis(cj);

         romMat_m->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_b->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_bt->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_m_bdr->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_b_bdr->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_bt_bdr->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_m_itf->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_b_itf->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
         romMat_bt_itf->SetBlock(i, j, new SparseMatrix(num_basis_i, num_basis_j));
      }

   // component domain matrix.
   for (int m = 0; m < numSub; m++)
   {
      int c_type = topol_handler->GetMeshType(m);
      int num_basis = rom_handler->GetNumBasis(c_type);

      romMat_m->GetBlock(m,m) += *(comp_mats_m[c_type]);
      romMat_b->GetBlock(m,m) += *(comp_mats_b[c_type]);
      romMat_bt->GetBlock(m,m) += *(comp_mats_bt[c_type]);

      // boundary matrixes of each component.
      Array<int> *bdr_c2g = topol_handler->GetBdrAttrComponentToGlobalMap(m);
      Array<SparseMatrix *> *bdr_mat_m = bdr_mats_m[c_type];
      Array<SparseMatrix *> *bdr_mat_b = bdr_mats_b[c_type];
      Array<SparseMatrix *> *bdr_mat_bt = bdr_mats_bt[c_type];

      for (int b = 0; b < bdr_c2g->Size(); b++)
      {
         int global_idx = global_bdr_attributes.Find((*bdr_c2g)[b]);
         if (global_idx < 0) continue;
         if (!BCExistsOnBdr(global_idx)) continue;

         romMat_m_bdr->GetBlock(m,m) += *(*bdr_mat_m)[b];
         romMat_b_bdr->GetBlock(m,m) += *(*bdr_mat_b)[b];
         romMat_bt_bdr->GetBlock(m,m) += *(*bdr_mat_bt)[b];
      }
   }

   // interface matrixes.
   for (int p = 0; p < topol_handler->GetNumPorts(); p++)
   {
      const PortInfo *pInfo = topol_handler->GetPortInfo(p);
      const int p_type = topol_handler->GetPortType(p);
      Array2D<SparseMatrix *> *port_mat_m = port_mats_m[p_type];
      Array2D<SparseMatrix *> *port_mat_b = port_mats_b[p_type];
      Array2D<SparseMatrix *> *port_mat_bt = port_mats_bt[p_type];

      const int m1 = pInfo->Mesh1;
      const int m2 = pInfo->Mesh2;
      const int c1 = topol_handler->GetMeshType(m1);
      const int c2 = topol_handler->GetMeshType(m2);
      const int num_basis1 = rom_handler->GetNumBasis(c1);
      const int num_basis2 = rom_handler->GetNumBasis(c2);

      Array<int> midx(2), num_basis(2);
      midx[0] = m1;
      midx[1] = m2;
      num_basis[0] = num_basis1;
      num_basis[1] = num_basis2;

      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++)
         {
            romMat_m_itf->GetBlock(midx[i], midx[j]) += *((*port_mat_m)(i, j));
            romMat_b_itf->GetBlock(midx[i], midx[j]) += *((*port_mat_b)(i, j));
            romMat_bt_itf->GetBlock(midx[i], midx[j]) += *((*port_mat_bt)(i, j));
         }
   }

   romMat_m->Finalize();
   romMat_b->Finalize();
   romMat_bt->Finalize();
   romMat_m_bdr->Finalize();
   romMat_b_bdr->Finalize();
   romMat_bt_bdr->Finalize();
   romMat_m_itf->Finalize();
   romMat_b_itf->Finalize();
   romMat_bt_itf->Finalize();
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

void SteadyNSSolver::ProjectRHSOnReducedBasis()
{
   MultiBlockSolver::ProjectRHSOnReducedBasis();

   f_rom = new BlockVector(*(rom_handler->GetBlockOffsets()));
   g_rom = new BlockVector(*(rom_handler->GetBlockOffsets()));
   (*f_rom) = 0.0;
   (*g_rom) = 0.0;

   BlockVector RHS_domain(RHS->GetData(), var_offsets); // View vector for RHS.

   // Each basis is applied to the same column blocks.
   Vector *from_block, *grom_block;
   DenseMatrix* basis_i;
   Array<int> offsets(3);
   offsets[0] = 0;
   for (int i = 0; i < numSub; i++)
   {
      offsets[1] = u_offsets[i+1] - u_offsets[i];
      offsets[2] = p_offsets[i+1] - p_offsets[i];
      offsets.PartialSum();

      rom_handler->GetBasisOnSubdomain(i, basis_i);
      from_block = &(f_rom->GetBlock(i));
      grom_block = &(g_rom->GetBlock(i));
      for (int c = 0; c < basis_i->NumCols(); c++)
      {
         BlockVector col(basis_i->GetColumn(c), offsets);

         (*from_block)[c] += (col.GetBlock(0) * RHS_domain.GetBlock(0 + num_var * i));
         (*grom_block)[c] += (col.GetBlock(1) * RHS_domain.GetBlock(1 + num_var * i));
      }
   }
}

void SteadyNSSolver::SolveROM()
{
   assert(subdomain_tensors.Size() == numSub);
   for (int m = 0; m < numSub; m++) assert(subdomain_tensors[m]);

   BlockVector U_domain(U->GetData(), domain_offsets); // View vector for U.
   bool use_restart = config.GetOption<bool>("solver/use_restart", false);
   std::string restart_file;
   if (use_restart)
   {
      restart_file = config.GetRequiredOption<std::string>("solver/restart_file");
      LoadSolution(restart_file);
   }

   // NOTE(kevin): currently assumes direct solve.
   SteadyNSTensorROM rom_oper(rom_handler->GetOperator(), subdomain_tensors, *(rom_handler->GetBlockOffsets()));
   rom_oper.romMat_m = romMat_m;
   rom_oper.romMat_b = romMat_b;
   rom_oper.romMat_bt = romMat_bt;
   rom_oper.romMat_m_bdr = romMat_m_bdr;
   rom_oper.romMat_b_bdr = romMat_b_bdr;
   rom_oper.romMat_bt_bdr = romMat_bt_bdr;
   rom_oper.romMat_m_itf = romMat_m_itf;
   rom_oper.romMat_b_itf = romMat_b_itf;
   rom_oper.romMat_bt_itf = romMat_bt_itf;
   rom_oper.f_rom = f_rom;
   rom_oper.g_rom = g_rom;
   rom_handler->NonlinearSolve(rom_oper, &U_domain);
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
