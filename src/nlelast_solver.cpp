// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "nlelast_solver.hpp"
#include "input_parser.hpp"
#include "linalg_utils.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;


/* NLElastOperator */

NLElastOperator::NLElastOperator(const int height_, const int width_, Array<NonlinearForm *> &hs_, InterfaceForm *nl_itf_,
   Array<int> &u_offsets_, const bool direct_solve_)
   : Operator(height_, width_), hs(hs_), nl_itf(nl_itf_),
     u_offsets(u_offsets_), direct_solve(direct_solve_)
{

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

NLElastOperator::~NLElastOperator()
{
   delete Hop;
   delete system_jac;
   delete hs_jac;
   delete uu_mono;
   delete mono_jac;
   delete jac_hypre;
}

void NLElastOperator::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;
   //cout<<"in mult"<<endl;
   Hop->Mult(x, y);
   //cout<<"out mult"<<endl;
   //if (nl_itf) nl_itf->InterfaceAddMult(x_u, y_u); // TODO: Add this when interface integrator is there
}

Operator& NLElastOperator::GetGradient(const Vector &x) const
{
   //cout<<"in grad"<<endl;

   delete hs_jac;
   delete jac_hypre;

   hs_jac = new BlockMatrix(u_offsets);
   for (int i = 0; i < hs.Size(); i++)
      for (int j = 0; j < hs.Size(); j++)
      {
         if (i == j)
         {
            x_u.MakeRef(const_cast<Vector &>(x), u_offsets[i], u_offsets[i+1] - u_offsets[i]);
         //cout<<"before getgrad"<<endl;

            hs_mats(i, j) = dynamic_cast<SparseMatrix *>(&hs[i]->GetGradient(x_u));
         //cout<<"after getgrad"<<endl;

         }
         else
         {
            delete hs_mats(i, j);
            hs_mats(i, j) = new SparseMatrix(u_offsets[i+1] - u_offsets[i], u_offsets[j+1] - u_offsets[j]);
         }
      }

   x_u.MakeRef(const_cast<Vector &>(x), 0, u_offsets.Last());
   //if (nl_itf) nl_itf->InterfaceGetGradient(x_u, hs_mats); // TODO: Enable when we have interface integrator

   for (int i = 0; i < hs.Size(); i++)
      for (int j = 0; j < hs.Size(); j++)
      {
         hs_mats(i, j)->Finalize();
         hs_jac->SetBlock(i, j, hs_mats(i, j));
      }

   mono_jac = hs_jac->CreateMonolithic();
   //cout<<"out grad"<<endl;

   if (direct_solve)
   {
      jac_hypre = new HypreParMatrix(MPI_COMM_SELF, sys_glob_size, sys_row_starts, mono_jac);
   //cout<<"out grad1"<<endl;
      return *jac_hypre;

   }  
   else
   //cout<<"out gra2"<<endl;

      return *mono_jac;
}

/* NLElastSolver */

NLElastSolver::NLElastSolver(DGHyperelasticModel* _model)
    : MultiBlockSolver()
{
   alpha = 0.0; // Only allow IIPG
   kappa = config.GetOption<double>("discretization/interface/kappa", (order + 1) * (order + 1));

   var_names = GetVariableNames();
   num_var = var_names.size();

   model = _model;

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

NLElastSolver::~NLElastSolver()
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

void NLElastSolver::SetupIC(std::function<void(const Vector &, Vector &)> F)
{
   init_x = new VectorFunctionCoefficient(dim, F);
   for (int m = 0; m < numSub; m++)
   {
      assert(us[m]);
      us[m]->ProjectCoefficient(*init_x);

      /* for (size_t i = 0; i < us[m]->Size(); i++)
       {
         cout<<"us[m]->Elem(i) is: "<<us[m]->Elem(i)<<endl;
       }

       cout<<endl<<endl; */
   }
   //MFEM_ABORT("test")
}

void NLElastSolver::SetupBCVariables()
{
   MultiBlockSolver::SetupBCVariables();
   bdr_coeffs.SetSize(numBdr);
   bdr_coeffs = NULL;

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

void NLElastSolver::InitVariables()
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
      meshes[m]->GetNodes(*us[m]);
      us[m]->SetTrueVector();
      //PrintVector(U->GetBlock(m), "ub.txt");
   }
   if (use_rom)
     MultiBlockSolver::InitROMHandler();
}

void NLElastSolver::BuildOperators()
{
   BuildRHSOperators();
   BuildDomainOperators();
}

bool NLElastSolver::BCExistsOnBdr(const int &global_battr_idx)
{
   assert((global_battr_idx >= 0) && (global_battr_idx < global_bdr_attributes.Size()));
   assert(bdr_coeffs.Size() == global_bdr_attributes.Size());
   return (bdr_coeffs[global_battr_idx]);
}

void NLElastSolver::BuildRHSOperators()
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

void NLElastSolver::SetupRHSBCOperators()
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

         switch (bdr_type[b])
         {
               case BoundaryType::DIRICHLET:
               //bs[m]->AddBdrFaceIntegrator(new DGElasticityDirichletLFIntegrator(
                  //*bdr_coeffs[b], *lambda_c[m], *mu_c[m], alpha, kappa), *bdr_markers[b]);
               bs[m]->AddBdrFaceIntegrator(new DGHyperelasticDirichletLFIntegrator(
               *bdr_coeffs[b], model, alpha, kappa), *bdr_markers[b]);
               break;
            case BoundaryType::NEUMANN:
               bs[m]->AddBdrFaceIntegrator(new VectorBoundaryLFIntegrator(*bdr_coeffs[b]), *bdr_markers[b]);
               break;
            default:
               printf("NLElastSolver::SetupRHSBCOperators - ");
               printf("boundary attribute %d has a non-zero function, but does not have boundary type. will not be enforced.",
                      global_bdr_attributes[b]);
               break;
         }
      }
   }
}

void NLElastSolver::BuildDomainOperators()
{
   // SanityCheckOnCoeffs();
   as.SetSize(numSub);

   for (int m = 0; m < numSub; m++)
   {
      as[m] = new NonlinearForm(fes[m]);
      as[m]->AddDomainIntegrator(new HyperelasticNLFIntegratorHR(model));

      if (full_dg)
      {
         as[m]->AddInteriorFaceIntegrator(
             new DGHyperelasticNLFIntegrator(model, alpha, kappa));
      }
   }

   a_itf = new InterfaceForm(meshes, fes, topol_handler); // TODO: Is this reasonable?
   //a_itf->AddIntefaceIntegrator(new InterfaceDGElasticityIntegrator(lambda_c[0], mu_c[0], alpha, kappa));
}

void NLElastSolver::Assemble()
{
   AssembleRHS();
   //AssembleOperator();
}

void NLElastSolver::AssembleRHS()
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

void NLElastSolver::AssembleOperator()
{
   "AssembleOperator is not needed for NLElastSolver!\n";
}

void NLElastSolver::AssembleInterfaceMatrices()
{
   assert(a_itf);
   a_itf->AssembleInterfaceMatrices(mats);
}

bool NLElastSolver::Solve()
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

   const int hw = U->Size();
   NLElastOperator oper(hw, hw, as, a_itf, var_offsets, direct_solve);

   // Test mult
   /* for (int i = 0; i < var_offsets.Size(); i++)
   {
      cout<<"var_offsets(i) is: "<<var_offsets[i]<<endl;
   }
   cout<<"hw is: "<<hw<<endl; */
   Vector _U(U->GetBlock(0));
   Vector _U0(_U);
   Vector _RHS(RHS->GetBlock(0));
   Vector _res(_U);
   _res = 0.0;
   cout<<"_U.Norml2() is: "<<_U.Norml2()<<endl;

   cout<<"_RHS.Norml2() is: "<<_RHS.Norml2()<<endl;

   oper.Mult(_U, _res);

   cout<<"_res.Norml2() is: "<<_res.Norml2()<<endl;
   
   /* for (size_t i = 0; i < _res.Size(); i++)
   {
      
      {
        cout<<"_res(i) is: "<<_res(i)<<endl;
        cout<<"i is: "<<i<<endl;
        cout<<"_U(i) is: "<<_U(i)<<endl;
        cout<<"_RHS(i) is: "<<_RHS(i)<<endl;
        if (abs(_res(i) - _RHS(i)) > 0.0000001)
        {
         cout<<"ohnon "<< abs(_res(i) - _RHS(i))<<endl;
        }
        cout<<endl;
      }
      
   } */

   for (size_t i = 0; i < _U.Size(); i++)
   {
      double _x1 = 0.5 * (sqrt(4.0 * _U0(i) + 1.0) - 1.0);
      _U0(i) = _x1;
   }
   

   for (size_t i = 0; i < _res.Size()/2; i++)
   {
      if ((abs(_res(i) - _RHS(i)) <= 0.0000001) && abs(_res(i + _res.Size()/2) - _RHS(i + _res.Size()/2) <= 0.0000001))
      {
         cout<<"_resx is: "<<_res(i)<<endl;
         cout<<"_resy is: "<<_res(i + _res.Size()/2)<<endl;
         cout<<"_RHSx is: "<<_RHS(i)<<endl;
         cout<<"_RHSy is: "<<_RHS(i + _res.Size()/2)<<endl;
         cout<<"_U0x is: "<<_U0(i)<<endl;
         cout<<"_U0y is: "<<_U0(i + _res.Size()/2)<<endl;
        cout<<endl;

      }
      
      
      /* {
        cout<<"_res(i) is: "<<_res(i)<<endl;
        cout<<"i is: "<<i<<endl;
        cout<<"_U(i) is: "<<_U(i)<<endl;
        cout<<"_RHS(i) is: "<<_RHS(i)<<endl;
        if (abs(_res(i) - _RHS(i)) > 0.0000001)
        {
         cout<<"ohnon "<< abs(_res(i) - _RHS(i))<<endl;
        }
        cout<<endl;
      } */
      
   }
   
   _res -= _RHS;
   
   cout<<"(_res - RHS).Norml2() is: "<<_res.Norml2()<<endl;
   MFEM_ABORT("test");

   cout<<"full_dg is: "<<full_dg<<endl;
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
   //cout<<"2 "<<endl;

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
  cout<<"4 "<<endl;

   newton_solver->Mult(*RHS, *U);
   //cout<<"5 "<<endl;

   bool converged = newton_solver->GetConverged();
   //cout<<"6 "<<endl;

   return converged;
}

void NLElastSolver::AddBCFunction(std::function<void(const Vector &, Vector &)> F, const int battr)
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

void NLElastSolver::AddRHSFunction(std::function<void(const Vector &, Vector &)> F)
{
   rhs_coeffs.Append(new VectorFunctionCoefficient(dim, F));
}

void NLElastSolver::SetupBCOperators()
{
   SetupRHSBCOperators();
   SetupDomainBCOperators();
}

void NLElastSolver::SetupDomainBCOperators()
{
   MFEM_ASSERT(as.Size() == numSub, "NonlinearForm bs != numSub.\n");
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

            if (bdr_type[b] == BoundaryType::DIRICHLET)
               //as[m]->AddBdrFaceIntegrator(new DGElasticityIntegrator(
                 // *(lambda_c[m]), *(mu_c[m]), alpha, kappa), *(bdr_markers[b]));
               //cout<<"*lambda_c[m] is: "<<model->lambda<<endl;
               //cout<<"*mu_c[m] is: "<<model->mu<<endl;
               as[m]->AddBdrFaceIntegrator(
                  new DGHyperelasticNLFIntegrator(model, alpha, kappa), *(bdr_markers[b]));
         }
      }
   } 
}

void NLElastSolver::SetParameterizedProblem(ParameterizedProblem *problem)
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
      /* Dirichlet bc requires a function specified, even for zero. */
      if (problem->bdr_type[b] == BoundaryType::DIRICHLET)
         assert(problem->vector_bdr_ptr[b]);
      
      /* Neumann bc does not require a function specified for zero */
      if (problem->vector_bdr_ptr[b])
         AddBCFunction(*(problem->vector_bdr_ptr[b]), problem->battr[b]);
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

void NLElastSolver::ProjectOperatorOnReducedBasis()
{ 
   Array2D<Operator *> tmp(mats.NumRows(), mats.NumCols());
   for (int i = 0; i < tmp.NumRows(); i++)
      for (int j = 0; j < tmp.NumCols(); j++)
         tmp(i, j) = mats(i, j);
         
   rom_handler->ProjectOperatorOnReducedBasis(tmp);
}

// Component-wise assembly
void NLElastSolver::BuildCompROMElement(Array<FiniteElementSpace *> &fes_comp)
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

void NLElastSolver::BuildBdrROMElement(Array<FiniteElementSpace *> &fes_comp)
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

void NLElastSolver::BuildInterfaceROMElement(Array<FiniteElementSpace *> &fes_comp)
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
//void NLElastSolver::SanityCheckOnCoeffs() { "NLElastSolver::SanityCheckOnCoeffs is not implemented yet!\n"; }
