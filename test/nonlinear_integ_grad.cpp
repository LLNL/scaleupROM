#include<gtest/gtest.h>
#include "nonlinear_integ.hpp"
#include "input_parser.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

enum IntegratorType {
   DOMAIN, INTERIOR, BDR
};

void CheckGradient(NonlinearFormIntegrator *integ, const IntegratorType type, bool use_dg=false)
{   
   // 1. Parse command-line options.
   std::string mesh_file = config.GetRequiredOption<std::string>("mesh/filename");
   int order = config.GetOption<int>("discretization/order", 1);
   int num_refinement = config.GetOption<int>("mesh/uniform_refinement", 0);

   Mesh *mesh = new Mesh(mesh_file.c_str(), 1, 1);
   int dim = mesh->Dimension();

   for (int l = 0; l < num_refinement; l++)
   {
      mesh->UniformRefinement();
   }

   FiniteElementCollection *dg_coll(new DG_FECollection(order, dim));
   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));

   FiniteElementSpace *fes;
   if (use_dg)
   {
      fes = new FiniteElementSpace(mesh, dg_coll, dim);
   }
   else
   {
      fes = new FiniteElementSpace(mesh, h1_coll, dim);
   }

   Array<int> ess_attr(mesh->bdr_attributes.Max()), ess_tdof(0);
   ess_attr = 0;
   ess_attr[3] = 1;

   // if (!use_dg)
   //    fes->GetEssentialTrueDofs(ess_attr, ess_tdof);

   // 12. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u(fes), us(fes);
   for (int k = 0; k < u.Size(); k++)
   {
      u[k] = UniformRandom();
      us[k] = UniformRandom();
   }

   NonlinearForm *nform(new NonlinearForm(fes));
   ConstantCoefficient one(1.0);

   IntegrationRule gll_ir_nl = IntRules.Get(fes->GetFE(0)->GetGeomType(),
                                             (int)(ceil(1.5 * (2 * fes->GetMaxElementOrder() - 1))));
   integ->SetIntRule(&gll_ir_nl);
   switch (type)
   {
      case DOMAIN:
         nform->AddDomainIntegrator(integ);
         break;
      case INTERIOR:
         nform->AddInteriorFaceIntegrator(integ);
         break;
      case BDR:
         nform->AddBdrFaceIntegrator(integ, ess_attr);
         break;
      default:
         mfem_error("Unknown integrator type!");
         break;
   }
   
   nform->SetEssentialTrueDofs(ess_tdof);

   Vector Nu(u.Size());
   nform->Mult(u, Nu);
   double J0 = us * (Nu);
   printf("J0: %.5E\n", J0);

   // Operator &jac(nform->GetGradient(u));
   SparseMatrix *jac = dynamic_cast<SparseMatrix *>(&(nform->GetGradient(u)));
   Vector grad(u.Size());
   jac->MultTranspose(us, grad);
   double gg = grad * grad;
   printf("gg: %.5E\n", gg);

   // printf("N[u]\n");
   // for (int k = 0 ; k < Nu.Size(); k++)
   //    printf("%.1E\t", Nu(k));
   // printf("\n");

   // printf("jac_mat\n");
   // DenseMatrix jac_mat;
   // jac->ToDenseMatrix(jac_mat);
   // for (int i = 0; i < jac_mat.NumRows(); i++)
   // {
   //    for (int j = 0; j < jac_mat.NumCols(); j++)
   //    {
   //       printf("%.1E\t", jac_mat(i, j));
   //    }
   //    printf("\n");
   // }

   GridFunction u0(fes);
   u0 = u;

   double error1 = 1.0e10;
   printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "amp", "J0", "J1", "dJdx", "error");
   for (int k = 0; k < 40; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      double dx = amp / sqrt(gg);

      u.Set(1.0, u0);
      u.Add(dx, grad);

      nform->Mult(u, Nu);
      double J1 = us * (Nu);
      double dJdx = (J1 - J0) / dx;
      double error = abs((dJdx - gg) / gg);

      printf("%.5E\t%.5E\t%.5E\t%.5E\t%.5E\n", amp, J0, J1, dJdx, error);

      if (k > 4)
      {
         if (error > error1)
            break;
         else
            error1 = error;
      }
   }
   EXPECT_TRUE(error1 < 1.0e-7);

   // 17. Free the used memory.
   delete nform;
   delete fes;
   delete dg_coll;
   delete h1_coll;
   delete mesh;

   return;
}

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(IncompressibleInviscidFlux, Test_grad)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;
   // config.dict_["mesh"]["uniform_refinement"] = 2;

   bool use_dg = config.GetOption<bool>("discretization/full-discrete-galerkin", false);

   ConstantCoefficient pi(3.141592);
   auto *nlc_nlfi = new IncompressibleInviscidFluxNLFIntegrator(pi);
   // auto *nlc_nlfi = new VectorConvectionNLFIntegrator(one);
    
   CheckGradient(nlc_nlfi, IntegratorType::DOMAIN, use_dg);

   return;
}

TEST(DGLaxFriedrichsFlux, Test_grad)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;

   ConstantCoefficient pi(3.141592);
   auto *nlc_nlfi = new DGLaxFriedrichsFluxIntegrator(pi);
    
   CheckGradient(nlc_nlfi, IntegratorType::INTERIOR, true);

   return;
}

TEST(DGTemamFlux, Test_grad)
{
   config = InputParser("inputs/dd_mms.yml");
   config.dict_["discretization"]["order"] = 1;

   ConstantCoefficient pi(3.141592);
   auto *nlc_nlfi = new DGTemamFluxIntegrator(pi);
    
   CheckGradient(nlc_nlfi, IntegratorType::INTERIOR, true);

   return;
}

int main(int argc, char* argv[])
{
   MPI_Init(&argc, &argv);
   ::testing::InitGoogleTest(&argc, argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}