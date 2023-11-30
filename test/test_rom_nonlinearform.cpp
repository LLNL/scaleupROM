#include<gtest/gtest.h>
#include "hyperreduction_integ.hpp"
#include "rom_nonlinearform.hpp"
#include "etc.hpp"

using namespace std;
using namespace mfem;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(ROMNonlinearForm, VectorConvectionTrilinearFormIntegrator)
{
   Mesh *mesh = new Mesh("meshes/test.4x4.mesh");
   const int dim = mesh->Dimension();
   const int order = UniformRandom(1, 3);

   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));
   FiniteElementSpace *fes(new FiniteElementSpace(mesh, h1_coll, dim));
   const int ndofs = fes->GetTrueVSize();

   const int num_basis = 10;
   // a fictitious basis.
   DenseMatrix basis(ndofs, num_basis);
   for (int i = 0; i < ndofs; i++)
      for (int j = 0; j < num_basis; j++)
         basis(i, j) = UniformRandom();

   IntegrationRule ir = IntRules.Get(fes->GetFE(0)->GetGeomType(),
                                    (int)(ceil(1.5 * (2 * fes->GetMaxElementOrder() - 1))));
   ConstantCoefficient pi(3.141592);
   auto *integ1 = new VectorConvectionTrilinearFormIntegrator(pi);
   integ1->SetIntRule(&ir);
   auto *integ2 = new VectorConvectionTrilinearFormIntegrator(pi);
   integ2->SetIntRule(&ir);

   NonlinearForm *nform(new NonlinearForm(fes));
   nform->AddDomainIntegrator(integ1);

   ROMNonlinearForm *rform(new ROMNonlinearForm(num_basis, fes));
   rform->AddDomainIntegrator(integ2);
   rform->SetBasis(basis);

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nqe = ir.GetNPoints();
   const int ne = fes->GetNE();
   Array<double> const& w_el = ir.GetWeights();
   Array<int> sample_el(ne * nqe), sample_qp(ne * nqe);
   Array<double> sample_qw(ne * nqe);
   for (int e = 0, idx = 0; e < ne; e++)
      for (int q = 0; q < nqe; q++, idx++)
      {
         sample_el[idx] = e;
         sample_qp[idx] = q;
         sample_qw[idx] = w_el[q];
      }
   rform->UpdateDomainIntegratorSampling(0, sample_el, sample_qp, sample_qw);

   Vector rom_u(num_basis), u(fes->GetTrueVSize());
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   basis.Mult(rom_u, u);

   Vector rom_y(num_basis), y(fes->GetTrueVSize()), Pty(num_basis);
   nform->Mult(u, y);
   basis.MultTranspose(y, Pty);
   rform->Mult(rom_u, rom_y);

   for (int k = 0; k < rom_y.Size(); k++)
      EXPECT_NEAR(rom_y(k), Pty(k), 1.0e-12);

   delete mesh;
   delete h1_coll;
   delete fes;
   delete nform;
   delete rform;
   return;
}

TEST(ROMNonlinearForm_gradient, VectorConvectionTrilinearFormIntegrator)
{
   Mesh *mesh = new Mesh("meshes/test.4x4.mesh");
   const int dim = mesh->Dimension();
   const int order = UniformRandom(1, 3);

   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));
   FiniteElementSpace *fes(new FiniteElementSpace(mesh, h1_coll, dim));
   const int ndofs = fes->GetTrueVSize();

   const int num_basis = 10;
   // a fictitious basis.
   DenseMatrix basis(ndofs, num_basis);
   for (int i = 0; i < ndofs; i++)
      for (int j = 0; j < num_basis; j++)
         basis(i, j) = UniformRandom();

   IntegrationRule ir = IntRules.Get(fes->GetFE(0)->GetGeomType(),
                                    (int)(ceil(1.5 * (2 * fes->GetMaxElementOrder() - 1))));
   ConstantCoefficient pi(3.141592);
   auto *integ = new VectorConvectionTrilinearFormIntegrator(pi);
   integ->SetIntRule(&ir);

   ROMNonlinearForm *rform(new ROMNonlinearForm(num_basis, fes));
   rform->AddDomainIntegrator(integ);
   rform->SetBasis(basis);

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nsample = UniformRandom(15, 20);
   const int nqe = ir.GetNPoints();
   const int ne = fes->GetNE();
   Array<double> const& w_el = ir.GetWeights();
   Array<int> sample_el(nsample), sample_qp(nsample);
   Array<double> sample_qw(nsample);
   for (int s = 0; s < nsample; s++)
   {
      sample_el[s] = UniformRandom(0, ne-1);
      sample_qp[s] = UniformRandom(0, nqe-1);
      sample_qw[s] = UniformRandom();
   }
   rform->UpdateDomainIntegratorSampling(0, sample_el, sample_qp, sample_qw);

   Vector rom_u(num_basis);
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   Vector rom_y(num_basis);
   rform->Mult(rom_u, rom_y);
   DenseMatrix *jac = dynamic_cast<DenseMatrix *>(&(rform->GetGradient(rom_u)));

   double J0 = 0.5 * (rom_y * rom_y);
   Vector grad(num_basis);
   jac->MultTranspose(rom_y, grad);
   double gg = sqrt(grad * grad);
   printf("J0: %.15E\n", J0);
   printf("grad: %.15E\n", gg);

   Vector du(grad);
   du /= gg;

   Vector rom_y1(num_basis);
   const int Nk = 40;
   double error1 = 1e100, error;
   printf("amp\tJ1\tdJdx\terror\n");
   for (int k = 0; k < Nk; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      Vector rom_u1(rom_u);
      rom_u1.Add(amp, du);

      rform->Mult(rom_u1, rom_y1);
      double J1 = 0.5 * (rom_y1 * rom_y1);
      double dJdx = (J1 - J0) / amp;
      error = abs(dJdx - gg) / gg;

      printf("%.5E\t%.5E\t%.5E\t%.5E\n", amp, J1, dJdx, error);
      if (error >= error1)
         break;
      error1 = error;
   }
   EXPECT_TRUE(min(error, error1) < 1.0e-7);

   delete mesh;
   delete h1_coll;
   delete fes;
   delete rform;
   return;
}

TEST(ROMNonlinearForm_fast, VectorConvectionTrilinearFormIntegrator)
{
   Mesh *mesh = new Mesh("meshes/test.4x4.mesh");
   const int dim = mesh->Dimension();
   const int order = UniformRandom(1, 3);

   FiniteElementCollection *h1_coll(new H1_FECollection(order, dim));
   FiniteElementSpace *fes(new FiniteElementSpace(mesh, h1_coll, dim));
   const int ndofs = fes->GetTrueVSize();

   const int num_basis = 10;
   // a fictitious basis.
   DenseMatrix basis(ndofs, num_basis);
   for (int i = 0; i < ndofs; i++)
      for (int j = 0; j < num_basis; j++)
         basis(i, j) = UniformRandom();

   IntegrationRule ir = IntRules.Get(fes->GetFE(0)->GetGeomType(),
                                    (int)(ceil(1.5 * (2 * fes->GetMaxElementOrder() - 1))));
   ConstantCoefficient pi(3.141592);
   auto *integ = new VectorConvectionTrilinearFormIntegrator(pi);
   integ->SetIntRule(&ir);

   ROMNonlinearForm *rform(new ROMNonlinearForm(num_basis, fes));
   rform->AddDomainIntegrator(integ);
   rform->SetBasis(basis);

   // we set the full elements/quadrature points,
   // so that the resulting vector is equilvalent to FOM.
   const int nsample = UniformRandom(15, 20);
   const int nqe = ir.GetNPoints();
   const int ne = fes->GetNE();
   Array<double> const& w_el = ir.GetWeights();
   Array<int> sample_el(nsample), sample_qp(nsample);
   Array<double> sample_qw(nsample);
   for (int s = 0; s < nsample; s++)
   {
      sample_el[s] = UniformRandom(0, ne-1);
      sample_qp[s] = UniformRandom(0, nqe-1);
      sample_qw[s] = UniformRandom();
   }
   rform->UpdateDomainIntegratorSampling(0, sample_el, sample_qp, sample_qw);
   rform->PrecomputeCoefficients();

   Vector rom_u(num_basis);
   for (int k = 0; k < rom_u.Size(); k++)
      rom_u(k) = UniformRandom();

   Vector rom_y(num_basis), rom_yfast(num_basis);
   rform->Mult(rom_u, rom_y);

   rform->SetPrecomputeMode(true);
   rform->Mult(rom_u, rom_yfast);
   for (int k = 0; k < rom_y.Size(); k++)
      EXPECT_NEAR(rom_y(k), rom_yfast(k), 1.0e-12);

   DenseMatrix *jac = dynamic_cast<DenseMatrix *>(&(rform->GetGradient(rom_u)));

   double J0 = 0.5 * (rom_y * rom_y);
   Vector grad(num_basis);
   jac->MultTranspose(rom_y, grad);
   double gg = sqrt(grad * grad);
   printf("J0: %.15E\n", J0);
   printf("grad: %.15E\n", gg);

   Vector du(grad);
   du /= gg;

   Vector rom_y1(num_basis);
   const int Nk = 40;
   double error1 = 1e100, error;
   printf("amp\tJ1\tdJdx\terror\n");
   for (int k = 0; k < Nk; k++)
   {
      double amp = pow(10.0, -0.25 * k);
      Vector rom_u1(rom_u);
      rom_u1.Add(amp, du);

      rform->Mult(rom_u1, rom_y1);
      double J1 = 0.5 * (rom_y1 * rom_y1);
      double dJdx = (J1 - J0) / amp;
      error = abs(dJdx - gg) / gg;

      printf("%.5E\t%.5E\t%.5E\t%.5E\n", amp, J1, dJdx, error);
      if (error >= error1)
         break;
      error1 = error;
   }
   EXPECT_TRUE(min(error, error1) < 1.0e-7);

   delete mesh;
   delete h1_coll;
   delete fes;
   delete rform;
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