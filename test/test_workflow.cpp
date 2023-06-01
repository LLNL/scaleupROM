#include<gtest/gtest.h>
#include "main_workflow.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-14;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(Poisson_Workflow, BaseIndividualTest)
{
   config = InputParser("inputs/test.base.yml");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun();

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(Poisson_Workflow, BaseUniversalTest)
{
   config = InputParser("inputs/test.base.yml");

   config.dict_["single_run"]["poisson0"][0]["value"] = 2.0;
   config.dict_["sample_generation"]["poisson0"][0]["sample_size"] = 1;
   config.dict_["model_reduction"]["subdomain_training"] = "universal";
   Array<int> num_basis(1);
   num_basis = 4;
   config.dict_["model_reduction"]["number_of_basis"] = num_basis;
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun();

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(Poisson_Workflow, MFEMIndividualTest)
{
   config = InputParser("inputs/test.base.yml");

   config.dict_["model_reduction"]["rom_handler_type"] = "mfem";
   config.dict_["model_reduction"]["visualization"]["enabled"] = true;
   config.dict_["model_reduction"]["visualization"]["prefix"] = "basis_paraview";

   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun();

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(Poisson_Workflow, MFEMUniversalTest)
{
   config = InputParser("inputs/test.base.yml");

   config.dict_["visualization"]["enabled"] = true;

   config.dict_["model_reduction"]["rom_handler_type"] = "mfem";
   config.dict_["model_reduction"]["visualization"]["enabled"] = true;
   config.dict_["model_reduction"]["visualization"]["prefix"] = "basis_paraview";

   config.dict_["single_run"]["poisson0"][0]["value"] = 2.0;
   config.dict_["sample_generation"]["poisson0"][0]["sample_size"] = 1;
   config.dict_["model_reduction"]["subdomain_training"] = "universal";
   Array<int> num_basis(1);
   num_basis = 4;
   config.dict_["model_reduction"]["number_of_basis"] = num_basis;
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun();

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(Poisson_Workflow, ComponentWiseTest)
{
   config = InputParser("inputs/test.component.yml");

   printf("\nSample Generation \n\n");

   config.dict_["mesh"]["type"] = "submesh";
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   // config.dict_["sample_generation"]["poisson0"][0]["sample_size"] = 4;
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun();

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(Stokes_Workflow, BaseIndividualTest)
{
   config = InputParser("inputs/stokes.base.yml");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   ParameterizedProblem *problem = InitParameterizedProblem();
   MultiBlockSolver *test = NULL;

   test = InitSolver();
   if (!test->UseRom()) mfem_error("ROM must be enabled for BuildROM!\n");
   test->InitVariables();
   // test->InitVisualization();

   // NOTE: you need this to set bc/rhs coefficients!
   // This case, we can use default parameter values of the problem.
   problem->SetSingleRun();
   test->SetParameterizedProblem(problem);

   // TODO: there are skippable operations depending on rom/fom mode.
   test->BuildOperators();
   test->SetupBCOperators();
   test->Assemble();
   
   ROMHandler *rom = test->GetROMHandler();
   if (!rom->UseExistingBasis())
   {
      // TODO: basis for multiple components
      SampleGenerator *sample_generator = InitSampleGenerator(MPI_COMM_WORLD, problem);
      sample_generator->SetParamSpaceSizes();
      const int total_samples = sample_generator->GetTotalSampleSize();
      
      test->FormReducedBasis(total_samples);
      delete sample_generator;
   }
   rom->LoadReducedBasis();

   // config.dict_["main"]["mode"] = "build_rom";
   // BuildROM(MPI_COMM_WORLD);

   // config.dict_["main"]["mode"] = "single_run";
   // double error = SingleRun();

   // // This reproductive case must have a very small error at the level of finite-precision.
   // printf("Error: %.15E\n", error);
   // EXPECT_TRUE(error < threshold);

   return;
}

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}
