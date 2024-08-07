// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "main_workflow.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-14;
static const double stokes_threshold = 2.0e-12;
static const double ns_threshold = 1.0e-7;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(ComponentWiseTest, PoissonTest)
{
   config = InputParser("test.component.yml");

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   // config.dict_["sample_generation"]["poisson0"][0]["sample_size"] = 4;
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < threshold);

   return;
}

TEST(ComponentWiseTest, StokesTest)
{
   config = InputParser("stokes.component.yml");

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

TEST(ComponentWiseTest, StokesTest_SeparateVariable)
{
   config = InputParser("stokes.component.yml");

   config.dict_["model_reduction"]["separate_variable_basis"] = true;

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

TEST(ComponentWiseTest, SteadyNSTest)
{
   config = InputParser("stokes.component.yml");

   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["solver"]["print_level"] = 1;
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(ComponentWiseTest, SteadyNSTest_SeparateVariable)
{
   config = InputParser("stokes.component.yml");

   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["solver"]["print_level"] = 1;
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(ComponentWiseTest, SteadyNSTest_SeparateVariable_EQP)
{
   config = InputParser("stokes.component.yml");
   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["solver"]["print_level"] = 1;

   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";
   config.dict_["model_reduction"]["nonlinear_handling"] = "eqp";
   config.dict_["model_reduction"]["eqp"]["relative_tolerance"] = 1.0e-12;

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_eqp";
   TrainEQP(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(SteadyNS_Workflow, LF)
{
   config = InputParser("steadyns.lf.yml");

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_eqp";
   TrainEQP(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD, "test_output.h5");

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(SteadyNS_Workflow, InterfaceEQP)
{
   config = InputParser("steadyns.interface_eqp.yml");

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   config.dict_["sample_generation"]["file_path"]["prefix"] = "stokes0";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["sample_generation"]["file_path"]["prefix"] = "stokes1";
   config.dict_["sample_generation"]["parameters"][0]["sample_size"] = 1;
   config.dict_["sample_generation"]["parameters"][0]["minimum"] = 1.2;
   config.dict_["sample_generation"]["parameters"][0]["maximum"] = 1.2;
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_eqp";
   TrainEQP(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD, "test_output.h5");

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(MultiComponentGlobalROM, StokesTest)
{
   config = InputParser("stokes.component.yml");
   config.dict_["model_reduction"]["save_operator"]["level"] = "global";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

TEST(MultiComponentGlobalROM, StokesTestDirectSolve)
{
   config = InputParser("stokes.component.yml");
   config.dict_["model_reduction"]["save_operator"]["level"] = "global";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "sid";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

TEST(MultiComponentGlobalROM, SteadyNSTestDirectSolve)
{
   config = InputParser("stokes.component.yml");
   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["solver"]["print_level"] = 1;

   config.dict_["model_reduction"]["save_operator"]["level"] = "global";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(MultiComponentGlobalROM, SteadyNSTest_SeparateVariable)
{
   config = InputParser("stokes.component.yml");
   config.dict_["main"]["solver"] = "steady-ns";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "square.tri.mesh";
   config.dict_["solver"]["print_level"] = 1;

   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["save_operator"]["level"] = "global";
   config.dict_["solver"]["direct_solve"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["mesh"]["type"] = "component-wise";
   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD);

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

   return;
}

TEST(UnsteadyNS_Workflow, Periodic)
{
   config = InputParser("usns.periodic.yml");

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "train_eqp";
   TrainEQP(MPI_COMM_WORLD);

   printf("\nBuild ROM \n\n");

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   // config.dict_["solver"]["use_restart"] = true;
   // config.dict_["solver"]["restart_file"] = "usns_restart_00000000.h5";
   double error = SingleRun(MPI_COMM_WORLD, "test_output.h5");

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < ns_threshold);

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
