// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "main_workflow.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

static const double threshold = 1.0e-14;
static const double stokes_threshold = 1.0e-12;
static const double linelast_threshold = 1.0e-13;

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound)
{
   SUCCEED();
}

TEST(NSTensor, Sampling)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   return;
}

TEST(NSTensor, Train)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   return;
}

TEST(NSTensor, Build_SingleRun)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";

   printf("\nBuild ROM \n\n");

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD, "test_output.h5");

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

TEST(NSEQP, Sampling)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";
   config.dict_["model_reduction"]["nonlinear_handling"] = "eqp";
   config.dict_["model_reduction"]["eqp"]["relative_tolerance"] = 1.0e-11;
   config.dict_["model_reduction"]["eqp"]["precompute"] = true;

   printf("\nSample Generation \n\n");
   
   config.dict_["main"]["mode"] = "sample_generation";
   GenerateSamples(MPI_COMM_WORLD);

   return;
}

TEST(NSEQP, Train)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";
   config.dict_["model_reduction"]["nonlinear_handling"] = "eqp";
   config.dict_["model_reduction"]["eqp"]["relative_tolerance"] = 1.0e-11;
   config.dict_["model_reduction"]["eqp"]["precompute"] = true;

   config.dict_["main"]["mode"] = "train_rom";
   TrainROM(MPI_COMM_WORLD);

   return;
}

TEST(NSEQP, Build_SingleRun)
{
   config = InputParser("inputs/steady_ns.component.yml");
   config.dict_["model_reduction"]["separate_variable_basis"] = true;
   config.dict_["model_reduction"]["linear_solver_type"] = "direct";
   config.dict_["model_reduction"]["linear_system_type"] = "us";
   config.dict_["model_reduction"]["nonlinear_handling"] = "eqp";
   config.dict_["model_reduction"]["eqp"]["relative_tolerance"] = 1.0e-11;
   config.dict_["model_reduction"]["eqp"]["precompute"] = true;

   printf("\nBuild ROM \n\n");

   config.dict_["main"]["mode"] = "build_rom";
   BuildROM(MPI_COMM_WORLD);

   config.dict_["main"]["mode"] = "single_run";
   double error = SingleRun(MPI_COMM_WORLD, "test_output.h5");

   // This reproductive case must have a very small error at the level of finite-precision.
   printf("Error: %.15E\n", error);
   EXPECT_TRUE(error < stokes_threshold);

   return;
}

int main(int argc, char *argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   MPI_Init(&argc, &argv);
   int result = RUN_ALL_TESTS();
   MPI_Finalize();
   return result;
}
