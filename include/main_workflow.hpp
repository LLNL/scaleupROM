// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef MAIN_WORKFLOW_HPP
#define MAIN_WORKFLOW_HPP

#include "mfem.hpp"
#include "multiblock_solver.hpp"
#include "random_sample_generator.hpp"

double dbc2(const Vector &);
double dbc4(const Vector &);
void RunExample();

MultiBlockSolver* InitSolver();

SampleGenerator* InitSampleGenerator(MPI_Comm comm);
void GenerateSamples(MPI_Comm comm);
void BuildROM(MPI_Comm comm);
void TrainROM(MPI_Comm comm);
// return relative error if comparing solution.
double SingleRun(MPI_Comm comm, const std::string output_file = "");

#endif
