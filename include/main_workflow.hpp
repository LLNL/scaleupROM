// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the scaleupROM library. For more information and source code
// availability visit https://lc.llnl.gov/gitlab/chung28/scaleupROM.git.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MAIN_WORKFLOW_HPP
#define MAIN_WORKFLOW_HPP

#include "mfem.hpp"
#include "parameterized_problem.hpp"
#include "random_sample_generator.hpp"

double dbc2(const Vector &);
double dbc4(const Vector &);
void RunExample();

SampleGenerator* InitSampleGenerator(MPI_Comm comm, ParameterizedProblem* problem);
void GenerateSamples(MPI_Comm comm);
void BuildROM(MPI_Comm comm);
// return relative error if comparing solution.
double SingleRun();

#endif
