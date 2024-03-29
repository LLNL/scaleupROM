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
std::vector<std::string> GetGlobalBasisTagList(const TopologyHandlerMode &topol_mode, const TrainMode &train_mode, bool separate_variable_basis);

void GenerateSamples(MPI_Comm comm);
void BuildROM(MPI_Comm comm);
void TrainROM(MPI_Comm comm);
// supremizer-enrichment, hypre-reduction optimization, etc..
void AuxiliaryTrainROM(MPI_Comm comm, SampleGenerator *sample_generator);
// Input parsing routine to list out all snapshot files for training a basis.
void FindSnapshotFilesForBasis(const std::string &basis_tag, const std::string &default_filename, std::vector<std::string> &file_list);
// return relative error if comparing solution.
double SingleRun(MPI_Comm comm, const std::string output_file = "");

#endif
