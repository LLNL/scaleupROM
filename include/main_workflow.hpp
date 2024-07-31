// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef MAIN_WORKFLOW_HPP
#define MAIN_WORKFLOW_HPP

#include "mfem.hpp"
#include "multiblock_solver.hpp"
#include "random_sample_generator.hpp"

double dbc2(const Vector &, double t);
double dbc4(const Vector &, double t);
void RunExample();

MultiBlockSolver* InitSolver();
SampleGenerator* InitSampleGenerator(MPI_Comm comm);
std::vector<BasisTag> GetGlobalBasisTagList(const TopologyHandlerMode &topol_mode, bool separate_variable_basis);

void GenerateSamples(MPI_Comm comm);
void CollectSamples(SampleGenerator *sample_generator);
void CollectSamplesByPort(SampleGenerator *sample_generator, const std::string &basis_prefix);
void CollectSamplesByBasis(SampleGenerator *sample_generator, const std::string &basis_prefix);
void BuildROM(MPI_Comm comm);
void TrainROM(MPI_Comm comm);
// supremizer-enrichment etc..
void AuxiliaryTrainROM(MPI_Comm comm);
// EQP training, could include hypre-reduction optimization.
void TrainEQP(MPI_Comm comm);
// Input parsing routine to list out all snapshot files for training a basis.
void FindSnapshotFilesForBasis(const BasisTag &basis_tag, const std::string &default_filename, std::vector<std::string> &file_list);
// return relative error if comparing solution.
double SingleRun(MPI_Comm comm, const std::string output_file = "");

#endif
