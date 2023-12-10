// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef MMS_SUITE_HPP
#define MMS_SUITE_HPP

#include "mfem.hpp"
#include "poisson_solver.hpp"
#include "stokes_solver.hpp"
#include "steady_ns_solver.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace mfem;

namespace mms
{

namespace poisson
{

static const double pi = 4.0 * atan(1.0);
static double amp[3];
static double L[3];
static double offset[3];
static double constant;

double ExactSolution(const Vector &);
double ExactRHS(const Vector &);
PoissonSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

}  // namespace poisson

namespace stokes
{

static double nu;

void uFun_ex(const Vector & x, Vector & u);
// void mlap_uFun_ex(const Vector & x, Vector & u)
// {
//    double xi(x(0));
//    double yi(x(1));
//    assert(x.Size() == 2);

//    u(0) = 2.0 * nu * cos(xi)*sin(yi);
//    u(1) = - 2.0 * nu * sin(xi)*cos(yi);
// }
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);

StokesSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence(const double &threshold = 0.1);

}   // namespace stokes

namespace steady_ns
{

static double nu;
static double zeta = 1.0;

void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);

SteadyNSSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence(const double &threshold = 1.0);

}   // namespace steady_ns

namespace fem
{

namespace dg_bdr_normal_lf
{

void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);

double EvalWithRefinement(const int num_refinement);
void CheckConvergence();

}   // namespace dg_bdr_normal_lf

}   // namespace fem

}  // namespace mms

#endif