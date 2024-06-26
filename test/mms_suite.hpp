// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef MMS_SUITE_HPP
#define MMS_SUITE_HPP

#include "mfem.hpp"
#include "poisson_solver.hpp"
#include "stokes_solver.hpp"
#include "steady_ns_solver.hpp"
#include "unsteady_ns_solver.hpp"
#include "linelast_solver.hpp"
#include "advdiff_solver.hpp"
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

double ExactSolution(const Vector &, double);
double ExactRHS(const Vector &, double);
PoissonSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

}  // namespace poisson

namespace stokes
{

static double nu;

void uFun_ex(const Vector & x, double t, Vector & u);
// void mlap_uFun_ex(const Vector & x, Vector & u)
// {
//    double xi(x(0));
//    double yi(x(1));
//    assert(x.Size() == 2);

//    u(0) = 2.0 * nu * cos(xi)*sin(yi);
//    u(1) = - 2.0 * nu * sin(xi)*cos(yi);
// }
double pFun_ex(const Vector & x, double t);
void fFun(const Vector & x, double t, Vector & f);
double gFun(const Vector & x, double t);

StokesSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence(const double &threshold = 0.1);

}   // namespace stokes

namespace steady_ns
{

static double nu;
static double zeta = 1.0;

void uFun_ex(const Vector & x, double t, Vector & u);
double pFun_ex(const Vector & x, double t);
void fFun(const Vector & x, double t, Vector & f);
double gFun(const Vector & x, double t);

SteadyNSSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence(const double &threshold = 1.0);

}   // namespace steady_ns

namespace unsteady_ns
{

UnsteadyNSSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence(const double &threshold = 1.0);

}   // namespace steady_ns

namespace linelast
{

static const double pi = 4.0 * atan(1.0);
static const double mu = 1.0; //only test case when mu = 1.0 (homogenous material)
static double lambda = 1.0;
static int dim;
static void ExactSolution(const Vector & x, double t, Vector & u);
static void ExactRHS(const Vector & x, double t, Vector & u);
LinElastSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

}  // namespace linelast

namespace advdiff
{

static const double pi = 4.0 * atan(1.0);
// solution parameters
static double amp[3];
static double L[3];
static double offset[3];
static double constant;
// flow parameters
static double Pe;
static double du;
static double wn;
static double uoffset[3];
static double u0, v0;

double ExactSolution(const Vector &, double t);
void ExactFlow(const Vector &, double t, Vector &);
double ExactRHS(const Vector &, double t);
AdvDiffSolver *SolveWithRefinement(const int num_refinement);
void CheckConvergence();

}  // namespace advdiff

namespace fem
{

namespace dg_bdr_normal_lf
{

void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);

double EvalWithRefinement(const int num_refinement);
void CheckConvergence();

}   // namespace dg_bdr_normal_lf

namespace dg_temam
{

void uFun_ex(const Vector & x, Vector & u);
void usFun_ex(const Vector & x, Vector & us);

void EvalWithRefinement(const int num_refinement, int &order_out, double &product1, double &product2);
void CheckConvergence();

}   // namespace dg_temam

}   // namespace fem

}  // namespace mms

#endif