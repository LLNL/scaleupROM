// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "parameterized_problem.hpp"
#include "input_parser.hpp"
#include <cmath>

using namespace mfem;
using namespace std;

namespace function_factory
{

namespace poisson0
{

double k, offset;

double rhs(const Vector &x)
{
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += x(d);
   tmp *= poisson0::k;
   tmp += poisson0::offset;
   return sin(2.0 * pi * tmp);
}

}  // namespace poisson0

namespace poisson_component
{

Vector k(3), bdr_k(3);
double offset, bdr_offset;
double bdr_idx;

double bdr(const Vector &x)
{
   assert(bdr_k.Size() >= x.Size());
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += poisson_component::bdr_k(d) * x(d);
   tmp += poisson_component::bdr_offset;
   return sin(2.0 * pi * tmp);
}

double rhs(const Vector &x)
{
   assert(k.Size() >= x.Size());
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += poisson_component::k(d) * x(d);
   tmp += poisson_component::offset;
   return sin(2.0 * pi * tmp);
}

}  // namespace poisson_component

namespace poisson_spiral
{

double L, Lw, k, s;

double rhs(const Vector &x)
{
   double r = 0.0;
   for (int d = 0; d < x.Size(); d++) r += (x(d) - 0.5 * L) * (x(d) - 0.5 * L);
   r = sqrt(r);

   double theta = atan2(x(1) - 0.5 * L, x(0) - 0.5 * L);
   if (theta < 0.0) theta += 2.0 * pi; // in [0, 2*pi]

   // r = theta * 0.6 * L / (2.0 * pi * N);
   const double slope = s * L / (2.0 * pi * N);
   double tmp = r - slope * (theta - 2.0 * pi);
   double dist = abs(tmp);
   for (int n = 0; n < N; n++)
   {
      tmp -= 2.0 * pi * slope;
      dist = min(dist, abs(tmp));
   }

   return exp( - 0.5 * dist * dist / Lw / Lw ) * cos( 2.0 * pi *  k * dist );
}

}  // namespace poisson_spiral

namespace stokes_problem
{

double nu;
double del_u;
Vector x0;

void dir(const Vector &x, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = x;

   assert(x0.Size() == dim);
   y -= x0;
}

void flux(const Vector &x, Vector &y)
{
   dir(x, y);
   y *= del_u;
}

namespace stokes_channel
{

double L, U, x0;

void ubdr(const Vector &x, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = 0.0;

   double yc = (x(1) - x0) / L;
   y(0) = U * (1.0 - 4.0 * yc * yc);
}

}  // namespace stokes_channel

namespace stokes_component
{

Vector u0, du, offsets;
DenseMatrix k;

void ubdr(const Vector &x, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   
   for (int i = 0; i < dim; i++)
   {
      double kx = 0.0;
      for (int j = 0; j < dim; j++) kx += k(j, i) * x(j);
      kx -= offsets(i);
      kx *= 2.0 * function_factory::pi;
      y(i) = u0(i) + du(i) * sin(kx);
   }

   // ensure incompressibility.
   Vector del_u(dim);
   stokes_problem::flux(x, del_u);
   y -= del_u;
}

}  // namespace stokes_component

} // namespace stokes_problem

namespace linelast_problem
{

double _lambda;
double _mu;

double lambda(const Vector &x){return _lambda;};
double mu(const Vector &x){return _mu;};

}

namespace linelast_disp
{

double rdisp_f;

void init_disp(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0)*rdisp_f;
}

void init_disp_lcantilever(const Vector &x, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*(x(u.Size()-1) - 5.0)*rdisp_f;
}

}  // namespace linelast_disp

namespace linelast_force
{

double rforce_f;

void tip_force(const Vector &x, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = -1.0e-2* rforce_f;
}

}  // namespace linelast_force

}  // namespace function_factory

ParameterizedProblem::ParameterizedProblem()
   : problem_name(config.GetRequiredOption<std::string>("parameterized_problem/name"))
{ 
   battr.SetSize(1); battr = -1;
   bdr_type.SetSize(1); bdr_type = BoundaryType::NUM_BDR_TYPE;

   scalar_bdr_ptr.SetSize(1);
   vector_bdr_ptr.SetSize(1);

   scalar_bdr_ptr = NULL;
   vector_bdr_ptr = NULL;
};

void ParameterizedProblem::SetParams(const std::string &key, const double &value)
{
   if (!param_map.count(key))
   {
      std::string msg = problem_name + ": unknown parameter name!\n";
      mfem_error(msg.c_str());
   }

   (*param_ptr[param_map[key]]) = value;
}

void ParameterizedProblem::SetParams(const Array<int> &indexes, const Vector &values)
{
   assert(indexes.Size() <= param_num);
   assert(indexes.Size() == values.Size());

   for (int idx = 0; idx < indexes.Size(); idx++)
      (*param_ptr[indexes[idx]]) = values(idx);
}

void ParameterizedProblem::SetSingleRun()
{
   std::string problem_name = GetProblemName();
   std::string param_list_str("single_run/" + problem_name);
   YAML::Node param_list = config.FindNode(param_list_str);
   if (!param_list) mfem_error("Single Run - cannot find the problem name!\n");

   // size_t num_params = param_list.size();
   // for (int p = 0; p < num_params; p++)
   // {
   //    std::string param_name = config.GetRequiredOptionFromDict<std::string>("parameter_name", param_list[p]);
   //    double value = config.GetRequiredOptionFromDict<double>("value", param_list[p]);
   //    SetParams(param_name, value);
   // }

   for(YAML::const_iterator it=param_list.begin(); it != param_list.end(); ++it)
   {
      std::string param_name = it->first.as<std::string>();
      double value = it->second.as<double>();
      SetParams(param_name, value);
   }
}

ParameterizedProblem* InitParameterizedProblem()
{
   ParameterizedProblem *problem = NULL;
   std::string problem_name = config.GetRequiredOption<std::string>("parameterized_problem/name");

   if (problem_name == "poisson0")
   {
      problem = new Poisson0();
   }
   else if (problem_name == "poisson_component")
   {
      problem = new PoissonComponent();
   }
   else if (problem_name == "poisson_spiral")
   {
      problem = new PoissonSpiral();
   }
   else if (problem_name == "stokes_channel")
   {
      problem = new StokesChannel();
   }
   else if (problem_name == "stokes_component")
   {
      problem = new StokesComponent();
   }
   else if (problem_name == "stokes_flow_past_array")
   {
      problem = new StokesFlowPastArray();
   }
   else if (problem_name == "linelast_disp")
   {
      problem = new LinElastDisp();
   }
   else if (problem_name == "linelast_disp_lcantilever")
   {
      problem = new LinElastDispLCantilever();
   }
   else if (problem_name == "linelast_disp_lattice")
   {
      problem = new LinElastDispLattice();
   }
   else if (problem_name == "linelast_force_cantilever")
   {
      problem = new LinElastForceCantilever();
   }
   else
   {
      mfem_error("Unknown parameterized problem name!\n");
   }

   return problem;
}

/*
   Poisson0
*/

Poisson0::Poisson0()
   : PoissonProblem()
{
   param_num = 2;
   battr = -1;
   bdr_type = BoundaryType::ZERO;

   scalar_bdr_ptr.SetSize(1);
   vector_bdr_ptr.SetSize(1);

   // pointer to static function.
   scalar_bdr_ptr = NULL;
   scalar_rhs_ptr = &(function_factory::poisson0::rhs);

   // Default values.
   function_factory::poisson0::k = 1.0;
   function_factory::poisson0::offset = 0.0;

   param_map["k"] = 0;
   param_map["offset"] = 1;

   param_ptr.SetSize(2);
   param_ptr[0] = &(function_factory::poisson0::k);
   param_ptr[1] = &(function_factory::poisson0::offset);
}

/*
   PoissonComponent
*/

PoissonComponent::PoissonComponent()
   : PoissonProblem()
{
   // k (max 3) + offset (1) + bdr_k (max 3) + bdr_offset(1) + bdr_idx(1)
   param_num = 9;
   battr = -1;
   bdr_type = BoundaryType::DIRICHLET;

   // pointer to static function.
   scalar_rhs_ptr = &(function_factory::poisson_component::rhs);
   scalar_bdr_ptr = &(function_factory::poisson_component::bdr);

   // Default values: a constant right-hand side with homogeneous Dirichlet BC.
   function_factory::poisson_component::k = 0.0;
   function_factory::poisson_component::offset = 0.1;
   function_factory::poisson_component::bdr_k = 0.0;
   function_factory::poisson_component::bdr_offset = 0.0;
   function_factory::poisson_component::bdr_idx = -1.0;

   for (int d = 0; d < 3; d++)
   {
      param_map["k" + std::to_string(d)] = d;
      param_map["bdr_k" + std::to_string(d)] = d + 4;
   }
   param_map["offset"] = 3;
   param_map["bdr_offset"] = 7;
   param_map["bdr_idx"] = 8;

   param_ptr.SetSize(param_num);
   for (int d = 0; d < 3; d++)
   {
      param_ptr[d] = &(function_factory::poisson_component::k[d]);
      param_ptr[d + 4] = &(function_factory::poisson_component::bdr_k[d]);
   }
   param_ptr[3] = &(function_factory::poisson_component::offset);
   param_ptr[7] = &(function_factory::poisson_component::bdr_offset);
   param_ptr[8] = &(function_factory::poisson_component::bdr_idx);
}

void PoissonComponent::SetBattr()
{
   double bidx = function_factory::poisson_component::bdr_idx;
   battr.SetSize(1);
   battr = -1;
   if (bidx >= 0.0)
   {
      battr = 1 + floor(bidx);
      assert((battr[0] >= 1) && (battr[0] <= 4));
   }
}

void PoissonComponent::SetParams(const std::string &key, const double &value)
{
   ParameterizedProblem::SetParams(key, value);
   SetBattr();
}

void PoissonComponent::SetParams(const Array<int> &indexes, const Vector &values)
{
   ParameterizedProblem::SetParams(indexes, values);
   SetBattr();
}

/*
   PoissonSpiral
*/

PoissonSpiral::PoissonSpiral()
   : PoissonProblem()
{
   param_num = 4;
   battr = -1;
   bdr_type = BoundaryType::ZERO;

   // pointer to static function.
   scalar_bdr_ptr = NULL;
   scalar_rhs_ptr = &(function_factory::poisson_spiral::rhs);

   // Default values.
   function_factory::poisson_spiral::L = 1.0;
   function_factory::poisson_spiral::Lw = 0.2;
   function_factory::poisson_spiral::k = 1.0;
   function_factory::poisson_spiral::s = 0.6;

   param_map["L"] = 0;
   param_map["Lw"] = 1;
   param_map["k"] = 2;
   param_map["s"] = 3;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::poisson_spiral::L);
   param_ptr[1] = &(function_factory::poisson_spiral::Lw);
   param_ptr[2] = &(function_factory::poisson_spiral::k);
   param_ptr[3] = &(function_factory::poisson_spiral::s);
}

/*
   StokesChannel
*/

StokesChannel::StokesChannel()
   : StokesProblem()
{
   battr.SetSize(4);
   battr[0] = 1;
   battr[1] = 3;
   battr[2] = 4;
   battr[3] = 5;
   bdr_type.SetSize(4);
   bdr_type = BoundaryType::ZERO;
   bdr_type[2] = BoundaryType::DIRICHLET;

   // pointer to static function.
   vector_bdr_ptr.SetSize(4);
   vector_rhs_ptr = NULL;
   vector_bdr_ptr = &(function_factory::stokes_problem::stokes_channel::ubdr);

   param_num = 4;

   // Default values.
   function_factory::stokes_problem::nu = 1.0;
   function_factory::stokes_problem::stokes_channel::L = 1.0;
   function_factory::stokes_problem::stokes_channel::U = 1.0;
   function_factory::stokes_problem::stokes_channel::x0 = 0.5;

   param_map["nu"] = 0;
   param_map["L"] = 1;
   param_map["U"] = 2;
   param_map["x0"] = 3;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::stokes_problem::nu);
   param_ptr[1] = &(function_factory::stokes_problem::stokes_channel::L);
   param_ptr[2] = &(function_factory::stokes_problem::stokes_channel::U);
   param_ptr[3] = &(function_factory::stokes_problem::stokes_channel::x0);
}

StokesComponent::StokesComponent()
   : StokesProblem()
{
   battr.SetSize(5);
   for (int b = 0; b < 5; b++)
      battr[b] = b+1;
   bdr_type.SetSize(5);
   bdr_type = BoundaryType::DIRICHLET;
   bdr_type[4] = BoundaryType::ZERO;

   // pointer to static function.
   vector_bdr_ptr.SetSize(5);
   vector_rhs_ptr = NULL;
   vector_bdr_ptr = &(function_factory::stokes_problem::stokes_component::ubdr);

   param_num = 1 + 3 * 3 + 3 * 3;
   function_factory::stokes_problem::stokes_component::u0.SetSize(3);
   function_factory::stokes_problem::stokes_component::du.SetSize(3);
   function_factory::stokes_problem::stokes_component::offsets.SetSize(3);
   function_factory::stokes_problem::stokes_component::k.SetSize(3);

   // Default values.
   function_factory::stokes_problem::nu = 1.0;
   function_factory::stokes_problem::stokes_component::u0 = 0.0;
   function_factory::stokes_problem::stokes_component::du = 1.0;
   function_factory::stokes_problem::stokes_component::offsets = 0.0;
   function_factory::stokes_problem::stokes_component::k = 1.0;

   std::vector<std::string> xc(3), uc(3);
   xc[0] = "_x";
   xc[1] = "_y";
   xc[2] = "_z";
   uc[0] = "_u";
   uc[1] = "_v";
   uc[2] = "_w";

   param_map["nu"] = 0;
   for (int i = 0; i < 3; i++)
   {
      param_map[std::string("u0") + xc[i]] = i + 1;
      param_map[std::string("du") + xc[i]] = i + 4;
      param_map[std::string("offsets") + xc[i]] = i + 7;
      for (int j = 0; j < 3; j++)
         param_map[std::string("k") + uc[i] + xc[j]] = 10 + 3*i + j;
   }

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::stokes_problem::nu);
   for (int i = 0; i < 3; i++)
   {
      param_ptr[1+i] = &(function_factory::stokes_problem::stokes_component::u0[i]);
      param_ptr[4+i] = &(function_factory::stokes_problem::stokes_component::du[i]);
      param_ptr[7+i] = &(function_factory::stokes_problem::stokes_component::offsets[i]);
      for (int j = 0; j < 3; j++)
         param_ptr[10 + 3*i + j] = &(function_factory::stokes_problem::stokes_component::k(j,i));
   }
}

/*
   StokesFlowPastArray
*/

StokesFlowPastArray::StokesFlowPastArray()
   : StokesComponent(), u0(&function_factory::stokes_problem::stokes_component::u0)
{}

void StokesFlowPastArray::SetParams(const std::string &key, const double &value)
{
   ParameterizedProblem::SetParams(key, value);
   SetBattr();
}

void StokesFlowPastArray::SetParams(const Array<int> &indexes, const Vector &values)
{
   ParameterizedProblem::SetParams(indexes, values);
   SetBattr();
}

void StokesFlowPastArray::SetBattr()
{
   if ((*u0)[0] > 0.0)
   {
      bdr_type[3] = BoundaryType::DIRICHLET;
      bdr_type[1] = BoundaryType::NEUMANN;
   }
   else
   {
      bdr_type[1] = BoundaryType::DIRICHLET;
      bdr_type[3] = BoundaryType::NEUMANN;
   }

   if ((*u0)[1] > 0.0)
   {
      bdr_type[0] = BoundaryType::DIRICHLET;
      bdr_type[2] = BoundaryType::NEUMANN;
   }
   else
   {
      bdr_type[2] = BoundaryType::DIRICHLET;
      bdr_type[0] = BoundaryType::NEUMANN;
   }
}

/*
   LinElastDisp
*/

LinElastDisp::LinElastDisp()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(3);
   battr.SetSize(3);
   vector_bdr_ptr.SetSize(3);
   for (size_t i = 0; i < vector_bdr_ptr.Size(); i++)
   {
      bdr_type[i] = BoundaryType::DIRICHLET;
      battr[i] = i+1;
      vector_bdr_ptr[i] = &(function_factory::linelast_disp::init_disp);
   }

   battr[2] = 3;
   bdr_type[2] = BoundaryType::ZERO;
   vector_bdr_ptr[2] = NULL;
   
   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;

   // Set IC
   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = function_factory::linelast_disp::init_disp;
   
   // Default values.
   function_factory::linelast_disp::rdisp_f = 1.0;
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   param_map["rdisp_f"] = 0;
   param_map["lambda"] = 1;
   param_map["mu"] = 2;

   param_ptr.SetSize(3);
   param_ptr[0] = &(function_factory::linelast_disp::rdisp_f);
   param_ptr[1] = &(function_factory::linelast_problem::_lambda);
   param_ptr[2] = &(function_factory::linelast_problem::_mu);
}

LinElastDispLCantilever::LinElastDispLCantilever()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(3);
   battr.SetSize(3);
   vector_bdr_ptr.SetSize(3);
   for (size_t i = 0; i < 2; i++)
   {
      battr[i] = i+1;
      bdr_type[i] = BoundaryType::DIRICHLET;
      vector_bdr_ptr[i] = &(function_factory::linelast_disp::init_disp_lcantilever);
   }

   /* homogeneous Neumann bc */
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;
   
   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;
   
   // Default values.
   function_factory::linelast_disp::rdisp_f = 1.0;
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   param_map["rdisp_f"] = 0;
   param_map["lambda"] = 1;
   param_map["mu"] = 2;

   param_ptr.SetSize(3);
   param_ptr[0] = &(function_factory::linelast_disp::rdisp_f);
   param_ptr[1] = &(function_factory::linelast_problem::_lambda);
   param_ptr[2] = &(function_factory::linelast_problem::_mu);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = NULL;
}

LinElastDispLattice::LinElastDispLattice()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(3);
   battr.SetSize(3);
   vector_bdr_ptr.SetSize(3);
  for (size_t i = 0; i < 2; i++)
      {
      battr[i] = i+1;
      bdr_type[i] = BoundaryType::DIRICHLET;
      vector_bdr_ptr[i] = &(function_factory::linelast_disp::init_disp);
   }

   /* homogeneous Neumann bc */
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;

   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;

   // Default values.
   function_factory::linelast_disp::rdisp_f = 1.0;
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   param_map["rdisp_f"] = 0;
   param_map["lambda"] = 1;
   param_map["mu"] = 2;

   param_ptr.SetSize(3);
   param_ptr[0] = &(function_factory::linelast_disp::rdisp_f);
   param_ptr[1] = &(function_factory::linelast_problem::_lambda);
   param_ptr[2] = &(function_factory::linelast_problem::_mu);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = NULL;

}


LinElastForceCantilever::LinElastForceCantilever()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(3);
   battr.SetSize(3);
   vector_bdr_ptr.SetSize(3);

   // Fixed end of cantilever
   battr[0] = 1;
   bdr_type[0] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[0] = &(function_factory::linelast_disp::init_disp);

   // Free end of cantilever
   battr[1] = 2;
   bdr_type[1] = BoundaryType::NEUMANN;
   vector_bdr_ptr[1] = &(function_factory::linelast_force::tip_force);

   /* homogeneous Neumann bc */
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;
   
   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;
   
   // Default values.
   function_factory::linelast_force::rforce_f = 1.0;
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   param_map["rforce_f"] = 0;
   param_map["lambda"] = 1;
   param_map["mu"] = 2;

   param_ptr.SetSize(3);
   param_ptr[0] = &(function_factory::linelast_force::rforce_f);
   param_ptr[1] = &(function_factory::linelast_problem::_lambda);
   param_ptr[2] = &(function_factory::linelast_problem::_mu);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = NULL;
}
