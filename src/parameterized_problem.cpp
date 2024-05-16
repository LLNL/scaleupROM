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

namespace linelast_frame_wind
{

double qwind_f;
double density;
double g;

void gravity_load(const Vector &x, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = -density * g;
}

void wind_load(const Vector &x, Vector &f)
{
   f = 0.0;
   f(0) = qwind_f;
}

void dirichlet(const Vector &x, Vector &u)
{
   u = 0.0;
}

}  // namespace linelast_frame_wind

namespace linelast_lattice_roof
{
double qsnow_f;
double qpoint_f;
double density;
double g;

void gravity_load(const Vector &x, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = -density * g;
}

void snow_load(const Vector &x, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = qsnow_f;
}

void point_load(const Vector &x, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = qpoint_f;
}

void dirichlet(const Vector &x, Vector &u)
{
   u = 0.0;
}

} //namespace linelast_lattice_roof


namespace advdiff_problem
{

bool analytic_flow;
StokesProblem *flow_problem;

namespace advdiff_flow_past_array
{

double q0, dq, qoffset;
Vector qk;

double qbdr(const Vector &x)
{
   assert(qk.Size() >= x.Size());
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += advdiff_flow_past_array::qk(d) * x(d);
   tmp += advdiff_flow_past_array::qoffset;
   return advdiff_flow_past_array::q0 + advdiff_flow_past_array::dq * sin(2.0 * pi * tmp);
}

}  // namespace advdiff_flow_past_array

}  // namespace advdiff_problem

namespace linelast_cwtrain
{
   // Probabilities
   double lx;
   double ly;
   double lz;
   double rx;
   double ry;
   double rz;
   double dx;
   double dy;
   double dz;
   double ux;
   double uy;
   double uz;
   double bx;
   double by;
   double bz;

   // Constant force
   double l_ux;
   double l_uy;
   double l_uz;
   double r_fx;
   double r_fy;
   double r_fz;
   double u_fx;
   double u_fy;
   double u_fz;
   double d_fx;
   double d_fy;
   double d_fz;
   double b_fx;
   double b_fy;
   double b_fz;

   // Amplitudes
   double xu_amp;
   double yu_amp;
   double zu_amp;
   double xf_amp;
   double yf_amp;
   double zf_amp;
   double bxf_amp;
   double byf_amp;
   double bzf_amp;

   // Frequencies
   double xu_freq;
   double yu_freq;
   double zu_freq;
   double xf_freq;
   double yf_freq;
   double zf_freq;
   double bxf_freq;
   double byf_freq;
   double bzf_freq;

   // Sine offsets
   double xu_offset;
   double yu_offset;
   double zu_offset;
   double xf_offset;
   double yf_offset;
   double zf_offset;
   double bxf_offset;
   double byf_offset;
   double bzf_offset;

double perturb_func(const double x, const double amp, const double freq, const double offset)
{
   return amp * sin(pi * freq *( x / 3.0 + 2 * offset) );
}

void left_disp(const Vector &x, Vector &u)
{
   if (lx >= 0.5)
      u(0) = l_ux + perturb_func(x(0), xu_amp, xu_freq, xu_offset);
   if (ly >= 0.5)
      u(1) = l_uy + perturb_func(x(1), yu_amp, yu_freq, yu_offset);
   if (x.Size() == 3)
      if (lz >= 0.5)
         u(2) = l_uz + perturb_func(x(2), zu_amp, zu_freq, zu_offset);

   
}

void right_disp(const Vector &x, Vector &u)
{
   if (rx >= 0.5)
      u(0) = r_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (ry >= 0.5)
      u(1) = r_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
   if (x.Size() == 3)
      if (rz >= 0.5)
         u(2) = r_fz + perturb_func(x(2), zf_amp, zf_freq, zf_offset);
}

void up_disp(const Vector &x, Vector &u)
{
   if (ux >= 0.5)
      u(0) = u_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (uy >= 0.5)
      u(1) = u_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
   if (x.Size() == 3)
      if (uz >= 0.5)
         u(2) = u_fz + perturb_func(x(2), zf_amp, zf_freq, zf_offset);
}

void down_disp(const Vector &x, Vector &u)
{
   if (dx >= 0.5)
      u(0) = d_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (dy >= 0.5)
      u(1) = d_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
   if (x.Size() == 3)
      if (dz >= 0.5)
         u(2) = d_fz + perturb_func(x(2), zf_amp, zf_freq, zf_offset);
}

void body_force(const Vector &x, Vector &u)
{
   if (bx >= 0.5)
      u(0) = b_fx + perturb_func(x(0), bxf_amp, bxf_freq, bxf_offset);
   if (by >= 0.5)
      u(1) = b_fy + perturb_func(x(1), byf_amp, byf_freq, byf_offset);
   if (x.Size() == 3)
      if (bz >= 0.5)
         u(2) = b_fz + perturb_func(x(2), bzf_amp, bzf_freq, bzf_offset);
}

}  // namespace linelast_cwtrain

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
   else if (problem_name == "linelast_cwtrain")
   {
      problem = new LinElastComponentWiseTrain();
   }
   else if (problem_name == "linelast_frame_wind")
   {
      problem = new LinElastFrameWind();
   }
   else if (problem_name == "linelast_lattice_roof")
   {
      problem = new LinElastLatticeRoof();
   }
   else if (problem_name == "advdiff_flow_past_array")
   {
      problem = new AdvDiffFlowPastArray();
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
   bdr_type.SetSize(5);
   battr.SetSize(5);
   vector_bdr_ptr.SetSize(5);

   // Down
   battr[0] = 1;
   bdr_type[0] = BoundaryType::NEUMANN;
   vector_bdr_ptr[0] = NULL;

   // Right
   battr[1] = 2;
   bdr_type[1] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[1] = &(function_factory::linelast_disp::init_disp);

   // Up
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;

   // Left
   battr[3] = 4;
   bdr_type[3] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[3] = &(function_factory::linelast_disp::init_disp);

   // None
   battr[4] = 5;
   bdr_type[4] = BoundaryType::NEUMANN;
   vector_bdr_ptr[4] = NULL;

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
   bdr_type.SetSize(5);
   battr.SetSize(5);
   vector_bdr_ptr.SetSize(5);

   // Down
   battr[0] = 1;
   bdr_type[0] = BoundaryType::NEUMANN;
   vector_bdr_ptr[0] = NULL;

   // Right
   battr[1] = 2;
   bdr_type[1] = BoundaryType::NEUMANN;
   vector_bdr_ptr[1] = &(function_factory::linelast_force::tip_force);

   // Up
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;

   // Left
   battr[3] = 4;
   bdr_type[3] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[3] = &(function_factory::linelast_disp::init_disp);

   // None
   battr[4] = 5;
   bdr_type[4] = BoundaryType::NEUMANN;
   vector_bdr_ptr[4] = NULL;
   
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

LinElastFrameWind::LinElastFrameWind()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(5);
   battr.SetSize(5);
   vector_bdr_ptr.SetSize(5);

   // battr 1: Wind load
   battr[0] = 1;
   bdr_type[0] = BoundaryType::NEUMANN;
   vector_bdr_ptr[0] = &(function_factory::linelast_frame_wind::wind_load);

   // battr 2: Line load (to be implemented)
   battr[1] = 2;
   bdr_type[1] = BoundaryType::NEUMANN;
   vector_bdr_ptr[1] = NULL;

   // battr 3: Other load (To be implemented)
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = NULL;

   // battr 4: Dirichlet BCs
   battr[3] = 4;
   bdr_type[3] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[3] = &(function_factory::linelast_frame_wind::dirichlet);

   // battr 5: Unloaded
   battr[4] = 5; // maybe change back to 3?
   bdr_type[4] = BoundaryType::NEUMANN;
   vector_bdr_ptr[4] = NULL;

   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;

   // Default values.
   function_factory::linelast_problem::_lambda = 3846153846.0;
   function_factory::linelast_problem::_mu = 769230769.0;
   function_factory::linelast_frame_wind::qwind_f = 500.0; // [N]
   function_factory::linelast_frame_wind::density = 78.5; //[kg/m2]
   function_factory::linelast_frame_wind::g = 9.81; 

   param_map["lambda"] = 0;
   param_map["mu"] = 1;
   param_map["qwind_f"] = 2;
   param_map["density"] = 3;
   param_map["g"] = 4;

   param_ptr.SetSize(5);
   param_ptr[0] = &(function_factory::linelast_problem::_lambda);
   param_ptr[1] = &(function_factory::linelast_problem::_mu);
   param_ptr[2] = &(function_factory::linelast_frame_wind::qwind_f);
   param_ptr[3] = &(function_factory::linelast_frame_wind::density);
   param_ptr[4] = &(function_factory::linelast_frame_wind::g);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = NULL; // for now, change if current params doesn't work well enough.

   vector_rhs_ptr = &(function_factory::linelast_frame_wind::gravity_load);

}

LinElastLatticeRoof::LinElastLatticeRoof()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(5);
   battr.SetSize(5);
   vector_bdr_ptr.SetSize(5);

   // battr 1: Dirichlet BCs
   battr[0] = 1;
   bdr_type[0] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[0] = &(function_factory::linelast_lattice_roof::dirichlet);

   // battr 2: Point load
   battr[1] = 2;
   bdr_type[1] = BoundaryType::NEUMANN;
   vector_bdr_ptr[1] = &(function_factory::linelast_lattice_roof::point_load);

   // battr 3: Snow load
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = &(function_factory::linelast_lattice_roof::point_load);

   // battr 4: Unloaded
   battr[3] = 4;
   bdr_type[3] = BoundaryType::NEUMANN;
   vector_bdr_ptr[3] = NULL;

   // battr 5: Unloaded
   battr[4] = 5; // maybe change back to 3?
   bdr_type[4] = BoundaryType::NEUMANN;
   vector_bdr_ptr[4] = NULL;

   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;

   // Default values.
   function_factory::linelast_problem::_lambda = 384615384615.3846;
   function_factory::linelast_problem::_mu = 76923076923.07692;
   function_factory::linelast_lattice_roof::qsnow_f = 2000.0; // [N/m2]
   function_factory::linelast_lattice_roof::qpoint_f = 100000.0; // [N/m2]
   function_factory::linelast_lattice_roof::density = 78.0; //[kg/m2]
   function_factory::linelast_lattice_roof::g = 9.81; 

   param_map["lambda"] = 0;
   param_map["mu"] = 1;
   param_map["qsnow_f"] = 2;
   param_map["qpoint_f"] = 3;
   param_map["density"] = 4;
   param_map["g"] = 5;

   param_ptr.SetSize(6);
   param_ptr[0] = &(function_factory::linelast_problem::_lambda);
   param_ptr[1] = &(function_factory::linelast_problem::_mu);
   param_ptr[2] = &(function_factory::linelast_lattice_roof::qsnow_f);
   param_ptr[3] = &(function_factory::linelast_lattice_roof::qpoint_f);
   param_ptr[4] = &(function_factory::linelast_lattice_roof::density);
   param_ptr[5] = &(function_factory::linelast_lattice_roof::g);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = NULL; // for now, change if current params doesn't work well enough.

   vector_rhs_ptr = &(function_factory::linelast_lattice_roof::gravity_load);
}

/*
   AdvDiffFlowPastArray
*/

AdvDiffFlowPastArray::AdvDiffFlowPastArray()
   : StokesFlowPastArray(), flow_problem(new StokesFlowPastArray)
{
   function_factory::advdiff_problem::analytic_flow = false;
   function_factory::advdiff_problem::flow_problem = flow_problem;

   bdr_type.SetSize(5);
   bdr_type = BoundaryType::DIRICHLET;
   bdr_type[4] = BoundaryType::NEUMANN;

   scalar_rhs_ptr = NULL;
   scalar_bdr_ptr.SetSize(5);
   scalar_bdr_ptr = &(function_factory::advdiff_problem::advdiff_flow_past_array::qbdr);

   // q0 + dq + qoffset + qk(3)
   param_num += 1 + 1 + 1 + 3;
   const int p0 = flow_problem->GetNumParams();

   param_map["q0"] = p0;
   param_map["dq"] = p0 + 1;
   param_map["qoffset"] = p0 + 2;
   param_map["qk_x"] = p0 + 3;
   param_map["qk_y"] = p0 + 4;
   param_map["qk_z"] = p0 + 5;

   // default values.
   function_factory::advdiff_problem::advdiff_flow_past_array::q0 = 1.0;
   function_factory::advdiff_problem::advdiff_flow_past_array::dq = 0.1;
   function_factory::advdiff_problem::advdiff_flow_past_array::qoffset = 0.0;
   function_factory::advdiff_problem::advdiff_flow_past_array::qk.SetSize(3);
   function_factory::advdiff_problem::advdiff_flow_past_array::qk = 0.0;

   param_ptr.Append(&(function_factory::advdiff_problem::advdiff_flow_past_array::q0));
   param_ptr.Append(&(function_factory::advdiff_problem::advdiff_flow_past_array::dq));
   param_ptr.Append(&(function_factory::advdiff_problem::advdiff_flow_past_array::qoffset));
   for (int j = 0; j < 3; j++)
      param_ptr.Append(&(function_factory::advdiff_problem::advdiff_flow_past_array::qk(j)));
}

AdvDiffFlowPastArray::~AdvDiffFlowPastArray()
{
   delete flow_problem;
}

LinElastComponentWiseTrain::LinElastComponentWiseTrain()
    : LinElastProblem()
{
   // pointer to static function.
   bdr_type.SetSize(5);
   battr.SetSize(5);
   vector_bdr_ptr.SetSize(5);

   // Down
   battr[0] = 1;
   bdr_type[0] = BoundaryType::NEUMANN;
   vector_bdr_ptr[0] = &(function_factory::linelast_cwtrain::down_disp);

   // Right
   battr[1] = 2;
   bdr_type[1] = BoundaryType::NEUMANN;
   vector_bdr_ptr[1] = &(function_factory::linelast_cwtrain::right_disp);

   // Up
   battr[2] = 3;
   bdr_type[2] = BoundaryType::NEUMANN;
   vector_bdr_ptr[2] = &(function_factory::linelast_cwtrain::up_disp);

   // Left
   battr[3] = 4;
   bdr_type[3] = BoundaryType::DIRICHLET;
   vector_bdr_ptr[3] = &(function_factory::linelast_cwtrain::left_disp);

   // None
   battr[4] = 5;
   bdr_type[4] = BoundaryType::NEUMANN;
   vector_bdr_ptr[4] = NULL;

   // Set materials
   general_scalar_ptr.SetSize(2);
   general_scalar_ptr[0] = function_factory::linelast_problem::lambda;
   general_scalar_ptr[1] = function_factory::linelast_problem::mu;

   // Probabilities default values
   function_factory::linelast_cwtrain::lx = 0.0;
   function_factory::linelast_cwtrain::ly = 0.0;
   function_factory::linelast_cwtrain::lz = 0.0;
   function_factory::linelast_cwtrain::rx = 0.0;
   function_factory::linelast_cwtrain::ry = 0.0;
   function_factory::linelast_cwtrain::rz = 0.0;
   function_factory::linelast_cwtrain::dx = 0.0;
   function_factory::linelast_cwtrain::dy = 0.0;
   function_factory::linelast_cwtrain::dz = 0.0;
   function_factory::linelast_cwtrain::ux = 0.0;
   function_factory::linelast_cwtrain::uy = 0.0;
   function_factory::linelast_cwtrain::uz = 0.0;
   function_factory::linelast_cwtrain::bx = 0.0;
   function_factory::linelast_cwtrain::by = 0.0;
   function_factory::linelast_cwtrain::bz = 0.0;

   // Constant force default values
   function_factory::linelast_cwtrain::l_ux= 0.0;
   function_factory::linelast_cwtrain::l_uy = 0.0;
   function_factory::linelast_cwtrain::l_uz = 0.0;
   function_factory::linelast_cwtrain::r_fx = 0.0;
   function_factory::linelast_cwtrain::r_fy = 0.0;
   function_factory::linelast_cwtrain::r_fz = 0.0;
   function_factory::linelast_cwtrain::u_fx = 0.0;
   function_factory::linelast_cwtrain::u_fy = 0.0;
   function_factory::linelast_cwtrain::u_fz = 0.0;
   function_factory::linelast_cwtrain::d_fx = 0.0;
   function_factory::linelast_cwtrain::d_fy = 0.0;
   function_factory::linelast_cwtrain::d_fz = 0.0;
   function_factory::linelast_cwtrain::b_fx = 0.0;
   function_factory::linelast_cwtrain::b_fy = 0.0;
   function_factory::linelast_cwtrain::b_fz = 0.0;

   // Amplitudes default values
   function_factory::linelast_cwtrain::xu_amp = 1.0;
   function_factory::linelast_cwtrain::yu_amp = 1.0;
   function_factory::linelast_cwtrain::zu_amp = 1.0;
   function_factory::linelast_cwtrain::xf_amp = 1.0;
   function_factory::linelast_cwtrain::yf_amp = 1.0;
   function_factory::linelast_cwtrain::zf_amp = 1.0;
   function_factory::linelast_cwtrain::bxf_amp = 1.0;
   function_factory::linelast_cwtrain::byf_amp = 1.0;
   function_factory::linelast_cwtrain::bzf_amp = 1.0;

   // Frequencies default values
   function_factory::linelast_cwtrain::xu_freq = 1.0;
   function_factory::linelast_cwtrain::yu_freq = 1.0;
   function_factory::linelast_cwtrain::zu_freq = 1.0;
   function_factory::linelast_cwtrain::xf_freq = 1.0;
   function_factory::linelast_cwtrain::yf_freq = 1.0;
   function_factory::linelast_cwtrain::zf_freq = 1.0;
   function_factory::linelast_cwtrain::bxf_freq = 1.0;
   function_factory::linelast_cwtrain::byf_freq = 1.0;
   function_factory::linelast_cwtrain::bzf_freq = 1.0;

   // Sine offsets default values
   function_factory::linelast_cwtrain::xu_offset = 0.0;
   function_factory::linelast_cwtrain::yu_offset = 0.0;
   function_factory::linelast_cwtrain::zu_offset = 0.0;
   function_factory::linelast_cwtrain::xf_offset = 0.0;
   function_factory::linelast_cwtrain::yf_offset = 0.0;
   function_factory::linelast_cwtrain::zf_offset = 0.0;
   function_factory::linelast_cwtrain::bxf_offset = 0.0;
   function_factory::linelast_cwtrain::byf_offset = 0.0;
   function_factory::linelast_cwtrain::bzf_offset = 0.0;
   
   // Material parameters default values
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   // Parameter map
   param_map["l_ux"] = 0;
   param_map["l_uy"] = 1;
   param_map["l_uz"] = 2;
   param_map["r_fx"] = 3;
   param_map["r_fy"] = 4;
   param_map["r_fz"] = 5;
   param_map["u_fx"] = 6;
   param_map["u_fy"] = 7;
   param_map["u_fz"] = 8;
   param_map["d_fx"] = 9;
   param_map["d_fy"] = 10;
   param_map["d_fz"] = 11;
   param_map["lx"] = 12;
   param_map["ly"] = 13;
   param_map["lz"] = 14;
   param_map["rx"] = 15;
   param_map["ry"] = 16;
   param_map["rz"] = 17;
   param_map["dx"] = 18;
   param_map["dy"] = 19;
   param_map["dz"] = 20;
   param_map["ux"] = 21;
   param_map["uy"] = 22;
   param_map["uz"] = 23;
   param_map["lambda"] = 24;
   param_map["mu"] = 25;
   param_map["xu_amp"] = 26;
   param_map["xu_freq"] = 27;
   param_map["xu_offset"] = 28;
   param_map["xf_amp"] = 29;
   param_map["xf_freq"] = 30;
   param_map["xf_offset"] = 31;
   param_map["yu_amp"] = 32;
   param_map["yu_freq"] = 33;
   param_map["yu_offset"] = 34;
   param_map["yf_amp"] = 35;
   param_map["yf_freq"] = 36;
   param_map["yf_offset"] = 37;
   param_map["zu_amp"] = 38;
   param_map["zu_freq"] = 39;
   param_map["zu_offset"] = 40;
   param_map["zf_amp"] = 41;
   param_map["zf_freq"] = 42;
   param_map["zf_offset"] = 43;
   param_map["b_fx"] = 44;
   param_map["b_fy"] = 45;
   param_map["b_fz"] = 46;
   param_map["bxf_amp"] = 47;
   param_map["bxf_freq"] = 48;
   param_map["bxf_offset"] = 49;
   param_map["byf_amp"] = 50;
   param_map["byf_freq"] = 51;
   param_map["byf_offset"] = 52;
   param_map["bzf_amp"] = 53;
   param_map["bzf_freq"] = 54;
   param_map["bzf_offset"] = 55;
   param_map["bx"] = 56;
   param_map["by"] = 57;
   param_map["bz"] = 58;

   param_ptr.SetSize(59);
   param_ptr[0] = &(function_factory::linelast_cwtrain::l_ux);
   param_ptr[1] = &(function_factory::linelast_cwtrain::l_uy);
   param_ptr[2] = &(function_factory::linelast_cwtrain::l_uz);
   param_ptr[3] = &(function_factory::linelast_cwtrain::r_fx);
   param_ptr[4] = &(function_factory::linelast_cwtrain::r_fy);
   param_ptr[5] = &(function_factory::linelast_cwtrain::r_fz);
   param_ptr[6] = &(function_factory::linelast_cwtrain::u_fx);
   param_ptr[7] = &(function_factory::linelast_cwtrain::u_fy);
   param_ptr[8] = &(function_factory::linelast_cwtrain::u_fz);
   param_ptr[9] = &(function_factory::linelast_cwtrain::d_fx);
   param_ptr[10] = &(function_factory::linelast_cwtrain::d_fy);
   param_ptr[11] = &(function_factory::linelast_cwtrain::d_fz);
   param_ptr[12] = &(function_factory::linelast_cwtrain::lx);
   param_ptr[13] = &(function_factory::linelast_cwtrain::ly);
   param_ptr[14] = &(function_factory::linelast_cwtrain::lz);
   param_ptr[15] = &(function_factory::linelast_cwtrain::rx);
   param_ptr[16] = &(function_factory::linelast_cwtrain::ry);
   param_ptr[17] = &(function_factory::linelast_cwtrain::rz);
   param_ptr[18] = &(function_factory::linelast_cwtrain::dx);
   param_ptr[19] = &(function_factory::linelast_cwtrain::dy);
   param_ptr[20] = &(function_factory::linelast_cwtrain::dz);
   param_ptr[21] = &(function_factory::linelast_cwtrain::ux);
   param_ptr[22] = &(function_factory::linelast_cwtrain::uy);
   param_ptr[23] = &(function_factory::linelast_cwtrain::uz);
   param_ptr[24] = &(function_factory::linelast_problem::_lambda);
   param_ptr[25] = &(function_factory::linelast_problem::_mu);
   param_ptr[26] = &(function_factory::linelast_cwtrain::xu_amp);
   param_ptr[27] = &(function_factory::linelast_cwtrain::xu_freq);
   param_ptr[28] = &(function_factory::linelast_cwtrain::xu_offset);
   param_ptr[29] = &(function_factory::linelast_cwtrain::xf_amp);
   param_ptr[30] = &(function_factory::linelast_cwtrain::xf_freq);
   param_ptr[31] = &(function_factory::linelast_cwtrain::xf_offset);
   param_ptr[32] = &(function_factory::linelast_cwtrain::yu_amp);
   param_ptr[33] = &(function_factory::linelast_cwtrain::yu_freq);
   param_ptr[34] = &(function_factory::linelast_cwtrain::yu_offset);
   param_ptr[35] = &(function_factory::linelast_cwtrain::yf_amp);
   param_ptr[36] = &(function_factory::linelast_cwtrain::yf_freq);
   param_ptr[37] = &(function_factory::linelast_cwtrain::yf_offset);
   param_ptr[38] = &(function_factory::linelast_cwtrain::zu_amp);
   param_ptr[39] = &(function_factory::linelast_cwtrain::zu_freq);
   param_ptr[40] = &(function_factory::linelast_cwtrain::zu_offset);
   param_ptr[41] = &(function_factory::linelast_cwtrain::zf_amp);
   param_ptr[42] = &(function_factory::linelast_cwtrain::zf_freq);
   param_ptr[43] = &(function_factory::linelast_cwtrain::zf_offset);

   param_ptr[44] = &(function_factory::linelast_cwtrain::b_fx);
   param_ptr[45] = &(function_factory::linelast_cwtrain::b_fy);
   param_ptr[46] = &(function_factory::linelast_cwtrain::b_fz);
   param_ptr[47] = &(function_factory::linelast_cwtrain::bxf_amp);
   param_ptr[48] = &(function_factory::linelast_cwtrain::bxf_freq);
   param_ptr[49] = &(function_factory::linelast_cwtrain::bxf_offset);
   param_ptr[50] = &(function_factory::linelast_cwtrain::byf_amp);
   param_ptr[51] = &(function_factory::linelast_cwtrain::byf_freq);
   param_ptr[52] = &(function_factory::linelast_cwtrain::byf_offset);
   param_ptr[53] = &(function_factory::linelast_cwtrain::bzf_amp);
   param_ptr[54] = &(function_factory::linelast_cwtrain::bzf_freq);
   param_ptr[55] = &(function_factory::linelast_cwtrain::bzf_offset);
   param_ptr[56] = &(function_factory::linelast_cwtrain::bx);
   param_ptr[57] = &(function_factory::linelast_cwtrain::by);
   param_ptr[58] = &(function_factory::linelast_cwtrain::bz);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = &(function_factory::linelast_cwtrain::body_force); 
}
