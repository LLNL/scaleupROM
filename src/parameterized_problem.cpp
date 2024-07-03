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

double rhs(const Vector &x, double t)
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

double bdr(const Vector &x, double t)
{
   assert(bdr_k.Size() >= x.Size());
   double tmp = 0.0;
   for (int d = 0; d < x.Size(); d++)
      tmp += poisson_component::bdr_k(d) * x(d);
   tmp += poisson_component::bdr_offset;
   return sin(2.0 * pi * tmp);
}

double rhs(const Vector &x, double t)
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

double rhs(const Vector &x, double t)
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

namespace flow_problem
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

namespace channel_flow
{

double L, U, x0;

void ubdr(const Vector &x, double t, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = 0.0;

   double yc = (x(1) - x0) / L;
   y(0) = U * (1.0 - 4.0 * yc * yc);
}

}  // namespace channel_flow

namespace component_flow
{

Vector u0, du, offsets;
DenseMatrix k;

void ubdr(const Vector &x, double t, Vector &y)
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
   flow_problem::flux(x, del_u);
   y -= del_u;
}

}  // namespace component_flow

namespace backward_facing_step
{

double u0, y0, y1;
Vector amp, ky, freq, t_offset;

void ubdr(const Vector &x, double t, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = 0.0;

   double ygap = (y1 - y0);
   y(0) = - u0 * 4.0 / ygap / ygap * (y0 - x(1)) * (y1 - x(1));

   for (int m = 0; m < amp.Size(); m++)
   {
      if (amp(m) == 0.0) continue;

      y(1) += amp(m) * sin(pi * ky(m) * (x(1) - y0) / ygap)
                     * sin(2. * pi * (freq(m) * t + t_offset(m)));
   }
}

void uic(const Vector &x, double t, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = 0.0;

   double ygap = (y1 - y0);
   y(0) = (x(1) < y0) ? 0.0 : - u0 * 4.0 / ygap / ygap * (y0 - x(1)) * (y1 - x(1));
}

void pic(const Vector &x, double t, Vector &y)
{
   y.SetSize(1);
   y = 0.0;
}

}  // namespace backward_facing_step

namespace lid_driven_cavity
{

double u0, L;

void ubdr(const Vector &x, double t, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = 0.0;

   y(0) = 4.0 / L / L * u0 * x(0) * (L - x(0));
}

}  // namespace lid_driven_cavity

namespace periodic_flow_past_array
{

Vector f;

void force(const Vector &x, double t, Vector &y)
{
   const int dim = x.Size();
   y.SetSize(dim);
   y = f;
}

}  // namespace periodic_flow_past_array

} // namespace flow_problem

namespace linelast_problem
{

double _lambda;
double _mu;

double lambda(const Vector &x, double t){return _lambda;};
double mu(const Vector &x, double t){return _mu;};

}

namespace linelast_disp
{

double rdisp_f;

void init_disp(const Vector &x, double t, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*x(0)*rdisp_f;
}

void init_disp_lcantilever(const Vector &x, double t, Vector &u)
{
   u = 0.0;
   u(u.Size()-1) = -0.2*(x(u.Size()-1) - 5.0)*rdisp_f;
}

}  // namespace linelast_disp

namespace linelast_force
{

double rforce_f;

void tip_force(const Vector &x, double t, Vector &f)
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

void gravity_load(const Vector &x, double t, Vector &f)
{
   f = 0.0;
   f(f.Size()-1) = -density * g;
}

void wind_load(const Vector &x, double t, Vector &f)
{
   f = 0.0;
   f(0) = qwind_f;
}

void dirichlet(const Vector &x, double t, Vector &u)
{
   u = 0.0;
}

}  // namespace linelast_frame_wind


namespace advdiff_problem
{

bool analytic_flow;
FlowProblem *flow_problem;

namespace advdiff_flow_past_array
{

double q0, dq, qoffset;
Vector qk;

double qbdr(const Vector &x, double t)
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
   double rx;
   double ry;
   double dx;
   double dy;
   double ux;
   double uy;
   double bx;
   double by;

   // Constant force
   double l_ux;
   double l_uy;
   double r_fx;
   double r_fy;
   double u_fx;
   double u_fy;
   double d_fx;
   double d_fy;
   double b_fx;
   double b_fy;

   // Amplitudes
   double xu_amp;
   double xf_amp;
   double yu_amp;
   double yf_amp;
   double bxf_amp;
   double byf_amp;

   // Frequencies
   double xu_freq;
   double xf_freq;
   double yu_freq;
   double yf_freq;
   double bxf_freq;
   double byf_freq;

   // Sine offsets
   double xu_offset;
   double xf_offset;
   double yu_offset;
   double yf_offset;
   double bxf_offset;
   double byf_offset;

double perturb_func(const double x, const double amp, const double freq, const double offset)
{
   return amp * sin(pi * freq *( x / 3.0 + 2 * offset) );
}

void left_disp(const Vector &x, double t, Vector &u)
{
   if (lx >= 0.5)
      u(0) = l_ux + perturb_func(x(0), xu_amp, xu_freq, xu_offset);
   if (ly >= 0.5)
      u(1) = l_uy + perturb_func(x(1), yu_amp, yu_freq, yu_offset);
}

void right_disp(const Vector &x, double t, Vector &u)
{
   if (rx >= 0.5)
      u(0) = r_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (ry >= 0.5)
      u(1) = r_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
}

void up_disp(const Vector &x, double t, Vector &u)
{
   if (ux >= 0.5)
      u(0) = u_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (uy >= 0.5)
      u(1) = u_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
}

void down_disp(const Vector &x, double t, Vector &u)
{
   if (dx >= 0.5)
      u(0) = d_fx + perturb_func(x(0), xf_amp, xf_freq, xf_offset);
   if (dy >= 0.5)
      u(1) = d_fy + perturb_func(x(1), yf_amp, yf_freq, yf_offset);
}

void body_force(const Vector &x, double t, Vector &u)
{
   if (bx >= 0.5)
      u(0) = b_fx + perturb_func(x(0), bxf_amp, bxf_freq, bxf_offset);
   if (by >= 0.5)
      u(1) = b_fy + perturb_func(x(1), byf_amp, byf_freq, byf_offset);
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
   else if (problem_name == "channel_flow")
   {
      problem = new ChannelFlow();
   }
   else if (problem_name == "component_flow")
   {
      problem = new ComponentFlow();
   }
   else if (problem_name == "flow_past_array")
   {
      problem = new FlowPastArray();
   }
   else if (problem_name == "backward_facing_step")
   {
      problem = new BackwardFacingStep();
   }
   else if (problem_name == "lid_driven_cavity")
   {
      problem = new LidDrivenCavity();
   }
   else if (problem_name == "force_driven_corner")
   {
      problem = new ForceDrivenCorner();
   }
   else if (problem_name == "periodic_flow_past_array")
   {
      problem = new PeriodicFlowPastArray();
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
   ChannelFlow
*/

ChannelFlow::ChannelFlow()
   : FlowProblem()
{
   battr.SetSize(5);
   for (int b = 0; b < 5; b++)
      battr[b] = b+1;
   bdr_type.SetSize(5);
   bdr_type = BoundaryType::ZERO;
   bdr_type[1] = BoundaryType::NEUMANN;
   bdr_type[3] = BoundaryType::DIRICHLET;

   // pointer to static function.
   vector_bdr_ptr.SetSize(5);
   vector_rhs_ptr = NULL;
   vector_bdr_ptr = &(function_factory::flow_problem::channel_flow::ubdr);

   param_num = 4;

   // Default values.
   function_factory::flow_problem::nu = 1.0;
   function_factory::flow_problem::channel_flow::L = 1.0;
   function_factory::flow_problem::channel_flow::U = 1.0;
   function_factory::flow_problem::channel_flow::x0 = 0.5;

   param_map["nu"] = 0;
   param_map["L"] = 1;
   param_map["U"] = 2;
   param_map["x0"] = 3;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::flow_problem::nu);
   param_ptr[1] = &(function_factory::flow_problem::channel_flow::L);
   param_ptr[2] = &(function_factory::flow_problem::channel_flow::U);
   param_ptr[3] = &(function_factory::flow_problem::channel_flow::x0);
}

ComponentFlow::ComponentFlow()
   : FlowProblem()
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
   vector_bdr_ptr = &(function_factory::flow_problem::component_flow::ubdr);

   param_num = 1 + 3 * 3 + 3 * 3;
   function_factory::flow_problem::component_flow::u0.SetSize(3);
   function_factory::flow_problem::component_flow::du.SetSize(3);
   function_factory::flow_problem::component_flow::offsets.SetSize(3);
   function_factory::flow_problem::component_flow::k.SetSize(3);

   // Default values.
   function_factory::flow_problem::nu = 1.0;
   function_factory::flow_problem::component_flow::u0 = 0.0;
   function_factory::flow_problem::component_flow::du = 1.0;
   function_factory::flow_problem::component_flow::offsets = 0.0;
   function_factory::flow_problem::component_flow::k = 1.0;

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
   param_ptr[0] = &(function_factory::flow_problem::nu);
   for (int i = 0; i < 3; i++)
   {
      param_ptr[1+i] = &(function_factory::flow_problem::component_flow::u0[i]);
      param_ptr[4+i] = &(function_factory::flow_problem::component_flow::du[i]);
      param_ptr[7+i] = &(function_factory::flow_problem::component_flow::offsets[i]);
      for (int j = 0; j < 3; j++)
         param_ptr[10 + 3*i + j] = &(function_factory::flow_problem::component_flow::k(j,i));
   }
}

/*
   FlowPastArray
*/

FlowPastArray::FlowPastArray()
   : ComponentFlow(), u0(&function_factory::flow_problem::component_flow::u0)
{}

void FlowPastArray::SetParams(const std::string &key, const double &value)
{
   ParameterizedProblem::SetParams(key, value);
   SetBattr();
}

void FlowPastArray::SetParams(const Array<int> &indexes, const Vector &values)
{
   ParameterizedProblem::SetParams(indexes, values);
   SetBattr();
}

void FlowPastArray::SetBattr()
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
   BackwardFacingStep
*/

BackwardFacingStep::BackwardFacingStep()
   : FlowProblem()
{
   /* Assume there are only three boundary attributes: 1, 2, 3 */
   battr.SetSize(3);
   for (int b = 0; b < 3; b++)
      battr[b] = b+1;
   
   /*
      1: dirichlet inflow
      2: dirichlet no-slip wall
      3: neumann outflow
   */
   bdr_type.SetSize(3);
   bdr_type[0] = BoundaryType::DIRICHLET;
   bdr_type[1] = BoundaryType::ZERO;
   bdr_type[2] = BoundaryType::NEUMANN;

   // pointer to static function.
   vector_bdr_ptr.SetSize(3);
   vector_rhs_ptr = NULL;
   /* technically only vector_bdr_ptr[0] will be used. */
   vector_bdr_ptr = &(function_factory::flow_problem::backward_facing_step::ubdr);

   /* pointer to initial condition */
   ic_ptr.SetSize(2);
   ic_ptr[0] = &(function_factory::flow_problem::backward_facing_step::uic);
   ic_ptr[1] = &(function_factory::flow_problem::backward_facing_step::pic);

   param_num = 4 + 2 * 4;

   // Default values.
   function_factory::flow_problem::nu = 1.0;
   function_factory::flow_problem::backward_facing_step::y0 = 0.0;
   function_factory::flow_problem::backward_facing_step::y1 = 1.0;
   function_factory::flow_problem::backward_facing_step::u0 = 1.0;

   function_factory::flow_problem::backward_facing_step::amp.SetSize(2);
   function_factory::flow_problem::backward_facing_step::ky.SetSize(2);
   function_factory::flow_problem::backward_facing_step::freq.SetSize(2);
   function_factory::flow_problem::backward_facing_step::t_offset.SetSize(2);
   function_factory::flow_problem::backward_facing_step::amp = 0.0;
   function_factory::flow_problem::backward_facing_step::ky = 0.0;
   function_factory::flow_problem::backward_facing_step::freq = 0.0;
   function_factory::flow_problem::backward_facing_step::t_offset = 0.0;

   param_map["nu"] = 0;
   param_map["y0"] = 1;
   param_map["y1"] = 2;
   param_map["u0"] = 3;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::flow_problem::nu);
   param_ptr[1] = &(function_factory::flow_problem::backward_facing_step::y0);
   param_ptr[2] = &(function_factory::flow_problem::backward_facing_step::y1);
   param_ptr[3] = &(function_factory::flow_problem::backward_facing_step::u0);

   for (int m = 0; m < 2; m++)
   {
      param_map["amp" + std::to_string(m)] = 4 + m * 4;
      param_map["ky" + std::to_string(m)] = 5 + m * 4;
      param_map["freq" + std::to_string(m)] = 6 + m * 4;
      param_map["t_offset" + std::to_string(m)] = 7 + m * 4;
      param_ptr[4 + m * 4] = &(function_factory::flow_problem::backward_facing_step::amp[m]);
      param_ptr[5 + m * 4] = &(function_factory::flow_problem::backward_facing_step::ky[m]);
      param_ptr[6 + m * 4] = &(function_factory::flow_problem::backward_facing_step::freq[m]);
      param_ptr[7 + m * 4] = &(function_factory::flow_problem::backward_facing_step::t_offset[m]);
   }
}

/*
   LidDrivenCavity
*/

LidDrivenCavity::LidDrivenCavity()
   : FlowProblem()
{
   /* Assume there are only five boundary attributes: 1, 2, 3, 4, 5 */
   battr.SetSize(5);
   for (int b = 0; b < 5; b++)
      battr[b] = b+1;
   
   /*
      All boundaries are zero except boundary attribute 3
   */
   bdr_type.SetSize(5);
   bdr_type = BoundaryType::ZERO;
   bdr_type[2] = BoundaryType::DIRICHLET;
   bdr_type[0] = BoundaryType::DIRICHLET;

   // pointer to static function.
   vector_bdr_ptr.SetSize(5);
   vector_rhs_ptr = NULL;
   /* technically only vector_bdr_ptr[2] will be used. */
   vector_bdr_ptr = &(function_factory::flow_problem::lid_driven_cavity::ubdr);

   param_num = 3;

   // Default values.
   function_factory::flow_problem::nu = 1.0;
   function_factory::flow_problem::lid_driven_cavity::u0 = 1.0;
   function_factory::flow_problem::lid_driven_cavity::L = 1.0;

   param_map["nu"] = 0;
   param_map["u0"] = 1;
   param_map["L"] = 2;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::flow_problem::nu);
   param_ptr[1] = &(function_factory::flow_problem::lid_driven_cavity::u0);
   param_ptr[2] = &(function_factory::flow_problem::lid_driven_cavity::L);
}

/*
   PeriodicFlowPastArray
*/

PeriodicFlowPastArray::PeriodicFlowPastArray()
   : FlowProblem()
{
   battr.SetSize(1);
   battr = 5;
   bdr_type.SetSize(1);
   bdr_type = BoundaryType::ZERO;

   // pointer to static function.
   vector_bdr_ptr.SetSize(1);
   vector_bdr_ptr = NULL;
   vector_rhs_ptr = &(function_factory::flow_problem::periodic_flow_past_array::force);

   param_num = 4;
   function_factory::flow_problem::periodic_flow_past_array::f.SetSize(3);

   // Default values.
   function_factory::flow_problem::nu = 1.0;
   function_factory::flow_problem::periodic_flow_past_array::f = 0.0;

   param_map["nu"] = 0;
   param_map["fx"] = 1;
   param_map["fy"] = 2;
   param_map["fz"] = 3;

   param_ptr.SetSize(param_num);
   param_ptr[0] = &(function_factory::flow_problem::nu);
   for (int i = 0; i < 3; i++)
      param_ptr[1+i] = &(function_factory::flow_problem::periodic_flow_past_array::f[i]);

}

/*
   ForceDrivenCorner
*/

ForceDrivenCorner::ForceDrivenCorner()
   : PeriodicFlowPastArray()
{
   battr.SetSize(5);
   for (int b = 0; b < 5; b++)
      battr[b] = b+1;
   bdr_type.SetSize(5);
   bdr_type = BoundaryType::ZERO;
   bdr_type[2] = BoundaryType::NEUMANN;
   bdr_type[3] = BoundaryType::NEUMANN;

   // pointer to static function.
   vector_bdr_ptr.SetSize(5);
   vector_bdr_ptr = NULL;
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
   battr[4] = 3;
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

/*
   AdvDiffFlowPastArray
*/

AdvDiffFlowPastArray::AdvDiffFlowPastArray()
   : FlowPastArray(), flow_problem(new FlowPastArray)
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
   function_factory::linelast_cwtrain::rx = 0.0;
   function_factory::linelast_cwtrain::ry = 0.0;
   function_factory::linelast_cwtrain::dx = 0.0;
   function_factory::linelast_cwtrain::dy = 0.0;
   function_factory::linelast_cwtrain::ux = 0.0;
   function_factory::linelast_cwtrain::uy = 0.0;
   function_factory::linelast_cwtrain::bx = 0.0;
   function_factory::linelast_cwtrain::by = 0.0;

   // Constant force default values
   function_factory::linelast_cwtrain::l_ux= 0.0;
   function_factory::linelast_cwtrain::l_uy = 0.0;
   function_factory::linelast_cwtrain::r_fx = 0.0;
   function_factory::linelast_cwtrain::r_fy = 0.0;
   function_factory::linelast_cwtrain::u_fx = 0.0;
   function_factory::linelast_cwtrain::u_fy = 0.0;
   function_factory::linelast_cwtrain::d_fx = 0.0;
   function_factory::linelast_cwtrain::d_fy = 0.0;
   function_factory::linelast_cwtrain::b_fx = 0.0;
   function_factory::linelast_cwtrain::b_fy = 0.0;

   // Amplitudes default values
   function_factory::linelast_cwtrain::xu_amp = 1.0;
   function_factory::linelast_cwtrain::yu_amp = 1.0;
   function_factory::linelast_cwtrain::xf_amp = 1.0;
   function_factory::linelast_cwtrain::yf_amp = 1.0;
   function_factory::linelast_cwtrain::bxf_amp = 1.0;
   function_factory::linelast_cwtrain::byf_amp = 1.0;

   // Frequencies default values
   function_factory::linelast_cwtrain::xu_freq = 1.0;
   function_factory::linelast_cwtrain::yu_freq = 1.0;
   function_factory::linelast_cwtrain::xf_freq = 1.0;
   function_factory::linelast_cwtrain::yf_freq = 1.0;
   function_factory::linelast_cwtrain::bxf_freq = 1.0;
   function_factory::linelast_cwtrain::byf_freq = 1.0;

   // Sine offsets default values
   function_factory::linelast_cwtrain::yu_offset = 0.0;
   function_factory::linelast_cwtrain::xu_offset = 0.0;
   function_factory::linelast_cwtrain::xf_offset = 0.0;
   function_factory::linelast_cwtrain::yf_offset = 0.0;
   function_factory::linelast_cwtrain::bxf_offset = 0.0;
   function_factory::linelast_cwtrain::byf_offset = 0.0;
   
   // Material parameters default values
   function_factory::linelast_problem::_lambda = 1.0;
   function_factory::linelast_problem::_mu = 1.0;

   // Parameter map
   param_map["l_ux"] = 0;
   param_map["l_uy"] = 1;
   param_map["r_fx"] = 2;
   param_map["r_fy"] = 3;
   param_map["u_fx"] = 4;
   param_map["u_fy"] = 5;
   param_map["d_fx"] = 6;
   param_map["d_fy"] = 7;
   param_map["lx"] = 8;
   param_map["ly"] = 9;
   param_map["rx"] = 10;
   param_map["ry"] = 11;
   param_map["dx"] = 12;
   param_map["dy"] = 13;
   param_map["ux"] = 14;
   param_map["uy"] = 15;
   param_map["lambda"] = 16;
   param_map["mu"] = 17;
   param_map["xu_amp"] = 18;
   param_map["xu_freq"] = 19;
   param_map["xu_offset"] = 20;
   param_map["xf_amp"] = 21;
   param_map["xf_freq"] = 22;
   param_map["xf_offset"] = 23;
   param_map["yu_amp"] = 24;
   param_map["yu_freq"] = 25;
   param_map["yu_offset"] = 26;
   param_map["yf_amp"] = 27;
   param_map["yf_freq"] = 28;
   param_map["yf_offset"] = 29;
   param_map["b_fx"] = 30;
   param_map["b_fy"] = 31;
   param_map["bxf_amp"] = 32;
   param_map["bxf_freq"] = 33;
   param_map["bxf_offset"] = 34;
   param_map["byf_amp"] = 35;
   param_map["byf_freq"] = 36;
   param_map["byf_offset"] = 37;
   param_map["bx"] = 38;
   param_map["by"] = 39;

   param_ptr.SetSize(40);
   param_ptr[0] = &(function_factory::linelast_cwtrain::l_ux);
   param_ptr[1] = &(function_factory::linelast_cwtrain::l_uy);
   param_ptr[2] = &(function_factory::linelast_cwtrain::r_fx);
   param_ptr[3] = &(function_factory::linelast_cwtrain::r_fy);
   param_ptr[4] = &(function_factory::linelast_cwtrain::u_fx);
   param_ptr[5] = &(function_factory::linelast_cwtrain::u_fy);
   param_ptr[6] = &(function_factory::linelast_cwtrain::d_fx);
   param_ptr[7] = &(function_factory::linelast_cwtrain::d_fy);
   param_ptr[8] = &(function_factory::linelast_cwtrain::lx);
   param_ptr[9] = &(function_factory::linelast_cwtrain::ly);
   param_ptr[10] = &(function_factory::linelast_cwtrain::rx);
   param_ptr[11] = &(function_factory::linelast_cwtrain::ry);
   param_ptr[12] = &(function_factory::linelast_cwtrain::dx);
   param_ptr[13] = &(function_factory::linelast_cwtrain::dy);
   param_ptr[14] = &(function_factory::linelast_cwtrain::ux);
   param_ptr[15] = &(function_factory::linelast_cwtrain::uy);
   param_ptr[16] = &(function_factory::linelast_problem::_lambda);
   param_ptr[17] = &(function_factory::linelast_problem::_mu);
   param_ptr[18] = &(function_factory::linelast_cwtrain::xu_amp);
   param_ptr[19] = &(function_factory::linelast_cwtrain::xu_freq);
   param_ptr[20] = &(function_factory::linelast_cwtrain::xu_offset);
   param_ptr[21] = &(function_factory::linelast_cwtrain::xf_amp);
   param_ptr[22] = &(function_factory::linelast_cwtrain::xf_freq);
   param_ptr[23] = &(function_factory::linelast_cwtrain::xf_offset);
   param_ptr[24] = &(function_factory::linelast_cwtrain::yu_amp);
   param_ptr[25] = &(function_factory::linelast_cwtrain::yu_freq);
   param_ptr[26] = &(function_factory::linelast_cwtrain::yu_offset);
   param_ptr[27] = &(function_factory::linelast_cwtrain::yf_amp);
   param_ptr[28] = &(function_factory::linelast_cwtrain::yf_freq);
   param_ptr[29] = &(function_factory::linelast_cwtrain::yf_offset);

   param_ptr[30] = &(function_factory::linelast_cwtrain::b_fx);
   param_ptr[31] = &(function_factory::linelast_cwtrain::b_fy);
   param_ptr[32] = &(function_factory::linelast_cwtrain::bxf_amp);
   param_ptr[33] = &(function_factory::linelast_cwtrain::bxf_freq);
   param_ptr[34] = &(function_factory::linelast_cwtrain::bxf_offset);
   param_ptr[35] = &(function_factory::linelast_cwtrain::byf_amp);
   param_ptr[36] = &(function_factory::linelast_cwtrain::byf_freq);
   param_ptr[37] = &(function_factory::linelast_cwtrain::byf_offset);
   param_ptr[38] = &(function_factory::linelast_cwtrain::bx);
   param_ptr[39] = &(function_factory::linelast_cwtrain::by);

   general_vector_ptr.SetSize(1);
   general_vector_ptr[0] = &(function_factory::linelast_cwtrain::body_force); 
}