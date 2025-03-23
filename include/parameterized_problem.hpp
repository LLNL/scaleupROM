// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef PARAMETERIZED_PROBLEM_HPP
#define PARAMETERIZED_PROBLEM_HPP

#include "mfem.hpp"
#include "input_parser.hpp"

using namespace mfem;

namespace function_factory
{

/*
   These take the spatial location and time.
   Time-constant functions won't use the time input in calculation.
   For time-varying systems, corresponding FunctionCoefficient
   must execuate SetTime to reflect the time.
*/
typedef double GeneralScalarFunction(const Vector &, double);
typedef void GeneralVectorFunction(const Vector &, double, Vector &);

static const double pi = 4.0 * atan(1.0);

namespace poisson0
{
   extern double k;
   extern double offset;

   double rhs(const Vector &x, double t);
}

namespace poisson_component
{
   extern Vector k, bdr_k;
   extern double offset, bdr_offset;
   extern double bdr_idx;

   double bdr(const Vector &x, double t);
   double rhs(const Vector &x, double t);
}

namespace poisson_spiral
{
   static const int N = 2;
   extern double L, Lw, k, s;
   double rhs(const Vector &x, double t);
}

namespace flow_problem
{

extern double nu;
extern double del_u;  // complementary flux to ensure incompressibility.
extern Vector x0;

void dir(const Vector &x, Vector &y);
void flux(const Vector &x, Vector &y);

namespace channel_flow
{
   extern double L, U, x0;
   void ubdr(const Vector &x, double t, Vector &y);
}

namespace component_flow
{
   extern Vector u0, du, offsets;
   extern DenseMatrix k;
   void ubdr(const Vector &x, double t, Vector &y);
}

namespace backward_facing_step
{
   extern double u0, y0, y1;
   extern Vector amp, ky, freq, t_offset;
   
   void ubdr(const Vector &x, double t, Vector &y);
   void uic(const Vector &x, double t, Vector &y);
   void pic(const Vector &x, double t, Vector &y);
}

namespace lid_driven_cavity
{
   extern double u0, L;
   
   void ubdr(const Vector &x, double t, Vector &y);
}

namespace periodic_flow_past_array
{
   extern Vector f;
   void force(const Vector &x, double t, Vector &y);
}

}

namespace linelast_problem
{
extern double _lambda;
extern double _mu;

double lambda(const Vector &x, double t);
double mu(const Vector &x, double t);

}

namespace linelast_disp
{
extern double rdisp_f;

void init_disp(const Vector &x, double t, Vector &u);
}

namespace linelast_force
{
extern double rforce_f;

void tip_force(const Vector &x, double t, Vector &u);
}


namespace linelast_octet
{

extern double disp_z;
extern double force_z;
extern double density;
extern double g;

void fixed_bc(const Vector &x, Vector &u);

void disp_z_bc(const Vector &x, Vector &u);

void force_z_bc(const Vector &x, Vector &f);

void fixed_yz_bc(const Vector &x, Vector &u);

void gravity_load(const Vector &x, Vector &f);

}  // namespace linelast_octet

namespace linelast_cwtrain
{
   // Probabilities
   extern double lx;
   extern double ly;
   extern double lz;
   extern double rx;
   extern double ry;
   extern double rz;
   extern double dx;
   extern double dy;
   extern double dz;
   extern double ux;
   extern double uy;
   extern double uz;
   extern double bx;
   extern double by;
   extern double bz;

   // Constant force
   extern double l_ux;
   extern double l_uy;
   extern double l_uz;
   extern double r_fx;
   extern double r_fy;
   extern double r_fz;
   extern double u_fx;
   extern double u_fy;
   extern double u_fz;
   extern double d_fx;
   extern double d_fy;
   extern double d_fz;
   extern double b_fx;
   extern double b_fy;
   extern double b_fz;

// Amplitudes
extern double xu_amp;
extern double yu_amp;
extern double zu_amp;
extern double xf_amp;
extern double yf_amp;
extern double zf_amp;
extern double bxf_amp;
extern double byf_amp;
extern double bzf_amp;

// Frequencies
extern double xu_freq;
extern double yu_freq;
extern double zu_freq;
extern double xf_freq;
extern double yf_freq;
extern double zf_freq;
extern double bxf_freq;
extern double byf_freq;
extern double bzf_freq;

// Sine offsets
extern double xu_offset;
extern double yu_offset;
extern double zu_offset;
extern double xf_offset;
extern double yf_offset;
extern double zf_offset;
extern double bxf_offset;
extern double byf_offset;
extern double bzf_offset;

double perturb_func(const double x, const double amp, const double freq, const double offset);
void left_disp(const Vector &x, double t, Vector &u);
void up_disp(const Vector &x, double t, Vector &u);
void down_disp(const Vector &x, double t, Vector &u);
void right_disp(const Vector &x, double t, Vector &u);
void in_disp(const Vector &x, double t, Vector &u);
void out_disp(const Vector &x, double t, Vector &u);
void body_force(const Vector &x, double t, Vector &u);
}

namespace linelast_frame_wind
{
extern double qwind_f;
extern double density;
extern double g;

void wind_load(const Vector &x, double t, Vector &f);

void gravity_load(const Vector &x, double t, Vector &f);

void dirichlet(const Vector &x, double t, Vector &u);

}

namespace linelast_lattice_roof
{
extern double qsnow_f;
extern double qpoint_f;
extern double density;
extern double g;

void snow_load(const Vector &x, Vector &f);

void gravity_load(const Vector &x, Vector &f);

void point_load(const Vector &x, Vector &f);

void dirichlet(const Vector &x, Vector &u);

}

namespace advdiff_problem
{

extern bool analytic_flow;

namespace advdiff_flow_past_array
{
   extern double q0, dq, qoffset;
   extern Vector qk;

   double qbdr(const Vector &x, double t);
}  //  namespace advdiff_flow_past_array

}  // namespace advdiff_problem

}

enum class BoundaryType
{ 
   ZERO,
   DIRICHLET,
   NEUMANN,
   NUM_BDR_TYPE
};

class ParameterizedProblem
{
friend class SampleGenerator;
friend class RandomSampleGenerator;

protected:
   std::string problem_name;

   std::size_t param_num;
   std::map<std::string, int> param_map;

   Array<double *> param_ptr; // address of parameters in function_factory.

public:
   ParameterizedProblem();

   virtual ~ParameterizedProblem() {};

   const std::string GetProblemName() { return problem_name; }
   const int GetNumParams() { return param_num; }
   const int GetParamIndex(const std::string &name)
   {
      if (!(param_map.count(name))) printf("%s\n", name.c_str());
      assert(param_map.count(name));
      return param_map[name];
   }

   // virtual member functions cannot be passed down as argument.
   // Instead use pointers to static functions.
   function_factory::GeneralScalarFunction *scalar_rhs_ptr = NULL;
   Array<function_factory::GeneralScalarFunction *> scalar_bdr_ptr;
   function_factory::GeneralVectorFunction *vector_rhs_ptr = NULL;
   Array<function_factory::GeneralVectorFunction *> vector_bdr_ptr;
   Array<int> battr;
   Array<BoundaryType> bdr_type; // abstract boundary type

   /* initial condition for time-dependent problem */
   /* size with number of variables */
   Array<function_factory::GeneralVectorFunction *> ic_ptr;

   Array<function_factory::GeneralScalarFunction *> general_scalar_ptr;
   Array<function_factory::GeneralVectorFunction *> general_vector_ptr;

   // TODO: use variadic function? what would be the best format?
   // TODO: support other datatypes such as integer?
   virtual void SetParams(const std::string &key, const double &value);
   virtual void SetParams(const Array<int> &indexes, const Vector &values);

   void SetSingleRun();
};

class PoissonProblem : public ParameterizedProblem
{
friend class PoissonSolver;

public:
   virtual ~PoissonProblem() {};
};

class Poisson0 : public PoissonProblem
{
public:
   Poisson0();
   virtual ~Poisson0() {};
};

class PoissonComponent : public PoissonProblem
{
public:
   PoissonComponent();
   virtual ~PoissonComponent() {};
   void SetParams(const std::string &key, const double &value) override;
   void SetParams(const Array<int> &indexes, const Vector &values) override;

private:
   void SetBattr();
};

class PoissonSpiral : public PoissonProblem
{
public:
   PoissonSpiral();
   virtual ~PoissonSpiral() {};
};

class FlowProblem : public ParameterizedProblem
{
friend class StokesSolver;

public:
   virtual ~FlowProblem() {};
};

class ChannelFlow : public FlowProblem
{
public:
   ChannelFlow();
   virtual ~ChannelFlow() {};
};

class ComponentFlow : public FlowProblem
{
public:
   ComponentFlow();
};

class FlowPastArray : public ComponentFlow
{
friend class AdvDiffFlowPastArray;

public:
   FlowPastArray();

   virtual void SetParams(const std::string &key, const double &value) override;
   virtual void SetParams(const Array<int> &indexes, const Vector &values) override;

protected:
   Vector *u0;
   virtual void SetBattr();
};

class BackwardFacingStep : public FlowProblem
{
public:
   BackwardFacingStep();
   virtual ~BackwardFacingStep() {};
};

class LidDrivenCavity : public FlowProblem
{
public:
   LidDrivenCavity();
   virtual ~LidDrivenCavity() {};
};

class PeriodicFlowPastArray : public FlowProblem
{
public:
   PeriodicFlowPastArray();
};

class ForceDrivenCorner : public PeriodicFlowPastArray
{
public:
   ForceDrivenCorner();
   virtual ~ForceDrivenCorner() {};
};

class LinElastProblem : public ParameterizedProblem
{
friend class LinElastSolver;

protected:
   Vector lambda, mu;

public:
   virtual ~LinElastProblem() {};
};
class LinElastDisp : public LinElastProblem
{
public:
   LinElastDisp();
};

class LinElastDispLCantilever : public LinElastProblem
{
public:
   LinElastDispLCantilever();
};

class LinElastDispLattice : public LinElastProblem
{
public:
   LinElastDispLattice();
};

class LinElastForceCantilever : public LinElastProblem
{
public:
   LinElastForceCantilever();
};

class LinElastFrameWind : public LinElastProblem
{
public:
   LinElastFrameWind();
};

class LinElastLatticeRoof : public LinElastProblem
{
public:
   LinElastLatticeRoof();
};

namespace function_factory
{

namespace advdiff_problem
{
/*
   flow_problem will be passed down to StokesSolver/SteadyNSSolver for obtaining velocity field.
   It must be set appropriately within each AdvDiffSolver problems.
*/
extern FlowProblem *flow_problem;

}  // namespace advdiff_problem

}  // namespace function_factory

class AdvDiffFlowPastArray : public FlowPastArray
{
protected:
   /*
      flow_problem shares the same pointers with this class.
      Thus every parameter set by this class is reflected to FlowPastArrayProblem as well.
      flow_problem will be passed down to StokesSolver/SteadyNSSolver for obtaining velocity field.
   */
   FlowPastArray *flow_problem = NULL;

public:
   AdvDiffFlowPastArray();
   virtual ~AdvDiffFlowPastArray();

protected:
   void SetBattr() override
   {
      FlowPastArray::SetBattr();
      flow_problem->SetBattr();
   }
};

class LinElastComponentWiseTrain : public LinElastProblem
{
public:
   LinElastComponentWiseTrain();
};

class LinElastComponentWiseTrain3D : public LinElastProblem
{
public:
LinElastComponentWiseTrain3D();
};

class LinElastOctetCube : public LinElastProblem
{
public:
LinElastOctetCube();
};

class LinElastOctetBeam : public LinElastProblem
{
public:
LinElastOctetBeam();
};

class LinElastOctetTop : public LinElastProblem
{
public:
LinElastOctetTop();
};

ParameterizedProblem* InitParameterizedProblem();

#endif
