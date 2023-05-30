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

#ifndef PARAMETERIZED_PROBLEM_HPP
#define PARAMETERIZED_PROBLEM_HPP

#include "mfem.hpp"
#include "input_parser.hpp"

using namespace mfem;

namespace function_factory
{

typedef double GeneralScalarFunction(const Vector &);
typedef void GeneralVectorFunction(const Vector &, Vector &);

static const double pi = 4.0 * atan(1.0);

namespace poisson0
{
   extern double k;
   extern double offset;

   // TODO: function template?
   double rhs(const Vector &x);
}

namespace poisson_component
{
   extern Vector k, bdr_k;
   extern double offset, bdr_offset;
   extern double bdr_idx;

   double bdr(const Vector &x);
   double rhs(const Vector &x);
}

namespace poisson_spiral
{
   static const int N = 2;
   extern double L, Lw, k;
   double rhs(const Vector &x);
}

namespace stokes_channel
{
   extern double L, U;
   void ubdr(const Vector &x, Vector &y);
}

}

class ParameterizedProblem
{
friend class SampleGenerator;
friend class RandomSampleGenerator;

protected:
   std::string problem_name;

   std::size_t param_num;
   std::map<std::string, int> param_map;

   Array<double *> param_ptr; // address of parameters in function_factory.

   // local sample index;
   int local_sample_index;

public:
   ParameterizedProblem();

   virtual ~ParameterizedProblem() {};

   const std::string GetProblemName() { return problem_name; }
   const int GetNumParams() { return param_num; }
   const int GetLocalSampleIndex() { return local_sample_index; }
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
   Array<int> bdr_type; // abstract boundary type

   // TODO: use variadic function? what would be the best format?
   // TODO: support other datatypes such as integer?
   virtual void SetParams(const std::string &key, const double &value);
   virtual void SetParams(const Array<int> &indexes, const Vector &values);
};

class PoissonProblem : public ParameterizedProblem
{
friend class PoissonSolver;

protected:
   enum BoundaryType
   { ZERO, DIRICHLET, NEUMANN, NUM_BDR_TYPE };

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
   virtual void SetParams(const std::string &key, const double &value);
   virtual void SetParams(const Array<int> &indexes, const Vector &values);

private:
   void SetBattr();
};

class PoissonSpiral : public PoissonProblem
{
public:
   PoissonSpiral();
   virtual ~PoissonSpiral() {};
};

class StokesProblem : public ParameterizedProblem
{
friend class StokesSolver;

protected:
   enum BoundaryType
   { ZERO, DIRICHLET, NEUMANN, NUM_BDR_TYPE };

public:
   virtual ~StokesProblem() {};
};

class StokesChannel : public StokesProblem
{
public:
   StokesChannel();
   virtual ~StokesChannel() {};
};

ParameterizedProblem* InitParameterizedProblem();

#endif
