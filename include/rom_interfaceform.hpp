// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_ROM_INTERFACEFORM_HPP
#define SCALEUPROM_ROM_INTERFACEFORM_HPP

#include "interface_form.hpp"
#include "rom_handler.hpp"

namespace mfem
{

class ROMInterfaceForm : public InterfaceForm
{
protected:
   const int numPorts;

   /* size of numSub */
   Array<DenseMatrix *> basis;   // not owned
   Array<int> basis_dof_offsets;

   /* EQP sample point info */
   /*
      Array of size (fnfi.Size() * topol_handler->GetNumPorts()),
      where each element is another Array of EQP samples
      at the given port p and the given integrator i.
      For the port p and the integrator i,
         fnfi_sample[p + i * numPorts]
   */
   Array<Array<SampleInfo> *> fnfi_sample;

   /// @brief Flag for precomputing necessary coefficients for fast computation.
   bool precompute = false;

public:
   ROMInterfaceForm(Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_, TopologyHandler *topol_);

   virtual ~ROMInterfaceForm();

   void SetBasisAtSubdomain(const int m, DenseMatrix &basis_, const int offset=0);
   void UpdateBlockOffsets();

   /// Adds new Interior Face Integrator.
   void AddInterfaceIntegrator(InterfaceNonlinearFormIntegrator *nlfi) override
   {
      InterfaceForm::AddInterfaceIntegrator(nlfi);

      for (int p = 0; p < numPorts; p++)
         fnfi_sample.Append(NULL);
   }

   void UpdateInterFaceIntegratorSampling(const int i, const int p, const Array<int> &itf,
                                          const Array<int> &qp, const Array<double> &qw)
   {
      assert((i >= 0) && (i < fnfi.Size()));
      assert((p >= 0) && (p < numPorts));
      assert(fnfi_sample.Size() == fnfi.Size() * numPorts);

      const int idx = p + i * numPorts;
      if (fnfi_sample[idx])
         delete fnfi_sample[idx];

      fnfi_sample[idx] = new Array<SampleInfo>(0);
      for (int s = 0; s < itf.Size(); s++)
         fnfi_sample[idx]->Append({.el=-1, .face=-1, .be=-1, .itf=itf[s], .qp=qp[s], .qw=qw[s]});
   }

   void InterfaceAddMult(const Vector &x, Vector &y) const override;

   void InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const override;

private:
   /* These methods are not available in this class */
   using InterfaceForm::AssembleInterfaceMatrices;
   using InterfaceForm::AssembleInterfaceMatrixAtPort;
   using InterfaceForm::AssembleInterfaceMatrix;
   using InterfaceForm::AssembleInterfaceVector;
   using InterfaceForm::AssembleInterfaceGrad;

};

} // namespace mfem

#endif
