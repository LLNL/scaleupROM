// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_ROM_INTERFACEFORM_HPP
#define SCALEUPROM_ROM_INTERFACEFORM_HPP

#include "interface_form.hpp"
#include "rom_handler.hpp"
#include "hdf5_utils.hpp"
#include "hyperreduction_integ.hpp"
#include "linalg/NNLS.h"

namespace mfem
{

class ROMInterfaceForm : public InterfaceForm
{
protected:
   const int numPorts;
   const int numRefPorts;
   const int num_comp;

   /* component finite element space */
   Array<FiniteElementSpace *> comp_fes;       // not owned

   /* size of num_comp */
   Array<DenseMatrix *> comp_basis;   // not owned
   Array<int> comp_basis_dof_offsets;

   /* size of numSub */
   /* view Arrays of component arrays */
   Array<DenseMatrix *> basis;   // not owned
   Array<int> basis_dof_offsets;

   /* EQP sample point info */
   /*
      View Array of fnfi_ref_sample.
      Array of size (fnfi.Size() * topol_handler->GetNumPorts()),
      where each element is an EQPElement
      at the given port p and the given integrator i.
      For the port p and the integrator i,
         fnfi_sample[p + i * numPorts]
   */
   Array<EQPElement *> fnfi_sample;

   /*
      Array of size (fnfi.Size() * topol_handler->GetNumRefPorts()),
      where each element is an EQPElement
      at the given reference port p and the given integrator i.
      For the reference port p and the integrator i,
         fnfi_sample[p + i * numRefPorts]
   */
   Array<EQPElement *> fnfi_ref_sample;

   /// @brief Flag for precomputing necessary coefficients for fast computation.
   bool precompute = false;

   /// @brief Energy norm criterion for NNLS.
   CAROM::NNLS_termination nnls_criterion = CAROM::NNLS_termination::L2;

public:
   ROMInterfaceForm(Array<Mesh *> &meshes_, Array<FiniteElementSpace *> &fes_,
                    Array<FiniteElementSpace *> &comp_fes_, TopologyHandler *topol_);

   virtual ~ROMInterfaceForm();

   const bool PrecomputeMode() { return precompute; }
   void SetPrecomputeMode(const bool precompute_) { precompute = precompute_; }

   void PrecomputeCoefficients();

   void SetBasisAtComponent(const int c, DenseMatrix &basis_, const int offset=0);
   void UpdateBlockOffsets();

   /// Adds new Interior Face Integrator.
   void AddInterfaceIntegrator(InterfaceNonlinearFormIntegrator *nlfi) override
   {
      InterfaceForm::AddInterfaceIntegrator(nlfi);

      for (int p = 0; p < numRefPorts; p++)
         fnfi_ref_sample.Append(NULL);
      for (int p = 0; p < numPorts; p++)
         fnfi_sample.Append(NULL);
   }

   void UpdateInterFaceIntegratorSampling(const int i, const int rp,
                                          const Array<SampleInfo> &samples)
   {
      assert((i >= 0) && (i < fnfi.Size()));
      assert((rp >= 0) && (rp < numRefPorts));
      assert(fnfi_ref_sample.Size() == fnfi.Size() * numRefPorts);

      const int idx = rp + i * numRefPorts;
      if (fnfi_ref_sample[idx])
         delete fnfi_ref_sample[idx];

      fnfi_ref_sample[idx] = new EQPElement(samples);

      for (int p = 0; p < numPorts; p++)
      {
         if (topol_handler->GetPortType(p) != rp)
            continue;
         
         fnfi_sample[p + i * numPorts] = fnfi_ref_sample[idx];
      }
   }

   void InterfaceAddMult(const Vector &x, Vector &y) const override;

   void InterfaceGetGradient(const Vector &x, Array2D<SparseMatrix *> &mats) const override;

   void TrainEQPForRefPort(const int p, const CAROM::Matrix &snapshot1, const CAROM::Matrix &snapshot2,
                           const Array2D<int> &snap_pair_idx, const double eqp_tol);

   void SetupEQPSystem(const CAROM::Matrix &snapshot1, const CAROM::Matrix &snapshot2,
                       const Array2D<int> &snap_pair_idx,
                       DenseMatrix &basis1, DenseMatrix &basis2,
                       const int &basis1_offset, const int &basis2_offset,
                       FiniteElementSpace *fes1, FiniteElementSpace *fes2,
                       Array<InterfaceInfo>* const itf_infos,
                       InterfaceNonlinearFormIntegrator* const nlfi,
                       CAROM::Matrix &Gt, CAROM::Vector &rhs_Gw);

   void TrainEQPForIntegrator(const int nqe, const CAROM::Matrix &Gt,
                              const CAROM::Vector &rhs_Gw, const double eqp_tol,
                              Array<SampleInfo> &samples);

   void SaveEQPForIntegrator(const int k, hid_t file_id, const std::string &dsetname);
   void LoadEQPForIntegrator(const int k, hid_t file_id, const std::string &dsetname);

private:
   /* These methods are not available in this class */
   using InterfaceForm::AssembleInterfaceMatrices;
   using InterfaceForm::AssembleInterfaceMatrixAtPort;
   using InterfaceForm::AssembleInterfaceMatrix;
   using InterfaceForm::AssembleInterfaceVector;
   using InterfaceForm::AssembleInterfaceGrad;

   void PrecomputeEQPSample(const IntegrationRule &ir,
                            FaceElementTransformations *tr1, FaceElementTransformations *tr2,
                            FiniteElementSpace *fes1, FiniteElementSpace *fes2,
                            const DenseMatrix &basis1, const DenseMatrix &basis2, EQPSample &eqp_sample);
};

} // namespace mfem

#endif
