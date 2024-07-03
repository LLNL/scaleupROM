// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#ifndef SCALEUPROM_ROM_ELEMENT_COLLECTION_HPP
#define SCALEUPROM_ROM_ELEMENT_COLLECTION_HPP

#include "topology_handler.hpp"
#include "rom_nonlinearform.hpp"
#include "rom_interfaceform.hpp"
#include "mfem.hpp"
#include "hdf5_utils.hpp"

// By convention we only use mfem namespace as default, not CAROM.
using namespace mfem;

class ROMElementCollection
{
protected:
   const int num_var;
   const int num_comp;
   const int num_ref_ports;
   const bool separate_variable;

   TopologyHandler *topol_handler = NULL; // not owned
   Array<FiniteElementSpace *> fes;       // not owned

public:
   ROMElementCollection(TopologyHandler *topol_handler_, const Array<FiniteElementSpace *> &fes_,
                        const bool separate_variable_)
      : topol_handler(topol_handler_), fes(fes_),
        num_comp(topol_handler_->GetNumComponents()),
        num_var(fes_.Size() / topol_handler_->GetNumComponents()),
        num_ref_ports(topol_handler_->GetNumRefPorts()),
        separate_variable(separate_variable_)
   { 
      assert(num_comp * num_var == fes.Size());
      assert(topol_handler->GetType() == TopologyHandlerMode::COMPONENT);
   }

   virtual ~ROMElementCollection() {}

   virtual void Save(const std::string &filename) = 0;
   virtual void Load(const std::string &filename) = 0;
};

class ROMLinearElement : public ROMElementCollection
{
public:
   Array<MatrixBlocks *> comp;     // Size(num_components);
   // boundary condition is enforced via forcing term.
   Array<Array<MatrixBlocks *> *> bdr;
   Array<MatrixBlocks *> port;   // reference ports.

public:
   ROMLinearElement(TopologyHandler *topol_handler_,
                    const Array<FiniteElementSpace *> &fes_,
                    const bool separate_variable_);

   virtual ~ROMLinearElement();

   void Save(const std::string &filename) override;
   void Load(const std::string &filename) override;

private:
   void SaveCompBdrElems(hid_t &file_id);
   void SaveBdrElems(hid_t &comp_grp_id, const int &comp_idx);
   void SaveItfaceElems(hid_t &file_id);

   void LoadCompBdrElems(hid_t &file_id);
   void LoadBdrElems(hid_t &comp_grp_id, const int &comp_idx);
   void LoadItfaceElems(hid_t &file_id);
};

class ROMTensorElement : public ROMElementCollection
{
public:
   Array<DenseTensor *> comp;     // Size(num_components);
   
   /* boundary/interface is not implemented yet.. should consider */
   // Array<Array<DenseTensor *> *> bdr;
   // Array<DenseTensor *> port;   // reference ports.

public:
   ROMTensorElement(TopologyHandler *topol_handler_,
                    const Array<FiniteElementSpace *> &fes_,
                    const bool separate_variable_)
      : ROMElementCollection(topol_handler_, fes_, separate_variable_) {}

   virtual ~ROMTensorElement();

   void Save(const std::string &filename) override {}
   void Load(const std::string &filename) override {}

};

class ROMEQPElement : public ROMElementCollection
{
public:
   Array<ROMNonlinearForm *> comp;     // Size(num_components);
   // boundary condition is enforced via forcing term.
   Array<Array<ROMNonlinearForm *> *> bdr;
   ROMInterfaceForm *port = NULL;   // reference ports.

public:
   ROMEQPElement(TopologyHandler *topol_handler_,
                 const Array<FiniteElementSpace *> &fes_,
                 const bool separate_variable_)
      : ROMElementCollection(topol_handler_, fes_, separate_variable_) {}

   virtual ~ROMEQPElement() {}

   void Save(const std::string &filename) override {}
   void Load(const std::string &filename) override {}

};

#endif
