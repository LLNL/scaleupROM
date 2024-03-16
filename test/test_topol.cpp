// Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include<gtest/gtest.h>
#include "topology_handler.hpp"
#include "component_topology_handler.hpp"
#include <fstream>
#include <iostream>

using namespace std;

void CompareWithSubMesh();

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
   SUCCEED();
}

TEST(Consistency2D_test, Test_topol)
{
   config = InputParser("inputs/test_topol.2d.yml");

   CompareWithSubMesh();
   return;
}

TEST(Consistency3D_hex_test, Test_topol)
{
   config = InputParser("inputs/test_topol.3d.yml");

   CompareWithSubMesh();
   return;
}

TEST(Consistency3D_tet_test, Test_topol)
{
   config = InputParser("inputs/test_topol.3d.yml");
   config.dict_["mesh"]["filename"] = "meshes/test.2x2x2.tet.mesh";
   config.dict_["mesh"]["component-wise"]["components"][0]["file"] = "meshes/test.1x1x1.tet.mesh";

   // For tetrahedra, indexing order becomes different, and it's difficult to match it.
   // Simply print out the result for now.
   // dd_mms verifies convergence rate with tetrahedra mesh.
   printf("Submesh\n");
   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler();
   submesh->PrintPortInfo();
   submesh->PrintInterfaceInfo();
   printf("\n");

   printf("Component\n");
   ComponentTopologyHandler *comp = new ComponentTopologyHandler();
   comp->PrintPortInfo();
   comp->PrintInterfaceInfo();
   printf("\n");
   return;
}

TEST(PortReadWrite_test, Test_topol)
{
   config = InputParser("inputs/test_topol.3d.yml");
   config.dict_["mesh"]["component-wise"]["write_ports"] = true;
   config.dict_["mesh"]["component-wise"]["ports"][0]["file"] = "port1.3d.h5";
   config.dict_["mesh"]["component-wise"]["ports"][1]["file"] = "port2.3d.h5";
   config.dict_["mesh"]["component-wise"]["ports"][2]["file"] = "port3.3d.h5";

   printf("Generate\n");
   ComponentTopologyHandler *old = new ComponentTopologyHandler();
   old->PrintPortInfo();
   old->PrintInterfaceInfo();
   printf("\n");

   printf("Read\n");
   ComponentTopologyHandler *cnew = new ComponentTopologyHandler();
   cnew->PrintPortInfo();
   cnew->PrintInterfaceInfo();
   printf("\n");

   EXPECT_EQ(old->GetNumRefPorts(), cnew->GetNumRefPorts());
   for (int p = 0; p < old->GetNumRefPorts(); p++)
   {
      ComponentTopologyHandler::PortData *o_port = old->GetPortData(p);
      ComponentTopologyHandler::PortData *n_port = cnew->GetPortData(p);

      EXPECT_EQ(o_port->Component1, n_port->Component1);
      EXPECT_EQ(o_port->Component2, n_port->Component2);
      EXPECT_EQ(o_port->Attr1, n_port->Attr1);
      EXPECT_EQ(o_port->Attr2, n_port->Attr2);

      EXPECT_EQ(o_port->be_pairs.NumRows(), n_port->be_pairs.NumRows());
      EXPECT_EQ(o_port->be_pairs.NumCols(), n_port->be_pairs.NumCols());
      for (int i = 0; i < o_port->be_pairs.NumRows(); i++)
         for (int j = 0; j < o_port->be_pairs.NumCols(); j++)
         {
            EXPECT_EQ(o_port->be_pairs(i,j), n_port->be_pairs(i,j));
         }

      EXPECT_EQ(o_port->vtx2to1.size(), n_port->vtx2to1.size());
      for (auto& it : o_port->vtx2to1)
      {
         EXPECT_TRUE(n_port->vtx2to1.count(it.first));
         EXPECT_TRUE(n_port->vtx2to1[it.first] == it.second);
      }
   }
   return;
}

int main(int argc, char* argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

void CompareWithSubMesh()
{
   printf("Submesh\n");
   SubMeshTopologyHandler *submesh = new SubMeshTopologyHandler();
   submesh->PrintPortInfo();
   submesh->PrintInterfaceInfo();
   printf("\n");

   printf("Component\n");
   ComponentTopologyHandler *comp = new ComponentTopologyHandler();
   comp->PrintPortInfo();
   comp->PrintInterfaceInfo();
   printf("\n");

   EXPECT_EQ(submesh->GetNumPorts(), comp->GetNumPorts());
   int num_ports = submesh->GetNumPorts();
   for (int p = 0; p < num_ports; p++)
   {
      const PortInfo *s_port, *c_port;
      s_port = submesh->GetPortInfo(p);
      c_port = comp->GetPortInfo(p);

      // The input file for component is manually calibrated to match the order.
      // In general, the order of these two objects do not necessarily match.
      EXPECT_EQ(s_port->PortAttr, c_port->PortAttr);
      EXPECT_EQ(s_port->Mesh1, c_port->Mesh1);
      EXPECT_EQ(s_port->Mesh2, c_port->Mesh2);

      Array<InterfaceInfo> const *s_info, *c_info;
      s_info = submesh->GetInterfaceInfos(p);
      c_info = comp->GetInterfaceInfos(p);

      // (The order of) BEs of submesh and component do not match in general.
      // Only checking the Infs...
      EXPECT_EQ(s_info->Size(), c_info->Size());
      for (int f = 0; f < s_info->Size(); f++)
      {
         EXPECT_EQ((*s_info)[f].Inf1, (*c_info)[f].Inf1);
         EXPECT_EQ((*s_info)[f].Inf2, (*c_info)[f].Inf2);
      }
   }
   return;
}