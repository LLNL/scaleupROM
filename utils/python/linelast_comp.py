# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import h5py

def SimpleL():
    n_mesh = 3
    mesh_type = [0, 1, 2]
    mesh_configs = np.zeros([n_mesh, 6])
    mesh_configs[1,:] = [1., 0., 0., 0., 0., 0.]
    mesh_configs[2,:] = [0., 1., 0., 0., 0., 0.]

    # interface data
    # mesh1 / mesh2 / battr1 / battr2 / port_idx
    if_data = np.zeros([2, 5])
    if_data[0, :] = [0, 1, 2, 4, 0]
    if_data[1, :] = [0, 2, 3, 1, 1]

    # boundary attributes
    # global_battr / mesh_idx / comp_battr
    bdr_data = []

    # applied loads and displacements
    bdr_data += [[1, 1, 2]]
    bdr_data += [[2, 2, 3]]

    # homogenous neumann
    bdr_data += [[3, 0, 4]]
    bdr_data += [[3, 0, 1]]
    bdr_data += [[3, 1, 1]]
    bdr_data += [[3, 1, 3]]
    bdr_data += [[3, 2, 2]]
    bdr_data += [[3, 2, 4]]

    bdr_data = np.array(bdr_data)
    print(bdr_data.shape)

    filename = "linelast.simpleL.h5"
    with h5py.File(filename, 'w') as f:
        # c++ currently cannot read datasets of string.
        # change to multiple attributes, only as a temporary implementation.
        grp = f.create_group("components")
        grp.attrs["number_of_components"] = 3
        grp.attrs["0"] = "joint2D"
        grp.attrs["1"] = "rod2D_H"
        grp.attrs["2"] = "rod2D_V"
        # component index of each mesh
        grp.create_dataset("meshes", (n_mesh,), data=mesh_type)
        # 3-dimension vector for translation / rotation
        grp.create_dataset("configuration", mesh_configs.shape, data=mesh_configs)

        grp = f.create_group("ports")
        grp.attrs["number_of_references"] = 2
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.create_dataset("interface", if_data.shape, data=if_data)

        port = grp.create_group("port1")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 2
        port.attrs["attr2"] = 4
        port.create_dataset("comp2_configuration", (6,), data=[1., 0., 0., 0., 0., 0.])

        port = grp.create_group("port2")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 3
        port.attrs["attr2"] = 1
        port.create_dataset("comp2_configuration", (6,), data=[0., 1., 0., 0., 0., 0.])

        # boundary attributes
        f.create_dataset("boundary", bdr_data.shape, data=bdr_data)
    return

if __name__ == "__main__":
    SimpleL()
