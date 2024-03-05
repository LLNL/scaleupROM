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

def grid_mesh_configs(nx, ny, l, w):
    bdr_data = []
    if_data = []
    n_joint = (nx + 1) * (ny + 1)
    n_rod_H = (ny + 1) * nx
    n_rod_V = (nx + 1) * ny
    n_rod = n_rod_H + n_rod_V
    n_mesh = n_joint + n_rod
    mesh_configs = np.zeros([n_mesh, 6])
    mesh_type = np.concatenate((np.full((int(n_joint),), 0.0),np.full((int(n_rod_H),), 1.0),np.full((int(n_rod_V),), 2.0)))
    # Setup joints
    for i in range(n_joint):
        xi = (i + nx+1) % (nx+1)
        yi = np.floor(i / (nx+1))
        mesh_configs[i,:] = [xi*(l+w), yi*(l+w), 0., 0., 0., 0.]

        # Boundary check
        # global_battr / mesh_idx / comp_battr
        if xi == 0.0:
            bdr_data += [[1, i, 4]] # constrain one end
        elif xi == nx:
            bdr_data += [[2, i, 2]] # constrain one end
        if yi==0.0:
            bdr_data += [[3, i, 1]] # free boundary
        elif yi==ny:
            bdr_data += [[3, i, 3]] # free boundary
        
        # Interface check
        # mesh1 / mesh2 / battr1 / battr2 / port_idx
        # Case port idx = 0
        if xi < nx:
            if_data += [[i, i + n_joint - yi, 2, 4, 0]]
        # Case port idx = 1
        if xi > 0:
            if_data += [[i, i + n_joint - yi - 1, 4, 2, 1]]
        # Case port idx = 2
        if yi < ny:
            if_data += [[i, n_joint + n_rod_H + i, 3, 1, 2]]
        # Case port idx = 3
        if yi > 0:
            if_data += [[i, n_joint + n_rod_H + i - (nx+1), 1, 3, 3]]

    # Setup horizontal rods
    for i in range(n_rod_H):
        xi = (i + nx) % (nx)
        yi = np.floor(i / (nx))
        mesh_configs[n_joint + i,:] = [xi*(l+w) + w, yi*(l+w), 0., 0., 0., 0.]
        # Boundary check
        bdr_data += [[3, n_joint + i, 1]] # free boundary
        bdr_data += [[3, n_joint + i, 3]] # free boundary
    
    # Setup vertical rods
    for i in range(n_rod_V):
        xi = (i + nx+1) % (nx+1)
        yi = np.floor(i / (nx+1))
        mesh_configs[n_joint + n_rod_H + i,:] = [xi*(l+w), yi*(l+w) + w, 0., 0., 0., 0.]

        # Boundary check
        if xi == 0.0:
            bdr_data += [[1, n_joint + n_rod_H + i, 4]] # constrain one end
            bdr_data += [[3, n_joint + n_rod_H + i, 2]] # free boundary
        elif xi == nx:
            bdr_data += [[2, n_joint + n_rod_H + i, 2]] # constrain one end
            bdr_data += [[3, n_joint + n_rod_H + i, 4]] # free boundary
        else:
            bdr_data += [[3, n_joint + n_rod_H + i, 2]] # constrain one end
            bdr_data += [[3, n_joint + n_rod_H + i, 4]] # free boundary

    return mesh_configs, bdr_data, if_data, mesh_type, n_mesh


def LatticeCantilever(nx, ny):
    # nx and ny are the number of sections in x- and y-direction, respectively
    l = 4.0
    w = 1.0
    mesh_configs, bdr_data, if_data, mesh_type, n_mesh = grid_mesh_configs(nx, ny, l, w)

    # interface data
    if_data = np.array(if_data)

    # boundary attributes
    bdr_data = np.array(bdr_data)

    filename = "linelast.lattice.h5"
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
        grp.attrs["number_of_references"] = 4
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.attrs["2"] = "port3"
        grp.attrs["3"] = "port4"
        grp.create_dataset("interface", if_data.shape, data=if_data)

        port = grp.create_group("port1")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 2
        port.attrs["attr2"] = 4
        port.create_dataset("comp2_configuration", (6,), data=[w, 0., 0., 0., 0., 0.])

        port = grp.create_group("port2")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 4
        port.attrs["attr2"] = 2
        port.create_dataset("comp2_configuration", (6,), data=[-l, 0., 0., 0., 0., 0.])

        port = grp.create_group("port3")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 3
        port.attrs["attr2"] = 1
        port.create_dataset("comp2_configuration", (6,), data=[0., w, 0., 0., 0., 0.])

        port = grp.create_group("port4")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 1
        port.attrs["attr2"] = 3
        port.create_dataset("comp2_configuration", (6,), data=[0., -l, 0., 0., 0., 0.])

        # boundary attributes
        f.create_dataset("boundary", bdr_data.shape, data=bdr_data)
    return

def ComponentWiseTrain():
    n_mesh = 5
    mesh_type = [0, 2, 1, 2, 1]
    mesh_configs = np.zeros([n_mesh, 6])
    mesh_configs[1,:] = [0., -4., 0., 0., 0., 0.]
    mesh_configs[2,:] = [1., 0., 0., 0., 0., 0.]
    mesh_configs[3,:] = [0., 1., 0., 0., 0., 0.]
    mesh_configs[4,:] = [-4., 0., 0., 0., 0., 0.]

    # interface data
    # mesh1 / mesh2 / battr1 / battr2 / port_idx
    if_data = np.zeros([4, 5])
    if_data[0, :] = [0, 1, 1, 3, 0]
    if_data[1, :] = [0, 2, 2, 4, 1]
    if_data[2, :] = [0, 3, 3, 1, 2]
    if_data[3, :] = [0, 4, 4, 2, 3]

    # boundary attributes
    # global_battr / mesh_idx / comp_battr
    bdr_data = []

    # applied loads and displacements
    bdr_data += [[1, 1, 1]]
    bdr_data += [[2, 2, 2]]
    bdr_data += [[3, 3, 3]]
    bdr_data += [[4, 4, 4]]

    # homogenous neumann
    bdr_data += [[5, 1, 2]]
    bdr_data += [[5, 1, 4]]
    bdr_data += [[5, 2, 1]]
    bdr_data += [[5, 2, 3]]
    bdr_data += [[5, 3, 2]]
    bdr_data += [[5, 3, 4]]
    bdr_data += [[5, 4, 1]]
    bdr_data += [[5, 4, 3]]

    bdr_data = np.array(bdr_data)
    print(bdr_data.shape)

    filename = "linelast.comp_train.h5"
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
        grp.attrs["number_of_references"] = 4
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.attrs["2"] = "port3"
        grp.attrs["3"] = "port4"
        grp.create_dataset("interface", if_data.shape, data=if_data)

        port = grp.create_group("port1")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 1
        port.attrs["attr2"] = 3
        port.create_dataset("comp2_configuration", (6,), data=[0., -4., 0., 0., 0., 0.])

        port = grp.create_group("port2")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 2
        port.attrs["attr2"] = 4
        port.create_dataset("comp2_configuration", (6,), data=[1., 0., 0., 0., 0., 0.])

        port = grp.create_group("port3")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 3
        port.attrs["attr2"] = 1
        port.create_dataset("comp2_configuration", (6,), data=[0., 1., 0., 0., 0., 0.])

        port = grp.create_group("port4")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 4
        port.attrs["attr2"] = 2
        port.create_dataset("comp2_configuration", (6,), data=[-4., 0., 0., 0., 0., 0.])

        # boundary attributes
        f.create_dataset("boundary", bdr_data.shape, data=bdr_data)
    return


def ComponentWiseTrain2():
    n_mesh = 8
    mesh_type = [0, 1, 0, 2, 2, 0, 1, 0]
    mesh_configs = np.zeros([n_mesh, 6])
    mesh_configs[1,:] = [1., 0., 0., 0., 0., 0.]
    mesh_configs[2,:] = [5., 0., 0., 0., 0., 0.]
    mesh_configs[3,:] = [0., 1., 0., 0., 0., 0.]
    mesh_configs[4,:] = [5., 1., 0., 0., 0., 0.]
    mesh_configs[5,:] = [0., 5., 0., 0., 0., 0.]
    mesh_configs[6,:] = [1., 5., 0., 0., 0., 0.]
    mesh_configs[7,:] = [5., 5., 0., 0., 0., 0.]

    # interface data
    # mesh1 / mesh2 / battr1 / battr2 / port_idx
    if_data = np.zeros([8, 5])
    if_data[0, :] = [0, 1, 2, 4, 0]
    if_data[1, :] = [2, 1, 4, 2, 1]
    if_data[2, :] = [0, 3, 3, 1, 2]
    if_data[3, :] = [2, 4, 3, 1, 2]
    if_data[4, :] = [5, 3, 1, 3, 3]
    if_data[5, :] = [7, 4, 1, 3, 3]
    if_data[6, :] = [5, 6, 2, 4, 0]
    if_data[7, :] = [7, 6, 4, 2, 1]

    # boundary attributes
    # global_battr / mesh_idx / comp_battr
    bdr_data = []

    # applied loads and displacements
    bdr_data += [[1, 0, 1]]
    bdr_data += [[1, 1, 1]]
    bdr_data += [[1, 2, 1]]
    bdr_data += [[2, 2, 2]]
    bdr_data += [[2, 4, 2]]
    bdr_data += [[2, 7, 2]]
    bdr_data += [[3, 5, 3]]
    bdr_data += [[3, 6, 3]]
    bdr_data += [[3, 7, 3]]
    bdr_data += [[4, 0, 4]]
    bdr_data += [[4, 3, 4]]
    bdr_data += [[4, 5, 4]]

    # homogenous neumann
    bdr_data += [[5, 1, 3]]
    bdr_data += [[5, 4, 4]]
    bdr_data += [[5, 6, 1]]
    bdr_data += [[5, 3, 2]]

    bdr_data = np.array(bdr_data)
    print(bdr_data.shape)

    filename = "linelast.comp_train.h5"
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
        grp.attrs["number_of_references"] = 4
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.attrs["2"] = "port3"
        grp.attrs["3"] = "port4"
        grp.create_dataset("interface", if_data.shape, data=if_data)

        port = grp.create_group("port1")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 2
        port.attrs["attr2"] = 4
        port.create_dataset("comp2_configuration", (6,), data=[1., 0., 0., 0., 0., 0.])

        port = grp.create_group("port2")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_H"
        port.attrs["attr1"] = 4
        port.attrs["attr2"] = 2
        port.create_dataset("comp2_configuration", (6,), data=[-4., 0., 0., 0., 0., 0.])

        port = grp.create_group("port3")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 3
        port.attrs["attr2"] = 1
        port.create_dataset("comp2_configuration", (6,), data=[0., 1., 0., 0., 0., 0.])

        port = grp.create_group("port4")
        port.attrs["comp1"] = "joint2D"
        port.attrs["comp2"] = "rod2D_V"
        port.attrs["attr1"] = 1
        port.attrs["attr2"] = 3
        port.create_dataset("comp2_configuration", (6,), data=[0., -4., 0., 0., 0., 0.])

        # boundary attributes
        f.create_dataset("boundary", bdr_data.shape, data=bdr_data)
    return

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        SimpleL()
    else:
        name = sys.argv[1]
        if name == "simple_l":
            SimpleL()
        elif name == "lattice_cantilever":
            nx = int(sys.argv[2])
            ny = int(sys.argv[3])
            LatticeCantilever(nx, ny)
        elif name == "componentwise_train":
            ComponentWiseTrain()
        elif name == "componentwise_train2":
            ComponentWiseTrain2()
