import numpy as np
import h5py

if __name__ == "__main__":
    filename = "port1.h5"
    with h5py.File(filename, 'w') as f:
        f.attrs["number_of_ports"] = 1
        grp = f.create_group("0")
        grp.attrs["name"] = "port1"
        grp.attrs["component1"] = "square"
        grp.attrs["component2"] = "square"
        grp.attrs["bdr_attr1"] = 3
        grp.attrs["bdr_attr2"] = 1
        grp.create_dataset("vtx1", (2,), data=[3, 2])
        grp.create_dataset("vtx2", (2,), data=[1, 0])
        grp.create_dataset("be1", (1,), data=[1])
        grp.create_dataset("be2", (1,), data=[0])

    filename = "port2.h5"
    with h5py.File(filename, 'w') as f:
        f.attrs["number_of_ports"] = 1
        grp = f.create_group("0")
        grp.attrs["name"] = "port2"
        grp.attrs["component1"] = "square"
        grp.attrs["component2"] = "square"
        grp.attrs["bdr_attr1"] = 2
        grp.attrs["bdr_attr2"] = 4
        grp.create_dataset("vtx1", (2,), data=[1, 3])
        grp.create_dataset("vtx2", (2,), data=[0, 2])
        grp.create_dataset("be1", (1,), data=[3])
        grp.create_dataset("be2", (1,), data=[2])

    filename = "global.h5"
    with h5py.File(filename, 'w') as f:
        # c++ currently cannot read datasets of string.
        # change to multiple attributes, only as a temporary implementation.
        # f.create_dataset("components", (1,), data=["square"])
        # f.create_dataset("ports", (2,), data=["port1", "port2"])
        grp = f.create_group("components")
        grp.attrs["number_of_components"] = 1
        grp.attrs["0"] = "square"
        grp = f.create_group("ports")
        grp.attrs["number_of_ports"] = 2
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"

        # meshes
        grp = f.create_group("meshes")
        # component index of each mesh
        grp.create_dataset("component", (4,), data=[0,0,0,0])
        # 3-dimension vector for translation / rotation
        grp.create_dataset("configuration", (4,6), data=[[0.,0.,0.,0.,0.,0.],
                                                         [1.,0.,0.,0.,0.,0.],
                                                         [0.,1.,0.,0.,0.,0.],
                                                         [1.,1.,0.,0.,0.,0.]])

        # boundary attributes
        # global_battr / mesh_idx / comp_battr
        bdr_data = [[1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 2],
                    [2, 3, 2],
                    [3, 2, 3],
                    [3, 3, 3],
                    [4, 2, 4],
                    [4, 0, 4]]
        f.create_dataset("boundary", (8,3), data=bdr_data)

        # interface data
        # mesh1 / mesh2 / battr1 / battr2 / port_idx
        if_data = [[0, 1, 2, 4, 1],
                   [0, 2, 3, 1, 0],
                   [1, 3, 3, 1, 0],
                   [2, 3, 2, 4, 0]]
        f.create_dataset("interface", (4,5), data=if_data)
