import numpy as np
import h5py

def ConfigScaleUp(ncomp):
    n_mesh = ncomp * ncomp
    mesh_type = [0] * n_mesh
    mesh_configs = np.zeros([n_mesh, 6])
    for j in range(ncomp):
        for i in range(ncomp):
            m_idx = i + j * ncomp
            mesh_configs[m_idx, 0] = i * 1.
            mesh_configs[m_idx, 1] = j * 1.

    if_data = np.zeros([2 * ncomp * (ncomp - 1), 5])
    for j in range(ncomp):
        for i in range(ncomp-1):
            p_idx = i + j * (ncomp-1)
            m1 = i + j * ncomp
            m2 = i+1 + j * ncomp
            if_data[p_idx, :] = [m1, m2, 2, 4, 1]
    offset = ncomp * (ncomp - 1)
    for j in range(ncomp-1):
        for i in range(ncomp):
            p_idx = i + j * ncomp + offset
            m1 = i + j * ncomp
            m2 = i + (j+1) * ncomp
            if_data[p_idx, :] = [m1, m2, 3, 1, 0]

    bdr_data = []
    for b in range(ncomp):
        bdr_data += [[1, b, 1]]
        bdr_data += [[2, (ncomp-1) + b * ncomp, 2]]
        bdr_data += [[3, b + (ncomp-1) * ncomp, 3]]
        bdr_data += [[4, b * ncomp, 4]]
    bdr_data = np.array(bdr_data)
    print(bdr_data.shape)

    filename = "scaleup.global.%d.h5" % ncomp
    with h5py.File(filename, 'w') as f:
        # c++ currently cannot read datasets of string.
        # change to multiple attributes, only as a temporary implementation.
        grp = f.create_group("components")
        grp.attrs["number_of_components"] = 1
        grp.attrs["0"] = "square"
        # component index of each mesh
        grp.create_dataset("meshes", (n_mesh,), data=mesh_type)
        # 3-dimension vector for translation / rotation
        grp.create_dataset("configuration", mesh_configs.shape, data=mesh_configs)

        grp = f.create_group("ports")
        grp.attrs["number_of_references"] = 2
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.create_dataset("interface", if_data.shape, data=if_data)

        # boundary attributes
        f.create_dataset("boundary", (4 * ncomp, 3), data=bdr_data)
    return

def Config2D():
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
        # component index of each mesh
        grp.create_dataset("meshes", (4,), data=[0,0,0,0])
        # 3-dimension vector for translation / rotation
        grp.create_dataset("configuration", (4,6), data=[[0.,0.,0.,0.,0.,0.],
                                                         [1.,0.,0.,0.,0.,0.],
                                                         [0.,1.,0.,0.,0.,0.],
                                                         [1.,1.,0.,0.,0.,0.]])

        grp = f.create_group("ports")
        grp.attrs["number_of_references"] = 2
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        # interface data
        # mesh1 / mesh2 / battr1 / battr2 / port_idx
        if_data = [[0, 1, 2, 4, 1],
                   [0, 2, 3, 1, 0],
                   [1, 3, 3, 1, 0],
                   [2, 3, 2, 4, 1]]
        grp.create_dataset("interface", (4,5), data=if_data)

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
    return

def Config3D():
    filename = "global.h5"
    with h5py.File(filename, 'w') as f:
        # c++ currently cannot read datasets of string.
        # change to multiple attributes, only as a temporary implementation.
        # f.create_dataset("components", (1,), data=["square"])
        # f.create_dataset("ports", (2,), data=["port1", "port2"])
        grp = f.create_group("components")
        grp.attrs["number_of_components"] = 1
        grp.attrs["0"] = "cube"
        # component index of each mesh
        grp.create_dataset("meshes", (8,), data=[0] * 8)
        # 3-dimension vector for translation / rotation
        grp.create_dataset("configuration", (8,6), data=[[0.,0.,0.,0.,0.,0.],
                                                         [.5,0.,0.,0.,0.,0.],
                                                         [0.,.5,0.,0.,0.,0.],
                                                         [.5,.5,0.,0.,0.,0.],
                                                         [0.,0.,.5,0.,0.,0.],
                                                         [.5,0.,.5,0.,0.,0.],
                                                         [0.,.5,.5,0.,0.,0.],
                                                         [.5,.5,.5,0.,0.,0.]])

        grp = f.create_group("ports")
        grp.attrs["number_of_references"] = 3
        grp.attrs["0"] = "port1"
        grp.attrs["1"] = "port2"
        grp.attrs["2"] = "port3"
        # interface data
        # mesh1 / mesh2 / battr1 / battr2 / port_idx
        if_data = [[0, 1, 3, 5, 0],
                   [0, 2, 4, 2, 1],
                   [0, 4, 6, 1, 2],
                   [1, 3, 4, 2, 1],
                   [1, 5, 6, 1, 2],
                   [2, 3, 3, 5, 0],
                   [2, 6, 6, 1, 2],
                   [3, 7, 6, 1, 2],
                   [4, 5, 3, 5, 0],
                   [4, 6, 4, 2, 1],
                   [5, 7, 4, 2, 1],
                   [6, 7, 3, 5, 0]]
        grp.create_dataset("interface", (12,5), data=if_data)

        # boundary attributes
        # global_battr / mesh_idx / comp_battr
        bdr_data = [[1, 0, 1],
                    [1, 1, 1],
                    [1, 2, 1],
                    [1, 3, 1],
                    [2, 0, 2],
                    [2, 1, 2],
                    [2, 4, 2],
                    [2, 5, 2],
                    [3, 1, 3],
                    [3, 3, 3],
                    [3, 5, 3],
                    [3, 7, 3],
                    [4, 2, 4],
                    [4, 3, 4],
                    [4, 6, 4],
                    [4, 7, 4],
                    [5, 0, 5],
                    [5, 2, 5],
                    [5, 4, 5],
                    [5, 6, 5],
                    [6, 4, 6],
                    [6, 5, 6],
                    [6, 6, 6],
                    [6, 7, 6]]
        f.create_dataset("boundary", (24,3), data=bdr_data)
    return

if __name__ == "__main__":
    ConfigScaleUp(128)
    #for ncomp in [4,8,16,32,64]:
    #    ConfigScaleUp(ncomp)
    # Config2D()
    #Config3D()
