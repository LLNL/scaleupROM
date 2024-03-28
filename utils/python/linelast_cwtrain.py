# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def shift_mesh_battrs(num_attr, path):
    # Read original file
    f = open(path, "r")
    all_lines = [x for x in f]
    f.close()
    f = open(path, "r")
    bdr_start = next(x for x, val in enumerate(f)
                                if val == "boundary\n")
    bdr_len = int(f.readline(bdr_start+1))
    bdr_txt = [f.readline(bdr_start+2 + i) for i in range(bdr_len)]
    battrs = [int(txt.split(' ')[0]) for txt in bdr_txt]
    first_lines = all_lines[0:bdr_start+2]
    last_lines = all_lines[bdr_start+2+bdr_len:]
    f.close()
    prefix = path.split('.')[0]
    for i in range(num_attr):
        filename = prefix + "_"+ str(i) + ".mesh"
        shift_battrs = [(b + i ) if (b + i) == num_attr else (b + i )%num_attr for b in battrs]
        shift_battrs_txt = [' '.join([str(shift_battrs[i]), *txt.split(' ')[1:]]) for i, txt in enumerate(bdr_txt)]
        f = open(filename, "w")
        new_txt = ''.join([''.join(first_lines), ''.join(shift_battrs_txt), ''.join(last_lines)])
        f.write(new_txt)
        f.close()

def create_training_meshes(prefix):
    shift_mesh_battrs(4, os.path.join(prefix, "joint2D.mesh"))
    shift_mesh_battrs(4, os.path.join(prefix, "rod2D_H.mesh"))
    shift_mesh_battrs(4, os.path.join(prefix, "rod2D_V.mesh"))

def cw_generate_nu_mxn(nx, ny, wh,wv,lh,lv):
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
        mesh_configs[i,:] = [xi*(lh+wv), yi*(lv+wh), 0., 0., 0., 0.]

        # Boundary check
        # global_battr / mesh_idx / comp_battr
        if xi == 0.0:
            bdr_data += [[1, i, 4]] # constrain one end
        elif xi == nx:
            bdr_data += [[2, i, 2]] # constrain one end
        if yi==0.0:
            bdr_data += [[4, i, 1]] # constrained ends
        elif yi==ny:
            bdr_data += [[5, i, 3]] # constrained ends

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
        mesh_configs[n_joint + i,:] = [xi*(lh+wv) + wv, yi*(lv+wh), 0., 0., 0., 0.]
        # Boundary check
        if yi==0.0:
            bdr_data += [[3, n_joint + i, 3]] # free boundary
            bdr_data += [[4, n_joint + i, 1]] # fixed boundary

        elif yi==ny:
            bdr_data += [[5, n_joint + i, 3]] # fixed boundary
            bdr_data += [[3, n_joint + i, 1]] # free boundary
        else:
            bdr_data += [[3, n_joint + i, 1]] # free boundary
            bdr_data += [[3, n_joint + i, 3]] # free boundary
    
    # Setup vertical rods
    for i in range(n_rod_V):
        xi = (i + nx+1) % (nx+1)
        yi = np.floor(i / (nx+1))
        mesh_configs[n_joint + n_rod_H + i,:] = [xi*(lh+wv), yi*(lv+wh) + wh, 0., 0., 0., 0.]

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

    # component config list J-H H-J J-V V-J
    comp_configs = []
    comp_configs += [[wv, 0., 0., 0., 0., 0.]]
    comp_configs += [[-lh, 0., 0., 0., 0., 0.]]
    comp_configs += [[0., wh, 0., 0., 0., 0.]]
    comp_configs += [[0., -lv, 0., 0., 0., 0.]]

    # interface data
    if_data = np.array(if_data)

    # boundary attributes
    bdr_data0 = np.array(bdr_data)

    return n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs
def cw_generate_mxn(nx, ny, w, l):
    cw_generate_nu_mxn(nx, ny, w,w,l,l)

def cw_generate_cwfom_mxn(nx, ny, wh,wv,lh,lv):
    n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs = cw_generate_nu_mxn(nx, ny, wh,wv,lh,lv)
    

    return n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs

def cw_generate_nu_1x1(wh,wv,lh,lv):
    n_mesh = 8
    mesh_type = [0, 1, 0, 2, 2, 0, 1, 0]
    mesh_configs = np.zeros([n_mesh, 6])
    mesh_configs[1,:] = [wv, 0., 0., 0., 0., 0.]
    mesh_configs[2,:] = [wv+lh, 0., 0., 0., 0., 0.]
    mesh_configs[3,:] = [0., wh, 0., 0., 0., 0.]
    mesh_configs[4,:] = [wv+lh, wh, 0., 0., 0., 0.]
    mesh_configs[5,:] = [0., wh+lv, 0., 0., 0., 0.]
    mesh_configs[6,:] = [wv, wh+lv, 0., 0., 0., 0.]
    mesh_configs[7,:] = [wv+lh, wh+lv, 0., 0., 0., 0.]

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

    # boundary attributes
    bdr_data0 = np.array(bdr_data)

    # component config list J-H H-J J-V V-J
    comp_configs = []
    comp_configs += [[wv, 0., 0., 0., 0., 0.]]
    comp_configs += [[-lh, 0., 0., 0., 0., 0.]]
    comp_configs += [[0., wh, 0., 0., 0., 0.]]
    comp_configs += [[0., -lv, 0., 0., 0., 0.]]

    return n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs

def cw_generate_1x1(w,l):
    return cw_generate_nu_1x1(w,w,l,l)

def Configure2DLatticeComponent(f, comp_list, comp_configs, n_mesh, mesh_type, mesh_configs, if_data, bdr_data):
    # comp_list is a list of component names
    # comp_configs is a list of component configuration
    grp = f.create_group("components")
    grp.attrs["number_of_components"] = len(comp_list)
    grp.attrs["0"] = comp_list[0] # Joint component name
    grp.attrs["1"] = comp_list[1] # Horizontal rod component name
    grp.attrs["2"] = comp_list[2] # Vertical rod component name
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
    port.attrs["comp1"] = comp_list[0] # Joint component name
    port.attrs["comp2"] = comp_list[1] # Horizontal rod component name
    port.attrs["attr1"] = 2
    port.attrs["attr2"] = 4
    port.create_dataset("comp2_configuration", (6,), data=comp_configs[0])

    port = grp.create_group("port2")
    port.attrs["comp1"] = comp_list[0] # Joint component name
    port.attrs["comp2"] = comp_list[1] # Horizontal rod component name
    port.attrs["attr1"] = 4
    port.attrs["attr2"] = 2
    port.create_dataset("comp2_configuration", (6,), data=comp_configs[1])

    port = grp.create_group("port3")
    port.attrs["comp1"] = comp_list[0] # Joint component name
    port.attrs["comp2"] = comp_list[2] # Vertical rod component name
    port.attrs["attr1"] = 3
    port.attrs["attr2"] = 1
    port.create_dataset("comp2_configuration", (6,), data=comp_configs[2])

    port = grp.create_group("port4")
    port.attrs["comp1"] = comp_list[0] # Joint component name
    port.attrs["comp2"] = comp_list[2] # Vertical rod component name
    port.attrs["attr1"] = 1
    port.attrs["attr2"] = 3
    port.create_dataset("comp2_configuration", (6,), data=comp_configs[3])

    # boundary attributes
    f.create_dataset("boundary", bdr_data.shape, data=bdr_data)

battr_map = np.array(
[
    [1, 2 ,3, 4],
    [4, 1, 2 ,3],
    [3, 4, 1, 2],
    [2, 3, 4, 1]
]
)

def PermuteBdrData(bdr_data0, j):
    bdr_data = bdr_data0.copy()
    for i in range(bdr_data.shape[0]):
        battr = bdr_data[i, 0]
        if battr < 5:
            bdr_data[i, 0] = battr_map[j, battr-1]
    return bdr_data

def CWTrain1x1():
    w = 1.0
    l = 4.0
    n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs = cw_generate_1x1(w,l)

    # list of component names J H V
    comp_list = ["joint2D", "rod2D_H", "rod2D_V"]

    for i in range(4): # Loop over all the variants
        bdr_data = PermuteBdrData(bdr_data0, i)
        filename = "linelast.comp_train" + str(i) + ".h5"
        with h5py.File(filename, 'w') as f:
            # c++ currently cannot read datasets of string.
            # change to multiple attributes, only as a temporary implementation.
            Configure2DLatticeComponent(f, comp_list, comp_configs, n_mesh, mesh_type, mesh_configs, if_data, bdr_data)
    return

def CWTrain2x2():
    nx = 2
    ny = 2
    l = 4.0
    w = 1.0

    n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs = cw_generate_mxn(nx, ny, w,l)

    # list of component names J H V
    comp_list = ["joint2D", "rod2D_H", "rod2D_V"]

    for i in range(4): # Loop over all the variants
        bdr_data = PermuteBdrData(bdr_data0, i)
        filename = "linelast.comp_train" + str(i) + ".h5"
        with h5py.File(filename, 'w') as f:
            # c++ currently cannot read datasets of string.
            # change to multiple attributes, only as a temporary implementation.
            Configure2DLatticeComponent(f, comp_list, comp_configs, n_mesh, mesh_type, mesh_configs, if_data, bdr_data)
    return


def CWTrainOptROM():
    wh = 0.2
    wv = 0.2
    lh = 6.0
    lv = 4.0

    n_mesh, mesh_type, mesh_configs, if_data, bdr_data0, comp_configs = cw_generate_nu_1x1(wh,wv,lh,lv)

    # list of component names J H V
    comp_list = ["optjoint", "optbeam", "optcol"]

    for i in range(4): # Loop over all the variants
        bdr_data = PermuteBdrData(bdr_data0, i)
        filename = "linelast.comp_train" + str(i) + ".h5"
        with h5py.File(filename, 'w') as f:
            # c++ currently cannot read datasets of string.
            # change to multiple attributes, only as a temporary implementation.
            Configure2DLatticeComponent(f, comp_list, comp_configs, n_mesh, mesh_type, mesh_configs, if_data, bdr_data)
    return

def CWTrainOptFOM():
    nx = 3
    ny = 3

    wh = 0.2
    wv = 0.2
    lh = 6.0
    lv = 4.0

    n_mesh, mesh_type, mesh_configs, if_data, bdr_data, comp_configs = cw_generate_cwfom_mxn(nx, ny, wh,wv,lh,lv)

    # list of component names J H V
    comp_list = ["optjoint", "optbeam", "optcol"]

    filename = "linelast.optfom.h5"
    with h5py.File(filename, 'w') as f:
        # c++ currently cannot read datasets of string.
        # change to multiple attributes, only as a temporary implementation.
        Configure2DLatticeComponent(f, comp_list, comp_configs, n_mesh, mesh_type, mesh_configs, if_data, bdr_data)
    return

def get_file_data(filename):
    with h5py.File(filename, 'r') as f:
        fom_t = f['fom_solve'][0]
        rom_t = f['rom_solve'][0]
        speedup = fom_t / rom_t
        relerr = f['rel_error'][0]
        return fom_t, rom_t, relerr, speedup

def get_results(samples, prefix):
    res=[]

    for name in samples:
        filename = os.path.join(prefix, 'comparison' + str(name) + '.h5')
        res.append(get_file_data(filename))

    return np.array(res)

def create_scaling_plot(samples, res, scale_prefix, plt_name = "plot.png"):
    fig, axs = plt.subplots(1,2, figsize=(15,5))

    axs[0].plot(samples, res[:,0], label='FOM solve time')
    axs[0].plot(samples, res[:,1], label='ROM solve time')
    axs[0].plot(samples, res[:,2], label='Relative error')
    axs[0].set_xlabel(scale_prefix)
    axs[0].set_yscale('log')
    axs[0].legend()


    axs[1].plot(samples, res[:,3], label='Speedup factor', )
    axs[1].set_xlabel(scale_prefix)
    axs[1].legend()

    fig.suptitle(scale_prefix+'-scaling', fontsize = 24)
    plt.tight_layout()
    plt.savefig(plt_name)

def basis_scaling_plot(prefix, plot_path):
    scale_prefix = '$n_{basis}$'
    samples = (2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128)
    res = get_results(samples, prefix)
    create_scaling_plot(samples, res, scale_prefix, plot_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Please specify a component to generate")
    else:
        name = sys.argv[1]
        if name == "cwtrain_1x1":
            CWTrain1x1()
        elif name == "cwtrain_2x2":
            CWTrain2x2()
        elif name == "cwtrain_optROM":
            CWTrainOptROM()
        elif name == "cwtrain_optFOM":
            CWTrainOptFOM()
        elif name == "bscale":
            prefix = sys.argv[2]
            plot_path = sys.argv[3]
            basis_scaling_plot(prefix, plot_path)
        elif name == "cwtrain_mesh":
            prefix = sys.argv[2]
            create_training_meshes(prefix)
        else:
            print("Option not found")
