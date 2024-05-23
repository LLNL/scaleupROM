# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from linelast_comp_base import *

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

battr_map = np.array(
[
    [1, 2 ,3, 4],
    [4, 1, 2 ,3],
    [3, 4, 1, 2],
    [2, 3, 4, 1]
])

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

    n_mesh, mesh_type, mesh_configs, if_data, bdr_data, comp_configs = cw_generate_nu_mxn(nx, ny, wh,wv,lh,lv)

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

def solve_time_scaling_plot(samples, res, scale_prefix, plt_name = "scaling.png"):
    plt.plot(samples, res[:,0], label='FOM')
    plt.plot(samples, res[:,1], label='ROM')
    plt.xlabel(scale_prefix)
    plt.yscale('log')
    plt.ylabel("Solve time [s]")
    plt.legend()

    plt.tight_layout()
    plt.savefig("solve_time_" + plt_name, dpi=300)
    plt.clf()

def relerr_scaling_plot(samples, res, scale_prefix, plt_name = "scaling.png"):
    plt.plot(samples, res[:,2], label='Relative error')
    plt.xlabel(scale_prefix)
    plt.yscale('log')
    plt.ylabel("Relative error [-]")
    
    plt.tight_layout()
    plt.savefig("relerr_" + plt_name, dpi=300)
    plt.clf()

def speedup_scaling_plot(samples, res, scale_prefix, plt_name = "scaling.png"):
    plt.plot(samples, res[:,3])
    plt.xlabel(scale_prefix)
    plt.ylabel("Speedup factor [-]")

    plt.tight_layout()
    plt.savefig("speedup_" + plt_name, dpi=300)
    plt.clf()

def create_scaling_plot(samples, res, scale_prefix, plt_name = "plot.png"):
    plt.rc('axes', labelsize=14)
    solve_time_scaling_plot(samples, res, scale_prefix, plt_name)
    relerr_scaling_plot(samples, res, scale_prefix, plt_name)
    speedup_scaling_plot(samples, res, scale_prefix, plt_name)

def get_nr(txt, split_txt = 'comparison'):
    return int(txt.split('.')[0].split(split_txt)[1])
def get_nrs(txts, split_txt = 'comparison'):
    return [get_nr(txt, split_txt) for txt in txts]
def get_sorted_nrs(txts, split_txt = 'comparison'):
    return sorted(zip(get_nrs(txts, split_txt),txts))

def basis_scaling_plot(prefix, plot_path, result_path = "basis_scaling"):
    cwd = os.getcwd()
    abs_scaling_folder = os.path.join(cwd,result_path)
    os.chdir(abs_scaling_folder)
    txts = os.listdir()
    samples = [i for i,_ in get_sorted_nrs(txts, split_txt = 'comparison')]
    os.chdir(cwd)
    res = get_results(samples, prefix)
    scale_prefix = '$n_{basis}$'
    create_scaling_plot(samples, res, scale_prefix, plot_path)

def get_svs(filename):
        with open(filename, 'r') as file:
            return [float(line.strip()) for line in file]

def sv_plot(jointname, h_name, v_name, legend_names, plt_name = "sv_plot.png"):
    svs_j = get_svs(jointname)
    svs_b = get_svs(h_name)
    svs_c = get_svs(v_name)

    plt.rc('axes', labelsize=18)
    plt.plot(range(len(svs_j)), svs_j, label=legend_names[0])
    plt.plot(range(len(svs_b)), svs_b, label=legend_names[1])
    plt.plot(range(len(svs_c)), svs_c, label=legend_names[2])
    plt.xscale('log')
    plt.xlabel('$n$')
    plt.ylabel('$\sigma$')
    plt.yscale('log')
    plt.legend(title="Component name")
    #plt.title('Advanced component SV spectrum')
    plt.tight_layout()
    #plt.savefig(os.path.join(os.getcwd(), plt_name))
    plt.savefig(os.path.join(os.getcwd(), plt_name), dpi=300)

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
        elif name == "svplot":
            prefix = sys.argv[2]
            jointname = sys.argv[3]
            h_name = sys.argv[4]
            v_name = sys.argv[5]
            legend_type = sys.argv[6]

            if legend_type == 'A':
                legend_names = ["Joint", "Beam", "Column"]
            else:
                legend_names = ["Joint", "Beam, X", "Beam, Y"]

            jointname_file = prefix + "_" + jointname + "_sv.txt"
            h_name_file = prefix + "_" + h_name + "_sv.txt"
            v_name_file = prefix + "_" + v_name + "_sv.txt"

            sv_plot(jointname_file, h_name_file, v_name_file, legend_names)        
        elif name == "cwtrain_mesh":
            prefix = sys.argv[2]
            create_training_meshes(prefix)
        else:
            print("Option not found")
