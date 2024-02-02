# Copyright 2023 Lawrence Livermore National Security, LLC. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
from copy import deepcopy
from config import Empty, ObjectInSpace
from box_channel_config import BoxChannelConfig

if __name__ == "__main__":
    comp_list = {'empty': Empty(),
                 'circle': ObjectInSpace('square-circle'),
                 'square': ObjectInSpace('square-square'),
                 'triangle': ObjectInSpace('square-triangle'),
                 'star': ObjectInSpace('square-star'),}
    
    example = BoxChannelConfig(1,5)
    for name, comp in comp_list.items():
        example.addComponent(comp)
    
    for icomp in example.comps:
        for iloc, iface in icomp.face_map.items():
            jloc = np.array(iloc)
            jloc *= -1
            jloc = tuple(jloc)
            for jcomp in example.comps:
                jface = -1
                if (jloc in jcomp.face_map):
                    jface = jcomp.face_map[jloc]
                else:
                    continue
                print(icomp.name, jcomp.name, iface, jface)
                if (jface < 0):
                    continue
                else:
                    print('accepted')
                    example.appendRefPort(icomp.name, jcomp.name, iface, jface)

    for c in range(len(example.comps)):
        example.addMesh(c, 0)

    example.close()
    example.save('box-channel.comp.h5')
