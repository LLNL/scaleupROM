import numpy as np
from copy import deepcopy
import h5py

class Component:
    name = ''

    num_face = -1
    dim = 2
    
    ref_battr = []
    face_map = {}       # {location of counterpart component: matching ref_battr}
    face_map_inv = {}   # {matching ref_battr: location of counterpart component}
    avail_face = {}     # same as face_map_inv

    def __init__(self):
        return

class Empty(Component):
    name = 'empty'

    def __init__(self):
        self.num_face = 4
        self.ref_battr = [1, 2, 3, 4]
        self.face_map = {(0, -1):  1,
                         (1, 0):   2,
                         (0, 1):   3,
                         (-1, 0):  4}
        for key, val in self.face_map.items():
            self.face_map_inv[val] = key
        self.avail_face = deepcopy(self.face_map_inv)
        return
    
class PipeHub(Component):
    name = 'pipe-hub'

    def __init__(self):
        self.num_face = 5
        self.ref_battr = [1, 2, 3, 4, 5]
        self.face_map = {(0, -1):  1,
                         (1, 0):   2,
                         (0, 1):   3,
                         (-1, 0):  4}
        for key, val in self.face_map.items():
            self.face_map_inv[val] = key
        self.avail_face = deepcopy(self.face_map_inv)
        return
    
class ObjectInSpace(PipeHub):
    name = 'object-in-space'

    def __init__(self, name='square-square'):
        PipeHub.__init__(self)
        self.name = name
        return
    
class Configuration:
    prefix = 'config'
    dim = 2

    # reference component
    comps = []
    # reference port
    ports = {}

    # meshes in the global configuration. These are in fact pointers to reference component.
    meshes = []
    mesh_types = []
    loc = np.array([], dtype=int)
    bdr_data = np.array([], dtype=int)
    if_data = np.array([], dtype=int)

    def __init__(self):
        return
    
    def addComponent(self, component):
        assert(self.dim == component.dim)
        self.comps += [component]
        return
    
    def close(self):
        raise RuntimeError('Abstract method Configuration.close()!')
        return
    
    def isLocationAvailable(self, new_loc):
        raise RuntimeError('Abstract method Configuration.isLocationAvailable!')
        return
    
    def getMeshConfigs(self):
        mesh_configs = np.zeros([len(self.meshes), 6])
        mesh_configs[:,:self.dim] = self.loc
        return mesh_configs
    
    def addMesh(self, comp_idx, new_loc):
        assert(self.isLocationAvailable(new_loc))

        # add new mesh and its type.
        # NOTE: this if-statement is necessary in order to keep proper memory address
        #       for GenerateAllConfigs!!
        if (len(self.meshes) == 0):
            self.meshes = [deepcopy(self.comps[comp_idx])]
        else:    
            self.meshes += [deepcopy(self.comps[comp_idx])]
        self.mesh_types += [comp_idx]

        # add the new mesh's location.
        if (self.loc.ndim == 1):
            self.loc = np.array([new_loc])
        else:
            self.loc = np.append(self.loc, [new_loc], axis=0)        
        return
    
    def appendBoundary(self, gbattr, m_idx, face):
        assert(face in self.meshes[m_idx].ref_battr)
        if face in self.meshes[m_idx].avail_face:
            self.meshes[m_idx].avail_face.pop(face)

        if (self.bdr_data.ndim == 1):
            self.bdr_data = np.array([[gbattr, m_idx, face]])
        else:
            self.bdr_data = np.append(self.bdr_data, [[gbattr, m_idx, face]], axis=0)
        return
    
    def appendInterface(self, midx1, midx2, face1, face2):
        assert(face1 in self.meshes[midx1].avail_face)
        assert(face2 in self.meshes[midx2].avail_face)
        if (midx1 < 0): midx1 += len(self.meshes)
        if (midx2 < 0): midx2 += len(self.meshes)

        self.meshes[midx1].avail_face.pop(face1)
        self.meshes[midx2].avail_face.pop(face2)

        if (face1 < face2):
            interface = (self.meshes[midx1].name, self.meshes[midx2].name,
                         face1, face2)
        else:
            interface = (self.meshes[midx2].name, self.meshes[midx1].name,
                         face2, face1)
            
        # add the interface if not exists.
        if interface not in self.ports:
            self.ports[interface] = len(self.ports)
        port = self.ports[interface]

        if (face1 < face2):
            if_data = [[midx1, midx2, face1, face2, port]]
        else:
            if_data = [[midx2, midx1, face2, face1, port]]

        if (self.if_data.ndim == 1):
            self.if_data = np.array(if_data)
        else:
            self.if_data = np.append(self.if_data, if_data, axis=0)
        return
    
    def save(self, filename):
        with h5py.File(filename, 'w') as f:
            # c++ currently cannot read datasets of string.
            # change to multiple attributes, only as a temporary implementation.
            grp = f.create_group("components")
            grp.attrs["number_of_components"] = len(self.comps)
            for k, comp in enumerate(self.comps):
                grp.attrs["%d" % k] = comp.name

            # component index of each mesh
            grp.create_dataset("meshes", (len(self.mesh_types),), data=self.mesh_types)

            # 3-dimension vector for translation / rotation
            mesh_configs = self.getMeshConfigs()
            grp.create_dataset("configuration", mesh_configs.shape, data=mesh_configs)

            grp = f.create_group("ports")
            grp.attrs["number_of_references"] = len(self.ports)
            for k in range(len(self.ports)):
                grp.attrs["%d" % k] = "port%d" % k

            grp.create_dataset("interface", self.if_data.shape, data=self.if_data)

            # boundary attributes
            f.create_dataset("boundary", self.bdr_data.shape, data=self.bdr_data)
        return
    
    def allFacesClosed(self):
        for mesh in self.meshes:
            if (len(mesh.avail_face) > 0):
                return False

        mesh_faces = []
        for mesh in self.meshes:
            mesh_faces += [mesh.ref_battr.copy()]

        for bdr in self.bdr_data:
            mesh_faces[bdr[1]].remove(bdr[2])

        for iface in self.if_data:
            mesh_faces[iface[0]].remove(iface[2])
            mesh_faces[iface[1]].remove(iface[3])

        closed = True
        for mesh_face in mesh_faces:
            if (len(mesh_face) != 0):
                return False

        return closed
