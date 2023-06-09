import numpy as np
from copy import deepcopy
import h5py

def ManhattanDistance(loc1, loc2):
    assert(loc1.size == loc2.size)
    assert(len(loc1.shape) == 1)
    dist = loc1 - loc2
    return np.sum(np.abs(dist))

class Component:
    name = ''

    num_face = -1
    dim = 2
    
    ref_battr = []
    face_map = {}       # {location of counterpart component: matching ref_battr}

    def __init__(self):
        return
    
    def switchBdrAttr(self, attr_dict, target_face, target_attr):
        # switch upstream face index with the face which has attr 4.
        attr_list = list(attr_dict.values())
        face_list = list(attr_dict.keys())
        uidx = attr_list.index(target_attr)
        if (face_list[uidx] != target_face):
            attr_dict[face_list[uidx]] = attr_dict[target_face]
            attr_dict[target_face] = target_attr
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
        return
    
    def DetermineGlobalAttr(self, uloc, loc, dloc, closure=False):
        # determine no-slip wall faces of the last mesh.
        faces = self.ref_battr.copy()
        uface = self.face_map[tuple(uloc - loc)]
        dface = self.face_map[tuple(dloc - loc)]

        # downstream face index always has global bdr attr 2.
        gbattr = {}
        for face in faces:
            attr = face - dface + 2
            if (attr < 1): attr += self.num_face
            if (attr > self.num_face): attr -= self.num_face
            gbattr[face] = attr

        # switch upstream face index with the face which has attr 4.
        self.switchBdrAttr(gbattr, uface, 4)

        gbattr.pop(uface)
        if (not closure): gbattr.pop(dface)

        return gbattr.values(), gbattr.keys()
    
class PipeHub(Component):
    name = 'pipe-hub'

    def __init__(self):
        self.num_face = 5
        self.ref_battr = [1, 2, 3, 4, 5]
        self.face_map = {(0, -1):  1,
                         (1, 0):   2,
                         (0, 1):   3,
                         (-1, 0):  4}
        return
    
    def DetermineGlobalAttr(self, uloc, loc, dloc, closure=False):
        # determine no-slip wall faces of the last mesh.
        faces = self.ref_battr.copy()
        uface = self.face_map[tuple(uloc - loc)]
        dface = self.face_map[tuple(dloc - loc)]

        # downstream face index always has global bdr attr 2.
        gbattr = {}
        for face in faces:
            attr = face - dface + 2
            if (attr < 1): attr += self.num_face
            if (attr > self.num_face): attr -= self.num_face
            gbattr[face] = attr

        # Keep the attr 5.
        self.switchBdrAttr(gbattr, 5, 5)

        # switch upstream face index with the face which has attr 4.
        self.switchBdrAttr(gbattr, uface, 4)

        gbattr.pop(uface)
        if (not closure): gbattr.pop(dface)

        return gbattr.values(), gbattr.keys()
    
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
    loc = np.array([[]], dtype=int)
    bdr_data = np.array([[]], dtype=int)
    if_data = np.array([[]], dtype=int)

    def __init__(self):
        return
    
    def addComponent(self, component):
        assert(self.dim == component.dim)
        self.comps += [component]
        return
    
    def close(self):
        raise RuntimeError('Abstract method Configuration.close()!')
        return
    
    def getMeshConfigs(self):
        mesh_configs = np.zeros([len(self.meshes), 6])
        mesh_configs[:,:self.dim] = self.loc
        return mesh_configs
    
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

class ChannelConfig(Configuration):
    prefix = 'channel.config'

    def __init__(self, start_comp='empty'):
        if (start_comp == 'empty'):
            self.comps = [Empty()]
        elif (start_comp == 'pipe-hub'):
            self.comps = [PipeHub()]
        self.meshes = [self.comps[0]]
        self.mesh_types = [0]
        self.loc = np.zeros([1, self.dim], dtype=int)
        # self.loc[1, 0] = 1

        # bdr_data / mesh_idx / comp_battr
        # self.bdr_data = np.array([[4, 0, 4],
        #                           [1, 0, 1],
        #                           [3, 0, 3]])
        self.bdr_data = np.array([[4, 0, 4]])
        # if (start_comp == 'pipe-hub'):
        #     self.bdr_data = np.append(self.bdr_data, [[5, 0, 5]], axis=0)
        
        # mesh1 / mesh2 / battr1 / battr2 / port_idx
        # self.if_data = np.array([[0, 1, 2, 4, 0]])
        self.if_data = np.array([])
        self.ports = {(self.comps[0].name, self.comps[0].name, 2, 4): 0}
        return
    
    def getAvailableFaces(self):
        assert(self.loc.shape[0] >= 1)
        avail_faces = []
        avail_locs = []

        if (self.loc.shape[0] == 1):
            udir = (-1, 0)
        else:
            udir = tuple(self.loc[-2] - self.loc[-1])
        uface = self.meshes[-1].face_map[udir]

        for key, val in self.meshes[-1].face_map.items():
            if (val == uface): continue

            test_loc = self.loc[-1] + np.array(key)
            if (self.isLocationAvailable(test_loc)):
                avail_faces += [val]
                avail_locs += [np.array(test_loc)]

        return avail_faces, avail_locs
    
    def isLocationAvailable(self, new_loc):
        avail = True

        # # only one neighbor can exist
        # neighbor = 0
        # for k in range(len(self.meshes)):
        #     dist = ManhattanDistance(new_loc, self.loc[k])
        #     if (dist == 1): neighbor += 1
        #     if (neighbor > 1):
        #         avail = False
        #         break

        # no one can overlap
        for k in range(len(self.meshes)):
            dist = ManhattanDistance(new_loc, self.loc[k])
            if (dist == 0):
                avail = False
                break

        return avail
    
    def addMesh(self, comp_idx, new_loc):
        assert(self.isLocationAvailable(new_loc))

        # upstream_face = self.meshes[-1].face_map[tuple(self.loc[-2] - self.loc[-1])]
        downstream_face = self.meshes[-1].face_map[tuple(new_loc - self.loc[-1])]

        # no-slip wall boundary for the last mesh.
        m_idx = len(self.meshes) - 1
        if (m_idx == 0):
            uloc = np.array([-1, 0]) + self.loc[0]
        else:
            uloc = self.loc[-2]
        gbattrs, faces = self.meshes[-1].DetermineGlobalAttr(uloc, self.loc[-1], new_loc)
        for gbattr, face in zip(gbattrs, faces):
            self.bdr_data = np.append(self.bdr_data, [[gbattr, m_idx, face]], axis=0)

        # interface between the last mesh and the new mesh
        c2_upstream = self.comps[comp_idx].face_map[tuple(self.loc[-1] - new_loc)]
        if (downstream_face < c2_upstream):
            interface = (self.meshes[-1].name, self.comps[comp_idx].name,
                         downstream_face, c2_upstream)
        else:
            interface = (self.comps[comp_idx].name, self.meshes[-1].name,
                         c2_upstream, downstream_face)
            
        # add the interface if not exists.
        if interface not in self.ports:
            self.ports[interface] = len(self.ports)
        port = self.ports[interface]

        if (downstream_face < c2_upstream):
            if_data = [[m_idx, m_idx+1, downstream_face, c2_upstream, port]]
        else:
            if_data = [[m_idx+1, m_idx, c2_upstream, downstream_face, port]]

        if (self.if_data.ndim == 1):
            self.if_data = np.array(if_data)
        else:
            self.if_data = np.append(self.if_data, if_data, axis=0)

        # add the new mesh and its location.
        self.meshes += [self.comps[comp_idx]]
        self.mesh_types += [comp_idx]
        self.loc = np.append(self.loc, [new_loc], axis=0)
        return
    
    def close(self):
        m_idx = len(self.meshes) - 1
        new_loc = self.loc[-1] + (self.loc[-1] - self.loc[-2])
        gbattrs, faces = self.meshes[-1].DetermineGlobalAttr(self.loc[-2], self.loc[-1], new_loc, True)
        for gbattr, face in zip(gbattrs, faces):
            self.bdr_data = np.append(self.bdr_data, [[gbattr, m_idx, face]], axis=0)

        if not (self.allFacesClosed()):
            raise RuntimeError('ChannelConfig is not closed even after close()!')
        return
    
    def allFacesClosed(self):
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
                closed = False
                break

        return closed
    
    def GenerateAllConfigs(self, extension, offset):
        assert(extension >= 0)
        if (extension == 0):
            self.close()
            filename = '%s-%05d.h5' % (self.prefix, offset)
            print(filename)
            self.save(filename)
            return 1

        avail_faces, avail_locs = self.getAvailableFaces()
        num_finals = 0
        for new_loc in avail_locs:
            config = deepcopy(self)
            config.addMesh(0, new_loc)
            num_finals += config.GenerateAllConfigs(extension-1, offset + num_finals)

        return num_finals
    
    def CreateRandomConfig(self, extension, filename):
        config = deepcopy(self)
        for k in range(extension):
            avail_faces, avail_locs = config.getAvailableFaces()
            idx = np.random.randint(len(avail_locs))
            config.addMesh(0, avail_locs[idx])
        config.close()
        config.save(filename)
        return

if __name__ == "__main__":
    example = ChannelConfig('pipe-hub')
    # example.GenerateAllConfigs(3, 0)

    example.CreateRandomConfig(8, 'channel.9comp.h5')

    # avail_faces, avail_locs = example.getAvailableFaces()
    # # print(avail_locs)

    # example.addMesh(0, avail_locs[-1])

    # # avail_faces, avail_locs = example.getAvailableFaces()
    # # # print(avail_locs)

    # # example.addMesh(0, avail_locs[-1])

    # example.close()
    
    # example.save('channel.4.h5')
