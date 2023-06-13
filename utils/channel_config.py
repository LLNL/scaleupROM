from config import *

def ManhattanDistance(loc1, loc2):
    assert(loc1.size == loc2.size)
    assert(len(loc1.shape) == 1)
    dist = loc1 - loc2
    return np.sum(np.abs(dist))

class ChannelConfig(Configuration):
    prefix = 'channel.config'

    def __init__(self, start_comp='empty'):
        if (start_comp == 'empty'):
            self.comps = [Empty()]
        elif (start_comp == 'pipe-hub'):
            self.comps = [PipeHub()]
        self.meshes = [deepcopy(self.comps[0])]
        self.mesh_types = [0]
        self.loc = np.zeros([1, self.dim], dtype=int)
        # self.loc[1, 0] = 1

        # bdr_data / mesh_idx / comp_battr
        # self.bdr_data = np.array([[4, 0, 4],
        #                           [1, 0, 1],
        #                           [3, 0, 3]])
        self.bdr_data = np.array([[4, 0, 4]])
        self.meshes[0].avail_face.pop(4)
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

        for face, loc in self.meshes[-1].avail_face.items():
            test_loc = self.loc[-1] + np.array(loc)
            if (self.isLocationAvailable(test_loc)):
                avail_faces += [face]
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
        d_face = self.meshes[-1].face_map[tuple(new_loc - self.loc[-1])]

        # no-slip wall boundary for the last mesh.
        m_idx = len(self.meshes) - 1
        if (m_idx == 0):
            uloc = np.array([-1, 0]) + self.loc[0]
        else:
            uloc = self.loc[-2]
        gbattrs, faces = self.DetermineGlobalAttr(-1, uloc, self.loc[-1], new_loc)
        for gbattr, face in zip(gbattrs, faces):
            self.appendBoundary(gbattr, m_idx, face)

        # add new mesh and its type.
        Configuration.addMesh(self, comp_idx, new_loc)

        # interface between the last mesh and the new mesh
        u_face2 = self.meshes[m_idx+1].face_map[tuple(self.loc[-2] - self.loc[-1])]
        self.appendInterface(m_idx, m_idx+1, d_face, u_face2)
        return
    
    def close(self):
        m_idx = len(self.meshes) - 1
        new_loc = self.loc[-1] + (self.loc[-1] - self.loc[-2])
        gbattrs, faces = self.DetermineGlobalAttr(-1, self.loc[-2], self.loc[-1], new_loc, True)
        for gbattr, face in zip(gbattrs, faces):
            self.appendBoundary(gbattr, m_idx, face)

        if not (self.allFacesClosed()):
            raise RuntimeError('ChannelConfig is not closed even after close()!')
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
    
    def DetermineGlobalAttr(self, midx, uloc, loc, dloc, closure=False):
        mesh = self.meshes[midx]
        # determine no-slip wall faces of the last mesh.
        faces = mesh.ref_battr.copy()
        uface = mesh.face_map[tuple(uloc - loc)]
        dface = mesh.face_map[tuple(dloc - loc)]

        # downstream face index always has global bdr attr 2.
        gbattr = {}
        for face in faces:
            attr = face - dface + 2
            if (attr < 1): attr += mesh.num_face
            if (attr > mesh.num_face): attr -= mesh.num_face
            gbattr[face] = attr

        # Keep the attr 5.
        if 5 in gbattr:
            self.switchBdrAttr(gbattr, 5, 5)

        # switch upstream face index with the face which has attr 4.
        self.switchBdrAttr(gbattr, uface, 4)

        gbattr.pop(uface)
        if (not closure): gbattr.pop(dface)

        return gbattr.values(), gbattr.keys()
    
    def switchBdrAttr(self, attr_dict, target_face, target_attr):
        # switch upstream face index with the face which has attr 4.
        attr_list = list(attr_dict.values())
        face_list = list(attr_dict.keys())
        uidx = attr_list.index(target_attr)
        if (face_list[uidx] != target_face):
            attr_dict[face_list[uidx]] = attr_dict[target_face]
            attr_dict[target_face] = target_attr
        return

if __name__ == "__main__":
    example = ChannelConfig('pipe-hub')
    # example = ChannelConfig()
    example.GenerateAllConfigs(3, 0)

    example.CreateRandomConfig(8, 'channel.9comp.h5')

    # avail_faces, avail_locs = example.getAvailableFaces()
    # # print(avail_locs)

    # example.addMesh(0, avail_locs[-1])

    # # avail_faces, avail_locs = example.getAvailableFaces()
    # # # print(avail_locs)

    # # example.addMesh(0, avail_locs[-1])

    # example.close()
    
    # example.save('channel.4.h5')
