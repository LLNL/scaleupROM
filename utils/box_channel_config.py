import numpy as np
from copy import deepcopy
from config import Configuration, Empty, ObjectInSpace
from channel_config import ManhattanDistance

class BoxChannelConfig(Configuration):
    prefix = 'box-channel.config'
    nx = -1
    ny = -1
    nmesh = -1
    avail_locs = []
    face_map = {(0, -1):  1,
                (1, 0):   2,
                (0, 1):   3,
                (-1, 0):  4}
    comp_used = []

    def __init__(self, nx_, ny_):
        self.nx, self.ny = nx_, ny_
        self.nmesh = self.nx * self.ny
        self.avail_locs = []
        for i in range(self.nx):
            for j in range(self.ny):
                self.avail_locs += [np.array([i, j], dtype=int)]

        self.test_locs = [k for k in range(self.nmesh)]
        self.test = []
        self.comp_used = []
        return
    
    def addComponent(self, component):
        for key, val in self.face_map.items():
            assert(key in component.face_map)
            assert(component.face_map[key] == val)

        Configuration.addComponent(self, component)
        if (len(self.comp_used) == 0):
            self.comp_used = [False]
        else:
            self.comp_used += [False]
        return

    
    def isLocationAvailable(self, new_loc):
        for loc in self.avail_locs:
            if ((loc == new_loc).all()):
                return True
        return False
    
    def addMesh(self, comp_idx, loc_idx):
        assert((loc_idx >= 0) and (loc_idx < len(self.avail_locs)))

        # add new mesh and its type.
        new_loc = self.avail_locs[loc_idx]
        Configuration.addMesh(self, comp_idx, new_loc)

        for k in range(len(self.meshes)):
            face1, face2 = self.interfaceForPair(k,-1)
            if ((face1 < 0) or (face2 < 0)): continue
            self.appendInterface(k, -1, face1, face2)

        self.avail_locs.remove(new_loc)
        self.comp_used[comp_idx] = True
        return
    
    def interfaceForPair(self, midx1, midx2):
        mesh1, mesh2 = self.meshes[midx1], self.meshes[midx2]
        loc1, loc2 = self.loc[midx1], self.loc[midx2]
        face1, face2 = -1, -1

        if (tuple(loc2-loc1) in mesh1.face_map):
            face1 = mesh1.face_map[tuple(loc2-loc1)]
        if (tuple(loc1-loc2) in mesh2.face_map):
            face2 = mesh2.face_map[tuple(loc1-loc2)]
        return face1, face2
    
    def close(self):
        assert(len(self.meshes) == self.nmesh)
        assert(len(self.avail_locs) == 0)

        for midx, mesh in enumerate(self.meshes):
            avail_face = deepcopy(mesh.avail_face)
            for face, loc in avail_face.items():
                self.appendBoundary(face, midx, face)
            if 5 in mesh.ref_battr:
                self.appendBoundary(5, midx, 5)

        if not (self.allFacesClosed()):
            raise RuntimeError('ChannelConfig is not closed even after close()!')
        return
    
    def GenerateAllConfigs(self, offset):
        # return num_finals
        assert(len(self.avail_locs) >= 0)
        if (len(self.avail_locs) == 0):
            self.close()
            filename = '%s-%05d.h5' % (self.prefix, offset)
            print(filename)
            self.save(filename)
            return 1

        num_finals = 0
        for cidx in range(len(self.comps)):
            config = deepcopy(self)
            config.addMesh(cidx, 0)
            num_finals += config.GenerateAllConfigs(offset + num_finals)

        return num_finals
    
    def CreateRandomConfig(self, filename):
        config = deepcopy(self)
        for k in range(len(self.avail_locs)):
            cidx = np.random.randint(len(self.comps))
            config.addMesh(cidx, 0)
        config.close()
        config.save(filename)
        return
    
    def save(self, filename):
        comp0, meshtype0 = self.removeUnusedComponents()

        Configuration.save(self, filename)
        self.comps, self.mesh_types = comp0, meshtype0
        return
    
    def removeUnusedComponents(self):
        assert(len(self.comps) == len(self.comp_used))
        orig_comps = deepcopy(self.comps)
        orig_mesh_types = deepcopy(self.mesh_types)

        idx_map = [-1] * len(self.comps)
        new_idx, new_comps = 0, []
        for k, used in enumerate(self.comp_used):
            if (used):
                idx_map[k] = new_idx
                new_comps += [self.comps[k]]
                new_idx += 1
        self.mesh_types = [idx_map[mesh_type] for mesh_type in self.mesh_types]
        self.comps = new_comps

        for mtype in self.mesh_types:
            assert(mtype >= 0)

        return orig_comps, orig_mesh_types

if __name__ == "__main__":
    example = BoxChannelConfig(1,1)
    example.addComponent(Empty())
    example.addComponent(ObjectInSpace('square-circle'))
    example.addComponent(ObjectInSpace('square-square'))
    example.addComponent(ObjectInSpace('square-triangle'))
    example.addComponent(ObjectInSpace('square-star'))

    example.GenerateAllConfigs(0)

    # example.CreateRandomConfig('channel.random.h5')

    # avail_faces, avail_locs = example.getAvailableFaces()
    # # print(avail_locs)

    # example.addMesh(0, avail_locs[-1])

    # # avail_faces, avail_locs = example.getAvailableFaces()
    # # # print(avail_locs)

    # # example.addMesh(0, avail_locs[-1])

    # example.close()
    
    # example.save('channel.4.h5')
