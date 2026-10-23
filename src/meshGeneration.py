import os

import numpy as np
from mpi4py import MPI
from dolfinx import io, mesh
import ufl


class MeshBase:

    def _finalise(self):
        self.gdim = self.domain.geometry.dim
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.domain.topology.create_connectivity(self.fdim, self.tdim)

        # exterior facets indices (global entity indices)
        self.boundary_facets = mesh.exterior_facet_indices(self.domain.topology)

        # FacetNormal(domain) yields the outward normal on the domain boundary facets
        self.n = ufl.FacetNormal(self.domain)

    def _bounds_from_geometry(self):
        # Global bounds across ranks
        comm = self.domain.comm
        x = self.domain.geometry.x

        if x.shape[0] > 0:
            local_min, local_max = x.min(axis=0), x.max(axis=0)
        else:
            local_min, local_max = np.full(3, np.inf), np.full(3, -np.inf)

        p_min, p_max = np.empty(3, dtype=np.float64), np.empty(3, dtype=np.float64)
        comm.Allreduce(np.ascontiguousarray(local_min, dtype=np.float64), p_min, op=MPI.MIN)
        comm.Allreduce(np.ascontiguousarray(local_max, dtype=np.float64), p_max, op=MPI.MAX)
        return p_min, p_max

    def _mean_cell_size(self):
        comm = self.domain.comm
        num_cells = self.domain.topology.index_map(self.tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        local_total = self.domain.h(self.tdim, cells).sum() if num_cells > 0 else 0.0

        total = comm.allreduce(local_total, op=MPI.SUM)
        count = comm.allreduce(num_cells, op=MPI.SUM)
        return total / count if count > 0 else 0.0


class BoxMesh(MeshBase):
    def __init__(self, meshDimensions, meshLengths, COMM_WORLD):
        self.nx = meshDimensions[0] # mesh divisions in x-direction
        self.ny = meshDimensions[1] # mesh divisions in y-direction
        self.nz = meshDimensions[2] # mesh divisions in z-direction

        self.lx = meshLengths[0] # domain length in x-direction
        self.ly = meshLengths[1] # domain length in y-direction
        self.lz = meshLengths[2] # domain length in z-direction

        self.dx = self.lx/self.nx
        self.dy = self.ly/self.ny
        self.dz = self.lz/self.nz

        self.p_min = np.array([1.0, 0, 0], dtype=np.float64)  # domain min corner
        self.p_max = np.array([1.0+self.lx,  self.ly,  self.lz], dtype=np.float64)  # domain max corner
        # create mesh and identify boundary facets
        self.domain = mesh.create_box(COMM_WORLD, [self.p_min, self.p_max], [self.nx, self.ny, self.nz],
                        cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)

        self._finalise()


class FileMesh(MeshBase):

    def __init__(self, mesh_file, COMM_WORLD, mesh_name="mesh"):
        if not os.path.exists(mesh_file):
            raise Exception("Mesh file, " + mesh_file + ", does not exist")

        suffix = os.path.splitext(mesh_file)[1].lower()
        match suffix:
            case ".msh":
                self.domain, _, _ = io.gmshio.read_from_msh(mesh_file, COMM_WORLD, gdim=3)
            case ".xdmf" | ".xmf":
                with io.XDMFFile(COMM_WORLD, mesh_file, "r") as xdmf:
                    self.domain = xdmf.read_mesh(name=mesh_name,
                                                 ghost_mode=mesh.GhostMode.shared_facet)
            case _:
                raise Exception("Unsupported mesh file type, " + suffix + ", for " + mesh_file
                                + ". Use .xdmf or .msh")

        self._finalise()

        self.p_min, self.p_max = self._bounds_from_geometry()
        self.lx, self.ly, self.lz = self.p_max - self.p_min

        # Mean spacing, cells not uniform
        self.dx = self.dy = self.dz = self._mean_cell_size()

        self.nx = self.ny = self.nz = None


def create_mesh(solver_parameters, meshDimensions, meshLengths, COMM_WORLD):
    mesh_type = solver_parameters.get("mesh_type", "box")

    match mesh_type:
        case "box":
            return BoxMesh(meshDimensions, meshLengths, COMM_WORLD)
        case "file":
            mesh_file = solver_parameters.get("mesh_file")
            if not mesh_file:
                raise Exception('Mandatory field, mesh_file, not included in solver.json '
                                'when mesh_type is "file"')
            return FileMesh(mesh_file, COMM_WORLD, solver_parameters.get("mesh_name", "mesh"))
        case _:
            raise Exception("Invalid mesh type, " + mesh_type + ", specified")
