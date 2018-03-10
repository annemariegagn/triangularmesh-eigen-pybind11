import numpy as np
import trimeshpy_data
import trimeshpy.math as tmath
import laplacian_pybind as ls
import timeit
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk

#triangles = np.load(trimeshpy_data.cube_triangles)
#vertices = np.load(trimeshpy_data.cube_vertices)
#vertices_csc = csc_matrix(vertices)
#adjacency_matrix = edge_adjacency(triangles, vertices_csc)
#weights = np.squeeze(np.array(adjacency_matrix.sum(1)))
#print(weights)'''

file_name = trimeshpy_data.spot
mesh = TriMesh_Vtk(file_name, None)
triangles = mesh.get_triangles()
vertices = mesh.get_vertices()

np.save("tri_spot.npy", triangles)
np.save("vts_spot.npy", vertices)
