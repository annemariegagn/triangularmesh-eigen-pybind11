import numpy as np
import trimeshpy.math as tmath
#import trimeshpy.trimesh_vtk as TriMesh_Vtk
import laplacian_pybind as ls
import timeit
from scipy.sparse.csc import csc_matrix
from trimeshpy.math.mesh_global import G_DTYPE

triangles = np.load("tri_spot.npy")
vertices = np.load("vts_spot.npy").astype(G_DTYPE)


ntri = triangles.shape[0]
nvert = vertices.shape[0]
niter = 10
step = 0.01
backward_step = True

print(ntri)
print(nvert)
print(niter)
print(backward_step)

# init
vertices_csc = csc_matrix(vertices, dtype=G_DTYPE)
diffusion_step = step * np.ones(len(vertices))

# smooth python
start_python = timeit.default_timer()
adjacency_matrix = tmath.edge_adjacency(triangles, vertices_csc)
end_python = timeit.default_timer()
print("temps adj dans python:")
print(end_python - start_python)

start_python = timeit.default_timer()
laplacian_matrix = tmath.laplacian(adjacency_matrix, diag_of_1=True)
end_python = timeit.default_timer()
print("temps lap dans python:")
print(end_python - start_python)


start_python = timeit.default_timer()
for i in range(niter):
    next_vertices_csc = tmath.euler_step(
        laplacian_matrix, vertices_csc, diffusion_step, backward_step)
    vertices_csc = next_vertices_csc
end_python = timeit.default_timer()
print("temps step dans python:")
step_time_python = end_python - start_python
print(step_time_python)


# Pybind11
print("----------------------------------------------------------------------------------------------------------------")
print("vertices out of before pybind11")
vts = np.array(vertices, dtype='f', order = 'F')
tri = np.array(triangles, dtype = 'i4', order = 'F')

start_pybind = timeit.default_timer()
vts2 = ls.Laplacian(tri, vts, ntri, nvert, step, niter, backward_step)
end_pybind = timeit.default_timer()
print("total pybind:")
print(end_pybind - start_pybind)



print("step_time/iter python:")
print(step_time_python/float(niter))


#mesh = TriMesh_Vtk(triangles, vts)
#mesh

print("----------------------------------------------------------------------------------------------------------------")
print("vertices out of pybind11")
print(vts2)
print("----------------------------------------------------------------------------------------------------------------")
print("vertices out of python")
print(vertices_csc.todense())
