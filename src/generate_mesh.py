# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path
import open3d as o3d
import numpy as np
#import pyrender
#import trimesh
import copy
import os
import time

import matplotlib.pyplot as plt

from src.align_frames import remove_cluster_artifacts
# import matplotlib.pyplot as plt


def recalculate_uniform_normals(pcd):
    
    #pcd = remove_cluster_artifacts(pcd)

    # Remover outliers residuais
    #pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Converter para CPU
    pcd_cpu = o3d.geometry.PointCloud()
    pcd_cpu.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pcd_cpu.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))

    print(f"Tem normais antes: {pcd_cpu.has_normals()}")
    print(f"Tipo: {type(pcd_cpu)}")
    
    # Recalcular normais uniformes
    pcd_cpu.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=30))
    
    print(f"Tem normais depois: {pcd_cpu.has_normals()}")
    print(f"Num normais: {len(pcd_cpu.normals)}")
    
    pcd_cpu.orient_normals_towards_camera_location(pcd.get_center())
    
    pcd_cpu.orient_normals_consistent_tangent_plane(k=30)
    
    return pcd_cpu


def remove_low_density_triangles(densities, mesh):
    densities_arr = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities_arr < np.percentile(densities_arr, 10))
    return mesh


# def fill_mesh_holes(pcd, mesh, densities_arr):
#     mesh = copy.deepcopy(mesh)
#     mesh.remove_vertices_by_mask(densities_arr < np.percentile(densities_arr, 2))

#     # Reparar buracos
#     meshfix = pymeshfix.MeshFix(
#         np.asarray(mesh.vertices),
#         np.asarray(mesh.triangles)
#     )
#     meshfix.repair()

#     # Reconstruir malha
#     mesh_fixed = o3d.geometry.TriangleMesh()
#     mesh_fixed.vertices  = o3d.utility.Vector3dVector(meshfix.points)
#     mesh_fixed.triangles = o3d.utility.Vector3iVector(meshfix.faces)

#     # Transferir cores
#     pcd_tree = o3d.geometry.KDTreeFlann(pcd)
#     mesh_colors = []
#     for vertex in np.asarray(mesh_fixed.vertices):
#         _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
#         mesh_colors.append(np.asarray(pcd.colors)[idx[0]])

#     mesh_fixed.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
#     mesh_fixed.compute_vertex_normals()

#     return mesh_fixed


# def render_mesh():
#     mesh_trimesh = trimesh.load("coca.ply")
#     mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh)

#     scene = pyrender.Scene()
#     scene.add(mesh_pyrender)

#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#     scene.add(camera)

#     light = pyrender.DirectionalLight(color=[1,1,1], intensity=2.0)
#     scene.add(light)

#     r = pyrender.OffscreenRenderer(640, 480)
#     color, depth = r.render(scene)

#     plt.imshow(color)
#     plt.show()
    if stop_in == ():
        o3d.io.write_point_cloud(os.path.join(save_path, f"{name}_reconstructed_pcd.ply"), aligned)
    else:
        o3d.io.write_point_cloud(os.path.join(save_path, f"{name}_reconstructed_pcd_with_{stop_in[0]}.ply"), aligned)
    
    
def generate_mesh(path, name, pcd, voxel_size, *stop_in):
    begin=time.time()
    print(f"tempo inicial de generate mesh:{begin}")
    # Uniformizar densidade
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    #_, ind = pcd.remove_radius_outlier(nb_points=50, radius=5)
    # pcd = pcd.select_by_index(ind)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    # pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    pcd = recalculate_uniform_normals(pcd)
    
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    R = pcd.get_rotation_matrix_from_xyz((np.pi, -np.pi / 4, 0))
    pcd.rotate(R, center=(0, 0, 0))
    print('Displaying input pointcloud ...')
    
    # pcd = remove_cluster_artifacts(pcd)
    # o3d.visualization.draw_geometries([pcd])
    
    # z_max = pts[:, 2].max()
    # z_corte = z_max - 50  # remove os últimos 50mm do topo — ajusta conforme necessário

    # indices = np.where(pts[:, 2] < z_corte)[0]
    # pcd_cortado = pcd.select_by_index(indices)
    
    print('Running Poisson surface reconstruction ...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=11, scale=1.0, linear_fit=True)

    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    
    # o3d.io.write_triangle_mesh(os.path.join(path, f"20260529_213208_frame_01_clean_mesh_after_poisson.ply"), mesh)
        
    densities_arr = np.asarray(densities)
    
    mesh_cpu = o3d.geometry.TriangleMesh()
    mesh_cpu.vertices  = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_cpu.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))

    mesh_final = copy.deepcopy(mesh_cpu)
    
    print(f"Vértices antes: {len(mesh_cpu.vertices)}")
    # mesh_final.remove_vertices_by_mask(densities_arr < np.percentile(densities_arr, 1))  # ← aqui
    print(f"Vértices depois: {len(mesh_final.vertices)}")
    
    #o3d.visualization.draw_geometries([mesh_final], mesh_show_back_face=True)

    # limpeza pós-remoção
    mesh_final.remove_degenerate_triangles()
    mesh_final.remove_non_manifold_edges()
    
    #o3d.visualization.draw_geometries([mesh_final], mesh_show_back_face=True)
    
    # ── Transferir cores ──────────────────────────────────────────────────────────
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    for vertex in np.asarray(mesh_final.vertices):
        _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 5)  # ← 5 neighbors
        mean_color = np.asarray(pcd.colors)[idx].mean(axis=0) 
        # _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors.append(mean_color) #(np.asarray(pcd.colors)[idx[0]])
    
    mesh_final.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
    
    #o3d.visualization.draw_geometries([mesh_final], mesh_show_back_face=True)
    
    print("computing vertex normals...")
    
    mesh_final.compute_vertex_normals()
    
    #o3d.visualization.draw_geometries([mesh_final], mesh_show_back_face=True)
    
    # # Transferir cores da nuvem para a malha
    # mesh = mesh.sample_points_poisson_disk(number_of_points=len(pcd.points))
    # mesh = remove_low_density_triangles(densities)
    
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    # mesh_colors = []

    # for vertex in np.asarray(mesh.vertices):
    #     _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
    #     mesh_colors.append(np.asarray(pcd.colors)[idx[0]])

    # mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
    
    # print('Displaying reconstructed mesh ...')
    
    #o3d.visualization.draw([mesh_final])

    print("Gravando malha")
    
    mesh_triangle      = o3d.t.geometry.TriangleMesh.from_legacy(mesh_final)
    mesh_filled = mesh_triangle.fill_holes()
    mesh_without_holes  = mesh_filled.to_legacy()
    
    mesh_smoothed = mesh_without_holes.filter_smooth_simple(number_of_iterations=8)
    mesh_smoothed.compute_vertex_normals()
    
    end = time.time() - begin
    print(f"tempo final de generate mesh:{end}")
    
    return mesh_final, mesh_without_holes, mesh_smoothed