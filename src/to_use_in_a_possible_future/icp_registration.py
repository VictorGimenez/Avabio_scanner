# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""ICP (Iterative Closest Point) registration algorithm"""

import open3d as o3d
import numpy as np
import glob
import copy


def estimate_normals(pcd, radius):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30
        )
    )


def preprocess(pcd, voxel=0.005):
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(30, 1.5)
    return pcd

# def draw_registration_result(source, target, transformation, title=""):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
    
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
    
#     source_temp.transform(transformation)
#     print(f"\n[VISUALIZAÇÃO] {title}")
#     o3d.visualization.draw([source_temp, target_temp])


# def point_to_point_icp(source, target, threshold, trans_init):
#     print("Apply point-to-point ICP")
#     reg = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#     print(reg)
#     print("Transformation is:")
#     print(reg.transformation, "\n")
    
#     #draw_registration_result(source, target, reg.transformation)
#     return reg


# def point_to_plane_icp(source, target, threshold, trans_init):
#     print("Apply point-to-plane ICP")
    
#     reg = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane()
#     )
#     print(reg)
#     print("Transformation is:")
#     print(reg.transformation, "\n")
    
#     #draw_registration_result(source, target, reg_p2l.transformation)
#     return reg


# if __name__ == "__main__":
#     # pcd_data = o3d.data.DemoICPPointClouds()
#     source = o3d.io.read_point_cloud("/home/victorbg/Documents/3DBodyCircumferenceEvaluator/point_clouds/Victor_full_3.ply")
#     target = o3d.io.read_point_cloud("/home/victorbg/Documents/3DBodyCircumferenceEvaluator/point_clouds/Victor_full_4.ply")
#     #source = o3d.io.read_point_cloud(pcd_data.paths[0])
#     #target = o3d.io.read_point_cloud(pcd_data.paths[1])
    
#     print("[INFO] Points source:", np.asarray(source.points).shape[0])
#     print("[INFO] Points target:", np.asarray(target.points).shape[0])
    
#     #o3d.visualization.draw_geometries([source])
#     #o3d.visualization.draw_geometries([target])
    
#     o3d.visualization.draw([source, target])
    
#     extent = np.ptp(np.asarray(source.points),axis=0)
    
#     radius = np.linalg.norm(extent) * 0.02
#     max_nn = 30
    
#     print(f"[INFO] Raio de normais: {radius:.4f}")
    
#     source.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius,
#             max_nn=max_nn
#         )
#     )

#     target.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=radius,
#             max_nn=max_nn
#         )
#     )
    
#     o3d.visualization.draw_geometries(
#         [source]
#         #,
#         #point_show_normal=True
#     )
    
#     angle_deg = 10.0  # <<< AJUSTE DEPOIS, agora deixe 10
#     angle = np.deg2rad(angle_deg)

#     R = source.get_rotation_matrix_from_xyz((0, angle, 0))
#     trans_init = np.eye(4)
#     trans_init[:3, :3] = R

#     print("\n[INFO] Usando rotação inicial de", angle_deg, "graus")

#     draw_registration_result(
#         source, target, trans_init,
#         title="Transformação inicial (ANTES do ICP)"
#     )
    
#     threshold = np.linalg.norm(extent) * 0.05
#     print(f"[INFO] Threshold ICP: {threshold:.4f}")
    
#     evaluation = o3d.pipelines.registration.evaluate_registration(
#         source, target, threshold, trans_init
#     )
#     print("\n[INFO] Avaliação inicial")
#     print(evaluation)
    
#     reg_p2p = point_to_point_icp(source, target, threshold, trans_init)
#     reg_p2l = point_to_plane_icp(source, target, threshold, trans_init)
    
#     source.transform(reg_p2l.transformation)
    
#     print("\n[TESTE 3] Source transformado + Target")
#     o3d.visualization.draw([source, target])
    
#     merged = target + source
    
#     o3d.visualization.draw([merged])

#     # point_to_point_icp(source, target, threshold, trans_init)
#     # point_to_plane_icp(source, target, threshold, trans_init)
    
#     # # opcional: salvar
#     # o3d.io.write_point_cloud(
#     #     "merged_result.ply",
#     #     merged
#     # )

if __name__ == "__main__":

    ply_files = sorted(
        glob.glob("/home/victorbg/Documents/3DBodyCircumferenceEvaluator/point_clouds/Victor_full_*.ply")
    )

    print(f"\nfiles from folder:{ply_files} \n[INFO] Total de frames: {len(ply_files)}")

    # ============================
    # 1️⃣ Primeira nuvem
    # ============================
    pcd_global = o3d.io.read_point_cloud(ply_files[0])
    pcd_global = preprocess(pcd_global)

    extent = np.ptp(np.asarray(pcd_global.points), axis=0)
    radius = np.linalg.norm(extent) * 0.02

    estimate_normals(pcd_global, radius)

    # ============================
    # 2️⃣ Loop incremental
    # ============================
    for i in range(1, len(ply_files)):

        print(f"\n[INFO] Registrando frame {i}")

        pcd = o3d.io.read_point_cloud(ply_files[i])
        pcd = preprocess(pcd)
        estimate_normals(pcd, radius)

        # 🔹 rotação inicial (base giratória)
        angle_deg = 10.0  # AJUSTE se necessário
        angle = np.deg2rad(angle_deg)

        R = pcd.get_rotation_matrix_from_xyz((0, angle, 0))
        trans_init = np.eye(4)
        trans_init[:3, :3] = R

        # 🔹 ICP
        threshold = np.linalg.norm(extent) * 0.05

        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            pcd_global,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        print("Fitness:", reg.fitness)
        print("RMSE:", reg.inlier_rmse)

        # 🔹 aplica transformação
        pcd.transform(reg.transformation)

        # 🔹 merge
        pcd_global += pcd

        # 🔹 limpeza leve
        pcd_global = pcd_global.voxel_down_sample(0.005)
        pcd_global, _ = pcd_global.remove_statistical_outlier(30, 1.5)

        # 🔹 recalcula normais do acumulado
        estimate_normals(pcd_global, radius)

    # ============================
    # 3️⃣ Resultado final
    # ============================
    print("\n[INFO] Reconstrução final")
    o3d.visualization.draw_geometries([pcd_global])

    o3d.io.write_point_cloud(
        "body_reconstruction_final.ply",
        pcd_global
    )
