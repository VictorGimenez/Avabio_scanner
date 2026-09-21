
import open3d as o3d
import os
import numpy as np
import cv2 as cv
import sys
#np.set_printoptions(threshold=sys.maxsize)

from pathlib import Path
from rembg import remove, new_session

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#     Old methodology
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# def aligning(name, limit):
#     print("Load a ply point cloud, print it, and render it")
    
#     pcd = [o3d.io.read_point_cloud(f"/home/victorbg/Documents/3DBodyCircumferenceEvaluator/point_clouds/{name}_{i}.ply") for i in range(0, limit)]

#     all_center = (0,0,0)  #Anchoring the first center as the universal center
    
#     #Para ângulos de 10° em 10°
#     #         t  t   t     r r r s s s
#     # 1° .ply 0  0       0        0 0 0 1 1 1
#     # 2° .ply 0  -0.25   1      -15 0 0 1 1 1
#     # 3° .ply 0  3   3      -30 0 0 1 1 1
#     # 4° .ply 0  3   3.5    -45 0 0 1 1 1
#     # 5° .ply 0  4.5 16     -60 0 0 1 1 1
#     # 6° .ply 0  5.5 1.5  -75 0 0 1 1 1
#     # 7° .ply 0  0   3    -90 0 0 1 1 1
#     # 8° .ply 0  1.5 1.5  -105 0 1 1 1
#     # 9° .ply 0  0   0    -120 0 1 1 1
#     # 10° .ply 0 -1.5 1.5 - 0 0 1 1 1
#     # 11° .ply 0  0   3   180 0 0 1 1 1
#     # 12° .ply 0  1.5 1.5 270 0 0 1 1 1
#     # 13° .ply 0  0   0     0 0 0 1 1 1
#     # 14° .ply 0 -1.5 1.5  90 0 0 1 1 1
#     # 15° .ply 0  0   3   180 0 0 1 1 1
#     # 16° .ply 0  1.5 1.5 270 0 0 1 1 1
#     # 17° .ply 0  0   0     0 0 0 1 1 1
#     # 18° .ply 0 -1.5 1.5  90 0 0 1 1 1
#     # 19° .ply 0  0   3   180 0 0 1 1 1
#     # 20° .ply 0  1.5 1.5 270 0 0 1 1 1
#     # 21° .ply 0  0   0     0 0 0 1 1 1
#     # 22° .ply 0 -1.5 1.5  90 0 0 1 1 1
#     # 23° .ply 0  0   3   180 0 0 1 1 1
#     # 24° .ply 0  1.5 1.5 270 0 0 1 1 1
    
    
#     #Para ângulos de 90° em 90°
#     #         t  t   t       r r r s s s
#     # 0° .ply 0  0   0       0 0 0 1 1 1
#     # 6° .ply 0 -1.4 1.4    90 0 0 1 1 1
#     #12° .ply 0  0   2.8   180 0 0 1 1 1
#     #18° .ply 0  1.4 1.4   270 0 0 1 1 1
    
#     R1 = pcd[0].get_rotation_matrix_from_xyz((np.deg2rad([-90,0,0])))
    
#     pcd[1].rotate(R1, all_center)
    
#     R2 = pcd[0].get_rotation_matrix_from_xyz((np.deg2rad([-180,0,0])))
    
#     pcd[2].rotate(R2, all_center)
    
#     R3 = pcd[0].get_rotation_matrix_from_xyz((np.deg2rad([-270,0,0])))
    
#     pcd[3].rotate(R3, all_center)
    
#     whole_body = [pcd[0]+pcd[1]+pcd[2]+pcd[3]]
    
#     o3d.visualization.draw_geometries(whole_body)
    
#     #Translating after rotation
    
#     pcd[1] = pcd[1].translate((0, -1.4, 1.55))  #0, -1.3, 1.3
    
#     pcd[2] = pcd[2].translate((0, 0.1, 2.9))  #0, 0, 2.65
    
#     pcd[3] = pcd[3].translate((0, 1.4, 1.4))  #0, 1.3, 1.3

#     whole_body = [pcd[0]+pcd[1]+pcd[2]+pcd[3]]
    
#     o3d.visualization.draw_geometries(whole_body)
    
    
# def cluster_volume(pcd, label, labels):
#     pts = np.asarray(
#         pcd.select_by_index(np.where(labels == label)[0]).points
#     )
#     # size = np.ptp(pts, axis=0)  # dx, dy, dz
#     # return size[0] * size[1] * size[2]
#     return np.prod(pts.max(axis=0) - pts.min(axis=0))


def cluster_volume(pcd, label, labels):
    idx = np.where(labels == label)[0]
    pts = np.asarray(pcd.select_by_index(idx).points)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins
    return dims[0] * dims[1] * dims[2]


def cluster_looks_like_human(pcd, label, labels, min_height=1200, max_width=800, min_points=500):
    
    idx = np.where(labels == label)[0]
    
    if len(idx) < min_points:
        return False

    pts  = np.asarray(pcd.select_by_index(idx).points)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins

    height  = dims[1]  # Y axis = vertical
    x_width = dims[0]
    z_width = dims[2]

    # # Human form criteria
    enough_tall       = height  >= min_height
    x_not_much_large  = x_width <= max_width
    z_not_much_large  = z_width <= max_width
    # more_tall_than_large = height > max(x_width, y_width)
    # return enough_tall and x_not_much_large and y_not_much_large and more_tall_than_large

    clearly_not_human = (x_width > max_width and z_width > max_width) and height < min_height # clearly_not_human = (x_width > 1500 or z_width > 1500) and height < 300

    return not clearly_not_human and (enough_tall or x_not_much_large or z_not_much_large)    # return not clearly_not_human
    

def cropping(): 
    base = Path("./point_clouds/")
    # path = sorted(base.glob(f"*_clean.ply"))[0]
    
    for path in sorted(base.glob(f"*_clean.ply")):  #for i, path in enumerate(sorted(base.glob(f"{name}_*.ply")), start=0):
        if "raw" not in path.name and "complete" not in path.name and "cropped" not in path.name:  #if "raw" not in path.name and "complete" not in path.name:
            
            pcd = o3d.io.read_point_cloud(str(path))
            
            # 1. Remover outliers estatísticos antes do DBSCAN
            labels = np.array(pcd.cluster_dbscan(eps=40, min_points=20, print_progress=False))
            #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            #    #labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
            #    labels = np.array(pcd.cluster_dbscan(eps=0.04, min_points=20, print_progress=False))
            
            # MIN_CLUSTER_POINTS = 200
            
            # keep_indices = np.where(
            #     (labels != -1) &  # remove pontos isolados
            #     np.isin(labels, [l for l in np.unique(labels)
            #                     if l != -1 and np.sum(labels == l) >= MIN_CLUSTER_POINTS])
            # )[0]
            
            # # 2. DBSCAN
            # labels = np.array(pcd.cluster_dbscan(eps=40, min_points=20, print_progress=False))
            valid_labels = [l for l in np.unique(labels) if l != -1 and np.sum(labels == l) >= 200]
            # print(f"   Clusters encontrados: {len(valid_labels)}")

            if not valid_labels:
                print(" Nenhum cluster encontrado")
            #     pass #continue >> usar somente no for
            
            # # 3. Filter only clusters with human form
            # candidates = [l for l in valid_labels if cluster_looks_like_human(pcd, l, labels)]
            # print(f"   Candidatos humanos: {len(candidates)}")
            
            # if not candidates:
            #     # Fallback: biggest cluster by volume
            #     print(" Nenhum candidato humano — usando maior cluster")
            #     candidates = [max(valid_labels, key=lambda l: cluster_volume(pcd, l, labels))]
            
            # 4. Pick the biggest one between candidates
            main_label = max(valid_labels, key=lambda l: np.sum(labels == l))
            main_pts = np.asarray(pcd.select_by_index(np.where(labels == main_label)[0]).points)
            main_center = main_pts.mean(axis=0)
            print(f"   Cluster principal: {np.sum(labels == main_label):,} pontos")

            # Manter só clusters próximos ao principal (400mm = 40cm)
            MAX_DIST = 400  # mm
            
            # # # 5. Include human clusters next the mainly one
            # # h, w = mask.shape
            # # cx, cy = w // 2, h // 2
            
            final_indices = []
            for l in valid_labels:
                idx = np.where(labels == l)[0]
                pts = np.asarray(pcd.select_by_index(idx).points)
                center = pts.mean(axis=0) #mins, maxs = pts.min(axis=0) #, pts.max(axis=0)
                dist   = np.linalg.norm(center - main_center)
                
                if dist <= MAX_DIST:
                    final_indices.extend(idx)
                    print(f" label={l} pts={len(idx):,} dist={dist:.0f}mm — mantido")
                else:
                    print(f" label={l} pts={len(idx):,} dist={dist:.0f}mm — removido")
                
            #     # center_x = (mins[0] + maxs[0]) / 2
            #     # center_y = (mins[1] + maxs[1]) / 2
            #     # center_z = (mins[2] + maxs[2]) / 2
                
            #     #  # Pegar o cluster principal (maior volume)
            #     # main_label = max(valid_labels, key=lambda l: cluster_volume(pcd, l, labels))
            #     # main_pts = np.asarray(pcd.select_by_index(np.where(labels == main_label)[0]).points)
            #     # main_center = main_pts.mean(axis=0)
                
            #     # center = pts.mean(axis=0)
            #     if np.linalg.norm(center - main_center) < 400: #0.35:
            #         final_indices.extend(idx)
            
            # #body_label = max([l for l in np.unique(labels) if l != -1],key=lambda l: np.ptp(np.asarray(pcd.select_by_index(np.where(labels == l)[0]).points)[:, 1]))
            # # body_label = max([l for l in np.unique(labels) if l != -1],key=lambda l: cluster_volume(pcd, l, labels))
            pcd_clean = pcd.select_by_index(final_indices) # pcd_body = pcd.select_by_index(np.where(labels == body_label)[0])
            print(f"   Pontos finais:    {len(pcd_clean.points):,}")
            
            out_path = path.with_name(path.stem + "_cropped.ply")
            o3d.io.write_point_cloud(str(out_path), pcd_clean)
            
            # max_label = labels.max()
            # print(f"point cloud has {max_label + 1} clusters")
            # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            # colors[labels < 0] = 0
            # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            # o3d.visualization.draw_geometries([pcd],
            #                                 zoom=0.455,
            #                                 front=[-0.4999, -0.1659, -0.8499],
            #                                 lookat=[2.1813, 2.0619, 2.0999],
            #                                 up=[0.1204, -0.9852, 0.1215])