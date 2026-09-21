import sys
import os
import pickle
import time
import warnings

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from src.arm_slices_adaptive import adaptive_arm_slices
from scipy.spatial import ConvexHull, KDTree, QhullError
from sklearn.cluster import DBSCAN

use_numba = False
#testes de redução do tempo de fatiamento + clustering descomente abaixo
if use_numba:
    from numba import njit, prange

EPS = 1e-8
global vertical_axis

"""
Dicionário com os landmarks a serem identificados na malha. 
    chave: nome abreviado, valor: tupla de dois elementos sendo o primeiro a cor no qual o arco será desenhado 
    e o segundo elemento o nome em extenso para geração do retorno final que será exibido na tabela de medidas.
"""
LANDMARK_STYLE = {
#     # Pescoço
     'pescoco':             ([1.00, 0.00, 0.00], 'Pescoço'),
#     # Tronco
#     # Ombros
     'omb_sup':             ([1.00, 0.95, 0.20], 'Ombros Superior'),
     'omb_med':             ([1.00, 0.90, 0.10], 'Ombros Médio'),
     'omb_inf':             ([1.00, 0.85, 0.00], 'Ombros Inferior'),     
#     # Busto 
     'bus_sup':             ([1.00, 0.80, 0.00], 'Busto Superior'),
     'bus_med':             ([1.00, 0.75, 0.00], 'Busto Médio'),
     'bus_inf':             ([1.00, 0.70, 0.00], 'Busto Inferior'),
#     #Cintura 
     'cin_sup':             ([0.95, 0.65, 0.00], 'Cintura Superior'),
     'cin_med':             ([0.90, 0.60, 0.00], 'Cintura Médio'),
     'cin_inf':             ([0.85, 0.55, 0.00], 'Cintura Inferior'),
#     #Quadril 
     'quadl_sup':           ([0.80, 0.50, 0.00], 'Quadril Superior'),
     'quadl_med':           ([0.75, 0.45, 0.00], 'Quadril Médio'),
     'quadl_inf':           ([0.70, 0.40, 0.00], 'Quadril Inferior'),
#     # Braço esquerdo
     'bic_longo_esq':       ([0.55, 0.00, 0.60], 'Bíceps Longo Esq.'),
     'bic_curto_esq':       ([0.70, 0.00, 0.75], 'Bíceps Curto Esq.'),
     'braq_esq':            ([0.85, 0.00, 0.85], 'Braquial Esq.'),
     'braquiorr_esq':       ([0.95, 0.10, 0.85], 'Braquiorradial Esq.'),
     'cot_esq':             ([1.00, 0.25, 0.85], 'Cotovelo Esq.'),
     'ant_esq':             ([1.00, 0.45, 0.85], 'Antebraço Esq.'),
     'puls_esq':            ([1.00, 0.65, 0.90], 'Pulso Esq.'),
#     # Braço direito
     'bic_longo_dir':       ([0.00, 0.20, 0.60], 'Bíceps Longo Dir.'),
     'bic_curto_dir':       ([0.00, 0.35, 0.75], 'Bíceps Curto Dir.'),
     'braq_dir':            ([0.00, 0.50, 0.90], 'Braquial Dir.'),
     'braquiorr_dir':       ([0.00, 0.65, 1.00], 'Braquiorradial Dir.'),
     'cot_dir':             ([0.00, 0.80, 1.00], 'Cotovelo Dir.'),
     'ant_dir':             ([0.20, 0.90, 1.00], 'Antebraço Dir.'),
     'puls_dir':            ([0.50, 0.95, 1.00], 'Pulso Dir.'),
#     # Perna esquerda
     'pect_esq':            ([0.55, 0.00, 0.00], 'Pectíneo Esq.'),
     'pect_quad_esq':       ([0.65, 0.00, 0.00], 'Pectíneo-Quadríceps Esq.'),
     'quad_esq':            ([0.75, 0.00, 0.00], 'Quadríceps Esq.'),
     'quad_coxa_esq':       ([0.85, 0.00, 0.00], 'Quadríceps-Coxa Esq.'),
     'coxa_esq':            ([1.00, 0.00, 0.00], 'Coxa Esq.'),
     'coxa_joelho_esq':     ([1.00, 0.15, 0.00], 'Coxa-Joelho Esq.'),
     'joelho_esq':          ([1.00, 0.25, 0.00], 'Joelho Esq.'),
     'gast_esq':            ([1.00, 0.35, 0.00], 'Gastrocnêmio Esq.'),
     'soleo_esq':           ([1.00, 0.45, 0.00], 'Sóleo Esq.'),
     'torn_esq':            ([1.00, 0.55, 0.00], 'Tornozelo Esq.'),
#     # Perna direita
     'pect_dir':            ([0.00, 0.40, 0.00], 'Pectíneo Dir.'),
     'pect_quad_dir':       ([0.00, 0.50, 0.00], 'Pectíneo-Quadríceps Dir.'),
     'quad_dir':            ([0.00, 0.60, 0.00], 'Quadríceps Dir.'),
     'quad_coxa_dir':       ([0.00, 0.70, 0.05], 'Quadríceps-Coxa Dir.'),
     'coxa_dir':            ([0.00, 0.80, 0.10], 'Coxa Dir.'),
     'coxa_joelho_dir':     ([0.00, 0.90, 0.15], 'Coxa-Joelho Dir.'),
     'joelho_dir':          ([0.00, 1.00, 0.20], 'Joelho Dir.'),
     'gast_dir':            ([0.15, 1.00, 0.30], 'Gastrocnêmio Dir.'),
     'soleo_dir':           ([0.30, 1.00, 0.45], 'Sóleo Dir.'),
     'torn_dir':            ([0.45, 1.00, 0.60], 'Tornozelo Dir.')
}


def plot_clusters(clusters, hull_pts_list, landmarks, full_name, slice_indices=None):
    """
    slice_indices: lista de índices a visualizar, ex: [0,1,5]
                   se None, plota todas as fatias, retorna um objeto matplotlib pyplot.
    """
    indices = slice_indices if slice_indices else range(len(clusters))
    
    # mapa idx -> (name, hull)
    idx_to_landmark = {}
    for name, idx in landmarks.items():
        hp = hull_pts_list.get(name)
        if hp is not None:
            idx_to_landmark[idx] = (name, hp)

    for k in indices:
        clstrs = clusters[k]
        if not clstrs:
            continue

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for c_idx, c in enumerate(clstrs):
            ax.scatter(c[:, 0], c[:, 1], s=1,
                       color=colors[c_idx % len(colors)],
                       label=f"cluster {c_idx}")

        # hull do landmark nessa fatia, se existir
        if k in idx_to_landmark:
            name, hp = idx_to_landmark[k]
            closed = np.vstack([hp, hp[0]])
            ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=2,
                    label=f"hull {name}")

        ax.set_title(f"Fatia {k:03d} — {len(clstrs)} cluster(s)")
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"./{full_name}/{full_name}_fatia_{k:03d}.png", dpi=150)
        plt.close()
        print(f"salvo: fatia_{k:03d}.png")


def calculate_height(mesh):
    """Função a ser usada posteriormente para desenho e cálculo da altura da malha em metros"""
    verts = np.asarray(mesh.vertices)

    z_min = verts[:, 2].min()
    z_max = verts[:, 2].max()
    altura = z_max - z_min
    
    return altura


def axis_name(axis: int) -> str:
    """Apresenta as alturas máximas e mínimas de x, y e z"""
    return {0: "X", 1: "Y", 2: "Z"}[axis]


# def _compute_db_scan(pts_2d, eps_frac=0.04, min_samples=5):
#     """O agrupamento baseado em densidade acabou separando membros dos lados opostos como juntos no mesmo lado: fatias da perna esquerda
#     e direta ficavam juntas além de serem separadas"""
#     bbox_diag = np.linalg.norm(pts_2d.max(axis=0) - pts_2d.min(axis=0))
#     eps = bbox_diag * eps_frac

#     labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_2d)
    
#     end = time.time() - begin
#     return labels, eps


def _classify_clusters(clusters):
    """Classificação dos clusters da malha: Corpo humano possui no máximo 3 (braços e tronco) daí 2 (pernas) e 1 que pode ser agrupamentos
    da cabeça até antes da abertura dos braços e posteriormente no fim dos braços até a abertura das pernas que é a cintura."""
    if not clusters:
        return {}

    if len(clusters) == 1:
        print("Somente um cluster")
        return {'center': clusters[0]}

    if len(clusters) == 2:
        print("dois clusters")
        sorted_c = sorted(clusters, key=lambda c: c.mean(axis=0)[0])
        return {
            'left':  sorted_c[0],
            'right': sorted_c[1],
        }

    # 3+ clusters: ordena por X e pega o do meio como central
    # sorted_c = sorted(clusters, key=lambda c: c.mean(axis=0)[0])

    if len(clusters) == 3:
        print("3 clusters")
        sorted_by_size = sorted(clusters, key=len, reverse=True)
        trunk = sorted_by_size[0]
        arms  = sorted(sorted_by_size[1:], key=lambda c: c.mean(axis=0)[0])
        return {'left': arms[0], 'center': trunk, 'right': arms[1]}


def make_ring_lineset(hull_pts_2d: np.ndarray, height: float, axis: int, color) -> o3d.geometry.LineSet:
    """Desenvolvimento de um conjunto de linhas coloridas obtidos pelos contornos das fatias nas alturas encontradas."""
    hp = np.asarray(hull_pts_2d, dtype=float)
    if hp.ndim != 2 or hp.shape[1] not in (2, 3) or len(hp) < 3:
        raise ValueError('Esperado contorno Nx2 ou Nx3 com pelo menos 3 pontos.')
    if hp.shape[1] == 3:
        # Ja esta na malha: nao fixar Z e nao reordenar pela projecao XY.
        pts_3d = hp.copy()
    else:
        centroid = hp.mean(axis=0)
        angles = np.arctan2(hp[:, 1]-centroid[1], hp[:, 0]-centroid[0])
        hp = hp[np.argsort(angles)]
        axes_2d = [i for i in range(3) if i != axis]
        pts_3d = np.empty((len(hp), 3))
        pts_3d[:, axes_2d] = hp
        pts_3d[:, axis] = height
    ids = np.arange(len(pts_3d))
    lines = np.column_stack((ids, np.roll(ids, -1)))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_3d)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


# def _get_central_cluster(pts_2d, labels):
#     valid = [l for l in set(labels) if l >= 0]

#     #Get central cluster
#     if len(valid) > 1:
#         global_center = np.median(pts_2d, axis=0)
#         best_label = min(valid, key=lambda l: np.linalg.norm(pts_2d[labels == l].mean(axis=0) - global_center))  
#         pts_2d = pts_2d[labels == best_label]

#     if pts_2d.shape[0] < 3: return None, 0.0
    
#     return pts_2d


def _get_convex_hull(pts_2d):
    """Obtenção de fechos convexos para um conjunto de pontos no(s) agrupamento(s) de cada fatia."""
    hull = ConvexHull(pts_2d)
    hp = pts_2d[hull.vertices]
    
    return hp


def _main_cluster_perim(cl):
    """Retorna o perímetro do cluster principal dependendo do número de clusters."""
    if not cl:
        return 0.0

    classified = _classify_clusters(cl)

    center = classified.get('center')
    print("Center is not none")
    
    if center is None:
        print("Center is None")
        # 2 clusters — não há center, pega o maior
        center = max(cl, key=len)

    _, perim = _hull_perimeter_and_pts(center)

    print(f"perim:{perim}")
    return perim or 0.0


def _is_inside(pts_inner, pts_outer, margin=0.05):
    """Retorna True se o centróide de pts_inner está dentro do bbox de pts_outer."""
    cx, cy = pts_inner.mean(axis=0)
    x_min, y_min = pts_outer.min(axis=0)
    x_max, y_max = pts_outer.max(axis=0)
    pad_x = (x_max - x_min) * margin
    pad_y = (y_max - y_min) * margin
    
    return (x_min - pad_x < cx < x_max + pad_x and y_min - pad_y < cy < y_max + pad_y)


def _slice_mesh_oriented(vertices, triangles, center, normal):
    """
    Fatia a malha num plano arbitrário definido por centro + normal.
    Permite fatiar perpendicularmente ao eixo do braço inclinado.
    """
    normal = normal / np.linalg.norm(normal)
    
    d_all = (vertices - center) @ normal  # distância de cada vértice ao plano

    # pré-filtra triângulos: só os que têm vértices em ambos os lados do plano
    d_tri = d_all[triangles]  # (N, 3)
    d0, d1, d2 = d_tri[:, 0], d_tri[:, 1], d_tri[:, 2]
    crosses = (
        ((d0 > 0) != (d1 > 0)) |
        ((d1 > 0) != (d2 > 0)) |
        ((d0 > 0) != (d2 > 0))
    )
    tris_filtered = triangles[crosses]  # só os triângulos relevantes

    if len(tris_filtered) == 0:
        return [], np.empty((0, 2))
    
    segments = []
    for tri in tris_filtered:
        v = vertices[tri]
        dd = d_all[tri]
        inter = []
        for i in range(3):
            j = (i + 1) % 3
            di, dj = dd[i], dd[j]
            if di == 0.0:
                inter.append(v[i])
            elif dj != 0.0 and di * dj < 0.0:
                t = di / (di - dj)
                inter.append(v[i] + t * (v[j] - v[i]))
        unique = []
        for p in inter:
            if not unique or np.linalg.norm(p - unique[-1]) > 1e-9:
                unique.append(p)
        if len(unique) >= 2:
            segments.append((unique[0], unique[1]))
    
    if not segments:
        return segments, np.empty((0, 2))
    
    # projeta pontos no plano local (2D)
    # base ortonormal do plano
    u = np.array([1, 0, 0], dtype=float)
    if abs(normal @ u) > 0.9:
        u = np.array([0, 1, 0], dtype=float)
    u = u - (u @ normal) * normal
    u /= np.linalg.norm(u)
    v_ax = np.cross(normal, u)
    
    pts_3d = np.array([p for seg in segments for p in seg])
    pts_2d = np.column_stack([pts_3d @ u, pts_3d @ v_ax])
    
    return segments, pts_2d


# if use_numba:
#     @njit(parallel=True)
#     def _slice_mesh_numba(vertices, triangles, height, axis):
#         """
#         Versão JIT-compilada do slice_mesh.
#         Retorna array de segmentos (N, 2, 3). Anteriormente foi implementada como tentativa de utilizar o numba para reduzir o tempo
#         de execução dos fatiamentos da malha, contudo, sua implementação foi descontinuada pois estava gerando diferentes pontos nas fatias
#         e interferindo na detecção dos landmarks e cálculo de medidas finais, além de ter tido um tempo de execução quase igual
#         ao slice_mesh convencional.
#         """
#         tri_min = vertices[triangles][:, :, vertical_axis].min(axis=1)
#         tri_max = vertices[triangles][:, :, vertical_axis].max(axis=1)

#         # pré-filtra por bounding box
#         mask = (tri_min <= height) & (tri_max >= height)
#         valid_tris = triangles[mask]
        
#         n = len(valid_tris)
#         # pré-aloca máximo possível
#         segs = np.zeros((n, 2, 3), dtype=np.float64)
#         count = 0
        
#         for idx in prange(n):
#             tri = valid_tris[idx]
#             v = vertices[tri]  # (3, 3)
#             d = v[:, axis] - height
            
#             inter = np.zeros((4, 3), dtype=np.float64)
#             n_inter = 0
            
#             for i in range(3):
#                 j = (i + 1) % 3
#                 di, dj = d[i], d[j]
#                 if di == 0.0:
#                     inter[n_inter] = v[i]
#                     n_inter += 1
#                 elif dj != 0.0 and di * dj < 0.0:
#                     t = di / (di - dj)
#                     inter[n_inter] = v[i] + t * (v[j] - v[i])
#                     n_inter += 1
            
#             if n_inter >= 2:
#                 segs[count, 0] = inter[0]
#                 segs[count, 1] = inter[1]
#                 count += 1
        
#         return segs[:count]

    
def _slice_mesh(vertices: np.ndarray, triangles: np.ndarray, height: float, axis: int):    
    """
        Responsável pelo fatiamento da malha por cada altura podendo ser com o numba (descontinuado) ou somente com o numpy através da
        paralelização de arrays para ganho de tempo de execução.
    """
    axes_2d = [i for i in range(3) if i != axis]

    # if use_numba:
    #     segs_arr = _slice_mesh_numba(vertices, triangles, height, axis)

    #     if len(segs_arr) == 0:
    #         return [], np.empty((0, 2))
        
    #     segments = [(segs_arr[i, 0], segs_arr[i, 1]) for i in range(len(segs_arr))]

    #     pts_3d   = segs_arr.reshape(-1, 3) # pts_3d = np.array([p for seg in segments for p in seg])
    # else:
    #####Somente com numpy:######
    v = vertices[triangles]          # (N, 3, 3)
    d = v[:, :, axis] - height       # (N, 3)
    
    # máscara de triângulos que cruzam o plano
    d0, d1, d2 = d[:, 0], d[:, 1], d[:, 2]
    crosses = (
        ((d0 > 0) != (d1 > 0)) |
        ((d1 > 0) != (d2 > 0)) |
        ((d0 > 0) != (d2 > 0))
    )
    v = v[crosses]
    d = d[crosses]

    if len(v) == 0:
        return [], np.empty((0, 2))

    segments = []
    #Todo o cálculo abaixo é realizado pelo Numba paralelamente!
    for tri_v, tri_d in zip(v, d):
        inter = []
        for i in range(3):
            j = (i + 1) % 3
            di, dj = tri_d[i], tri_d[j]
            if di == 0.0:
                inter.append(tri_v[i])
    #         # elif dj == 0.0:
    #         #     pass
    #         # elif di * dj < 0.0:
    #         #     t = di / (di - dj)
    #         #     inter.append(tri_v[i] + t * (tri_v[j] - tri_v[i]))
            elif dj != 0.0 and di * dj < 0.0:
                t = di / (di - dj)
                inter.append(tri_v[i] + t * (tri_v[j] - tri_v[i]))

        unique = []
        for p in inter:
            if not unique or np.linalg.norm(p - unique[-1]) > 1e-9:
                unique.append(p)

        if len(unique) >= 2:
            segments.append((unique[0], unique[1]))
    
    if not segments:
        return [], np.empty((0, 2))

    pts_3d = np.array([p for seg in segments for p in seg])

    pts_2d = pts_3d[:, axes_2d]

    # end = time.time() - begin
    # print(f"_slice_mesh execution time:{end}")
    return segments, pts_2d


def _detect_member_drift(vertices, triangles, h_start, h_end, axis, n_probe_slices=8, side='left'):
    """
    Detecta a direção de deslocamento real de um membro entre h_start e h_end, rastreando o centróide do cluster principal fatia a 
    fatia sem assumir lado (esquerda/direita) previamente.
    
    Retorna: lista de centros 3D e o vetor de deslocamento total.
    """
    axes_2d = [i for i in range(3) if i != axis]
    centers_3d = []

    for h in np.linspace(h_start, h_end, n_probe_slices):
        segs, _ = _slice_mesh(vertices, triangles, h, axis)
        cl = _extract_slice_clusters(segs, 0.01) if len(segs) >= 2 else []
        #if not cl:
        if len(cl) != 3:
            centers_3d.append(None)
            continue

        classified = _classify_clusters(cl)
        c = classified.get(side)  # 'left' ou 'right' — o braço, não o tronco
        if c is None:
            centers_3d.append(None)
            continue
        #c = max(cl, key=len)  # cluster principal dessa fatia
        centroid_2d = c.mean(axis=0)
        center_3d = np.zeros(3)
        center_3d[axes_2d[0]] = centroid_2d[0]
        center_3d[axes_2d[1]] = centroid_2d[1]
        center_3d[axis]       = h
        centers_3d.append(center_3d)

    valid_centers = [c for c in centers_3d if c is not None]
    if len(valid_centers) < 2:
        return None, None

    valid_centers = np.array(valid_centers)

    # vetor de deslocamento total: do primeiro ao último centro válido
    drift_vector = valid_centers[-1] - valid_centers[0]

    return valid_centers, drift_vector


def _member_needs_tilted_slice(drift_vector, axis, threshold_ratio=0.15):
    """
    Decide se o deslocamento lateral (XY) é grande o suficiente em relação ao deslocamento vertical (eixo principal) 
    para justificar corte transversal.
    """
    if drift_vector is None:
        return False

    vertical_disp   = abs(drift_vector[axis])
    lateral_disp     = np.linalg.norm(np.delete(drift_vector, axis))

    if vertical_disp == 0:
        return False

    ratio = lateral_disp / vertical_disp
    return ratio > threshold_ratio


# def _slice_member_adaptive(vertices, triangles, h_start, h_end, axis, n_probe_slices=8, threshold_ratio=0.15, side='left'):
#     """
#     Descontinuado: Detecta automaticamente a direção de deslocamento do membro e decide entre corte horizontal normal ou
#     corte transversal via PCA — sem assumir de antemão qual lado o membro se inclina.
#     """
#     print(f"axis:{axis}")
#     centers_3d, drift_vector = _detect_member_drift(vertices, triangles, h_start, h_end, axis, n_probe_slices, side)

#     if centers_3d is None:
#         return 'none', None, None

#     if not _member_needs_tilted_slice(drift_vector, axis, threshold_ratio):
#         print(f"[{side}] Membro reto — drift={drift_vector}")
#         return 'horizontal', None, None

#     # membro inclinado — calcula eixo real via PCA nos centros já coletados
#     center_mean = centers_3d.mean(axis=0)
#     centered    = centers_3d - center_mean
#     _, _, Vt = np.linalg.svd(centered, full_matrices=False)
#     tilt_axis = Vt[0]

#     # garante que aponta na mesma direção geral do deslocamento vertical real
#     if np.dot(tilt_axis, drift_vector) < 0:
#         tilt_axis = -tilt_axis

#     print(f"Membro inclinado detectado — drift={drift_vector}  tilt_axis={tilt_axis}")
#     return 'tilted', center_mean, tilt_axis


# def _filter_trunk_mad(pts_2d: np.ndarray, n_mad: float = 3.5, max_iter: int = 10) -> np.ndarray:
#     """Descontinuado: Estava interferindo gerando fechos convexos e perímetros errados"""

#     begin = time.time()
    
#     pts = pts_2d.copy()
#     for _ in range(max_iter):
#         center    = np.median(pts, axis=0)
#         dists     = np.linalg.norm(pts - center, axis=1)
#         median_d  = np.median(dists)
#         mad       = np.median(np.abs(dists - median_d))
#         threshold = median_d + n_mad * mad * 1.4826   # 1.4826 = fator de consistência gaussiana
#         mask = dists <= threshold
#         if mask.sum() < 3 or mask.sum() == len(pts):
#             break                                       # convergiu
#         pts = pts[mask]
        
#     end = time.time() - begin
#     print(f"_filter_trunk_mad finishing time:{end}")
#     return pts


def _hull_perimeter_and_pts(pts_2d: np.ndarray, eps_frac=0.04, min_samples = 5): #Fatiando a malha do topo a base
    """Obtenção dos fechos convexos de todos os agrupamentos encontrados pelo método _get_convex_hull() e cálculos dos perímetros desses
    fechos convexos.
    
    """
    # begin = time.time()
    
    if pts_2d.shape[0] < 3:
        return None, 0.0

    # pts_2d = _filter_trunk_mad(
    #     pts_2d,
    #     n_mad=3.5
    # )

    # if pts_2d_clean.shape[0] < 3:
    #     pts_2d_clean = pts_2d
    # #     return None, 0.0

    # labels, eps = _compute_db_scan(pts_2d, eps_frac, min_samples)
    # pts_2d = _get_central_cluster(pts_2d, labels)
    
    try:
        hp = _get_convex_hull(pts_2d)
        perim = 0.0
        for i in range(len(hp)):
            perim += np.linalg.norm(hp[(i + 1) % len(hp)] - hp[i])
        # end = time.time() - begin
        # print(f"_hull_perimeter_and_pts execution time:{end}")
        return hp, perim
    except Exception:
        # end = time.time() - begin
        # print(f"_hull_perimeter_and_pts execution time:{end}")
        return None, 0.0


def _extract_slice_clusters(segments, tol_frac=0.01, min_pts=4):
    """
    Obtém cada fatia e através do segmentos desses pontos divide a fatia em uma quantidade de agrupamentos.  
    """
    if not segments:
        return []

    n = len(segments)
    parent = list(range(n))

    endpoints = [(np.array(s[0][:2]), np.array(s[1][:2])) for s in segments]
    all_pts   = np.array([p for ep in endpoints for p in ep])
    bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    tol       = bbox_diag * tol_frac

    #Ao invés de usar Grid usa-se o KDTree para a busca dos vizinhos com tempo O(n log n)
    tree = KDTree(all_pts)
    pairs = tree.query_pairs(tol)  # todos os pares dentro de tol
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    # cada ponto pertence ao segmento i = pt_idx // 2
    for pi, pj in pairs:
        si, sj = pi // 2, pj // 2
        if si != sj:
            union(si, sj)

    from collections import defaultdict
    comp = defaultdict(list)
    for i, seg in enumerate(segments):
        comp[find(i)].append(seg)

    candidates = []
    for segs in comp.values():
        pts = np.array([p[:2] for s in segs for p in s])
        if len(pts) >= min_pts:
            bbox_area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
            candidates.append((bbox_area, pts))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    clusters = []
    for i, (area, pts) in enumerate(candidates):
        inside = any(_is_inside(pts, candidates[j][1]) for j in range(i))
        if not inside:
            clusters.append(pts)

    if clusters:
        centers = np.array([c.mean(axis=0) for c in clusters])
        sort_ax = int(np.argmax(np.ptp(centers, axis=0)))
        clusters.sort(key=lambda c: c.mean(axis=0)[sort_ax])
    
    return clusters


def _arm_contour_center(points):
    """Centroide de area do hull: evita vies por densidade da triangulacao."""
    hp, _ = _hull_perimeter_and_pts(points)
    if hp is None:
        return np.mean(points, axis=0)
    origin = hp.mean(axis=0)
    p = hp - origin
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]
    twice_area = cross.sum()
    if abs(twice_area) <= np.finfo(float).eps * max(np.sum(p*p), np.finfo(float).tiny):
        return origin
    return origin + ((p+q)*cross[:, None]).sum(axis=0)/(3*twice_area)


def _arm_final_planes(clusters, heights, window=5, threshold_ratio=0.15):
    """Estima planos em Z a partir dos clusters horizontais ja calculados.

    Apenas sequencias contiguas com exatamente 3 clusters contribuem.
    Centro = media dos pontos do cluster, como no codigo original.
    """
    heights = np.asarray(heights, dtype=float)
    if len(heights) != len(clusters):
        raise ValueError('clusters e heights devem ter o mesmo comprimento.')
    if window < 3 or window % 2 != 1 or threshold_ratio <= 0:
        raise ValueError('Use janela impar >= 3 e threshold_ratio positivo.')
    diff = np.diff(heights)
    if not (np.all(diff > 0) or np.all(diff < 0)):
        raise ValueError('heights deve ser estritamente crescente ou decrescente.')
    n = len(heights)
    centers = {s: np.full((n, 3), np.nan) for s in ('left', 'right')}
    selected = {s: [None] * n for s in centers}
    for i, cl in enumerate(clusters):
        if len(cl) != 3:
            continue
        classified = _classify_clusters(cl) or {}
        for side in centers:
            c = classified.get(side)
            if c is None or len(c) < 3 or not np.isfinite(c).all():
                continue
            selected[side][i] = c
            centers[side][i] = [*_arm_contour_center(c), heights[i]]

    planes = {s: [None] * n for s in centers}
    for side, cc in centers.items():
        tilted = False
        for i in range(n):
            if selected[side][i] is None:
                tilted = False
                continue
            lo, hi = i, i + 1
            while lo > max(0, i - window//2) and selected[side][lo-1] is not None:
                lo -= 1
            while hi < min(n, i + window//2 + 1) and selected[side][hi] is not None:
                hi += 1
            if hi - lo < 2:
                tilted = False
                continue
            dh = heights[lo:hi] - heights[lo:hi].mean()
            local = cc[lo:hi]
            tangent = (dh[:, None] * (local - local.mean(axis=0))).sum(axis=0) / (dh @ dh)
            tangent[2] = 1.0
            ratio = float(np.linalg.norm(tangent[:2]))
            threshold = threshold_ratio * (2/3) if tilted else threshold_ratio
            tilted = ratio > threshold
            normal = tangent / np.linalg.norm(tangent) if tilted else np.array([0., 0., 1.])
            planes[side][i] = {
                'origin': cc[i].copy(), 'normal': normal, 'tilted': tilted,
                'ratio': ratio, 'cluster': selected[side][i],
            }
    return planes


def _arm_final_hull(verts, tris, plane):
    """Retorna (hull Nx2 ou Nx3, perimetro), refatiando somente se inclinado."""
    """Retorna (hull Nx2 ou Nx3, perimetro), refatiando somente se inclinado."""
    if not plane['tilted']:
        return _hull_perimeter_and_pts(plane['cluster'])

    origin, normal = plane['origin'], plane['normal']
    segs, _ = _slice_mesh_oriented(verts, tris, origin, normal)
    if len(segs) < 2:
        return None, 0.0
    # Base ortonormal: distancias em 2D sao as distancias reais do plano.
    ref = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    segs = np.asarray(segs, dtype=float)
    rel = segs - origin
    xy = np.stack((rel @ u, rel @ v), axis=-1)
    # Seu agrupador original agora opera nas coordenadas LOCAIS do plano.
    cl = _extract_slice_clusters([(a, b) for a, b in xy], 0.02)
    if not cl:
        return None, 0.0
    # O centro sondado tem coordenadas locais (0, 0). Nao classificar
    # esquerda/direita nesta base, que gira com a normal.
    c = min(cl, key=lambda c: np.linalg.norm(_arm_contour_center(c)))
    radius = np.max(np.linalg.norm(plane['cluster'] - origin[:2], axis=1))
    if np.linalg.norm(_arm_contour_center(c)) > radius:
        return None, 0.0
    hp, perimeter = _hull_perimeter_and_pts(c)
    if hp is None:
        return None, 0.0
    hp3d = origin + hp[:, :1]*u + hp[:, 1:]*v
    return hp3d, perimeter


def measure_arm_side_single_plane(verts, tris, h_start, h_end, axis, side,
                                 n_slices=50, n_probe=10, threshold_ratio=0.15,
                                 *, seed_center=None, lateral_axis=None):
    """Nome e retorno antigos preservados; agora a normal varia localmente.

    n_probe continua aceito por compatibilidade. A sondagem agora usa
    n_slices alturas e janela de 5 centros, em vez de uma unica direcao.
    Retorna (h_measure_end, perims).
    Cada perim = (height, perimeter, hp, plane_center, plane_normal).
    Horizontal: hp nas coordenadas globais dos dois eixos restantes;
                plane_center e plane_normal sao None, como antes.
    Inclinado: hp local, relativo a plane_center, na base reconstruida
               por arm_perim_points_3d. Use esse helper para desenhar.
    """
    slices = adaptive_arm_slices(
        verts, tris, h_start, h_end, axis, side,
        slice_horizontal=_slice_mesh,
        slice_oriented=_slice_mesh_oriented,
        horizontal_clusterer=lambda segs: _extract_slice_clusters(segs, 0.02),
        n_slices=n_slices,
        window=5,
        threshold_on=threshold_ratio,
        threshold_off=threshold_ratio * (2.0 / 3.0),
        seed_center=seed_center,
        lateral_axis=lateral_axis,
    )
    axes_2d = [a for a in range(3) if a != axis]
    perims = []
    consecutive_anomaly = 0
    h_hands_begin = None
    invalid_count = 0
    first_invalid_reason = None

    # Mantem a heuristica original: 10 amostras de referencia e 3
    # anomalias consecutivas fora de 0.5x..1.5x a media dos aceitos.
    # Isso e uma heuristica de parada, nao uma deteccao anatomica validada.
    for s in slices:
        if not s['valid']:
            consecutive_anomaly = 0
            invalid_count += 1
            first_invalid_reason = first_invalid_reason or s['reason']
            continue
        h, p = s['height'], s['perimeter_hull']
        if len(perims) >= 10:
            mean_p = np.mean([r[1] for r in perims])
            if p < 0.5 * mean_p or p > 1.5 * mean_p:
                consecutive_anomaly += 1
                if consecutive_anomaly >= 3:
                    h_hands_begin = h
                    break
                continue
        consecutive_anomaly = 0
        if s['tilted']:
            hp = s['hull_2d']
            center, normal = s['origin'], s['normal']
        else:
            hp = s['hull_3d'][:, axes_2d]
            center, normal = None, None
        perims.append((h, p, hp, center, normal))

    if invalid_count:
        warnings.warn(f'[{side}] {invalid_count} fatias invalidas: {first_invalid_reason}',
                      RuntimeWarning, stacklevel=2)
    h_measure_end = h_hands_begin if h_hands_begin is not None else (
        perims[-1][0] if perims else h_end
    )
    return h_measure_end, perims


def detect_arms_heights(verts, tris, axis, h_arms_begin, h_arms_end, *, return_slices=False, seed_left=None, seed_right=None):
    """Por padrao retorna os dois dicionarios de alturas,

        return_slices=True acrescenta dois dicionarios nome -> tupla perim.
        Seleciona 7 cortes aceitos existentes, aproximadamente equiespacados
        por indice. As alturas correspondem exatamente a esses cortes.
        Mantem os nomes solicitados; nao identifica marcos anatomicos reais.
        """
    _, perims_esq = measure_arm_side_single_plane(
        verts, tris, h_arms_begin, h_arms_end, axis, 'left', seed_center=seed_left)
    _, perims_dir = measure_arm_side_single_plane(
        verts, tris, h_arms_begin, h_arms_end, axis, 'right', seed_center=seed_right)

    names_esq = ['bic_longo_esq', 'bic_curto_esq', 'braq_esq', 'braquiorr_esq',
                 'cot_esq', 'ant_esq', 'puls_esq']
    names_dir = ['bic_longo_dir', 'bic_curto_dir', 'braq_dir', 'braquiorr_dir',
                 'cot_dir', 'ant_dir', 'puls_dir']

    def select(perims, names):
        if len(perims) < len(names):
            raise ValueError(f'Somente {len(perims)} cortes aceitos; precisa de 7.')
        ids = np.rint(np.linspace(0, len(perims) - 1, len(names))).astype(int)
        cuts = {name: perims[i] for name, i in zip(names, ids)}
        heights = {name: cut[0] for name, cut in cuts.items()}
        return heights, cuts

    landmarks_esq, cuts_esq = select(perims_esq, names_esq)
    landmarks_dir, cuts_dir = select(perims_dir, names_dir)
    if return_slices:
        return landmarks_esq, landmarks_dir, cuts_esq, cuts_dir
    return landmarks_esq, landmarks_dir


def detect_preliminary_heights(h_max, h_mid, verts, tris, axis):
    """Detecção de landmarks iniciais: Varre toda a malha desde o topo (cabeça) até a base (primeiramente procura pela primeira fatia com 
    apenas um cluster: até que somente haja um agrupamento para evitar erros ignorando as demais), durante a varredura calculando os 
    perímetros das anteriores e próximas até encontrar abrupto crescimento (fim do pescoço e início do trapézio) até uma altura em que 
    hajam 3 fatias (início dos braços)."""

    h_head_top      = None
    h_shoulders_end = None
    h_arms_begin    = None

    best_neck       = {'h': None, 'perim': float('inf')}
    best_head       = {'h': None, 'perim': 0.0}
    best_shoulder   = {'h': None, 'perim': 0.0}

    topo            = False
    perims_upper    = []

    for h_s in np.linspace(h_max, h_mid, 50):
        print(f"h_s:{h_s}")
        segs, pts_2d = _slice_mesh(verts, tris, h_s, axis)        
        cl = _extract_slice_clusters(segs, 0.02) if len(segs) >= 2 else []

        print(f"len(cl):{len(cl)}")
        # top_clusters.append(cl)
        if len(cl) == 1:
            if not topo:
                h_head_top = h_s  # primeiro com 1 cluster = topo da cabeça
                topo = True
            h_shoulders_end = h_s      # atualiza sempre — último com 1 cluster

            # calcula perímetro do único cluster        
            perim = _main_cluster_perim(cl)
            perims_upper.append((h_s, perim))

        elif len(cl) > 2:
            # saiu de 1 cluster — marca início dos braços
            h_arms_begin = h_s
            break

        if perims_upper:
            hs     = [p[0] for p in perims_upper]
            perims = [p[1] for p in perims_upper]
            
            if len(perims_upper) > 1:
                hs_valid    = hs[1:]
                perims_valid = perims[1:]
            else:
                hs_valid    = hs
                perims_valid = perims
        
            if not hs_valid:
                print("[AVISO] Região superior sem pontos válidos")
            else:
                # cabeça: primeiro pico local
                i_head = 0
                for i in range(1, len(perims_valid)):
                    if perims_valid[i] > perims_valid[i_head]:
                        i_head = i
                    if perims_valid[i] < perims_valid[i - 1] and i > i_head:
                        break
            best_head = {'h': hs_valid[i_head], 'perim': perims_valid[i_head]}
        
            # pescoço: vale local após o pico
            i_neck = i_head
            for i in range(i_head, len(perims_valid)):
                if perims_valid[i] < perims_valid[i_neck]:
                    i_neck = i
                if perims_valid[i] > perims_valid[i - 1] and i > i_head:
                    break
            best_neck = {'h': hs_valid[i_neck], 'perim': perims_valid[i_neck]}
        
            # ombro: maior após o pescoço
            i_shoulder = i_neck + int(np.argmax(perims_valid[i_neck:]))
            best_shoulder = {'h': hs_valid[i_shoulder], 'perim': perims_valid[i_shoulder]}

    print(f"h_head_top={h_head_top:.2f}   topo da cabeça")
    print(f"h_shoulders_end={h_shoulders_end:.2f}   fim dos ombros")
    print(f"h_arms_begin={h_arms_begin:.2f}  início dos braços")
    print(f"cabeca:  h={best_head['h']:.2f}   perim={best_head['perim']:.4f}")
    print(f"pescoco: h={best_neck['h']:.2f}   perim={best_neck['perim']:.4f}")
    print(f"ombro:   h={best_shoulder['h']:.2f}   perim={best_shoulder['perim']:.4f}")

    return h_arms_begin, best_neck, best_shoulder


def detect_legs_heights(h_min, h_arms_begin, verts, tris, axis):
    """Detecção de landmarks do início dos braços até a base: Varre a malha a partir da altura em que há 3 agrupamentos encontrada
    em detect_preliminary_heights que é a fatia (tronco + braços) até a altura em que haja somente um cluster (tronco mais provavelmente 
    entre o fim da cintura e início do quadril), marca a altura de início em que há 2 clusters (início das pernas) e continua até que os
    perímetros dos dois agrupamentos crescam abruptamente e a varredura chegue até a última altura (início dos pés).
    
    Ademais, as variáveis h_arms_end e h_hip_end armazenam os valores de fim dos braços (transição tronco+braços - tronco/quadril) 
    e fim do quadril (transição quadril - pernas) que são usadas posteriormente para marcações das alturas do quadril. 
    """

    h_arms_end = None
    h_hip_end = None
    h_legs_begin = None
    h_feet_begin = None
    state = "looking_3"
    perims_lower = []  # lista de (h_s, perim_esq, perim_dir)

    for i, h_s in enumerate(np.linspace(h_arms_begin, h_min, 50)):
        segs, _ = _slice_mesh(verts, tris, h_s, axis)
        cl = _extract_slice_clusters(segs, 0.02) if len(segs) >= 2 else []
        n_clusters = len(cl)

        print(
            f"i:{i}\n"
            f"h_s:{h_s}\n"
            f"len(cl):{n_clusters}"
        )

        # fim dos braços
        if state == "looking_3":
            if n_clusters == 3:
                h_arms_end = h_s
            elif h_arms_end is not None:
                state = "looking_1"

        # fim do tronco/quadril e início das pernas (perímetros esquerdo e direito separados)
        if state == "looking_1":
            if n_clusters == 1:
                h_hip_end = h_s
            elif h_hip_end is not None and n_clusters == 2:
                h_legs_begin = h_s
                state = "looking_2"

        if state == "looking_2":
            if n_clusters == 2 and len(segs) >= 300:
                classified = _classify_clusters(cl)
                left  = classified.get('left')
                right = classified.get('right')

                _, p_esq = _hull_perimeter_and_pts(left)  if left  is not None else (None, 0.0)
                _, p_dir = _hull_perimeter_and_pts(right) if right is not None else (None, 0.0)
                p_esq, p_dir = p_esq or 0.0, p_dir or 0.0

                if perims_lower:
                    mean_esq = np.mean([p[1] for p in perims_lower])
                    mean_dir = np.mean([p[2] for p in perims_lower])
                    if p_esq > mean_esq * 1.3 or p_dir > mean_dir * 1.3:
                        h_feet_begin = h_s
                        state = "finished"
                        continue
                
                perims_lower.append((h_s, p_esq, p_dir))

            if state == "finished":
                break


    def _find_landmarks_leg(perms, hs):
        n = len(perms)

        lo, hi   = n // 3, 2 * n // 3
        i_joelho = lo + int(np.argmin(perms[lo:hi]))

        i_torn = i_joelho
        rising_count = 0
        for i in range(i_joelho + 1, n):
            if perms[i] > perms[i - 1]:
                rising_count += 1
                if rising_count >= 3:
                    i_torn = i - rising_count
                    break
            else:
                rising_count = 0
                if perms[i] < perms[i_torn]:
                    i_torn = i

        if i_torn <= i_joelho:
            i_torn = n - 1

        upper_idx = np.linspace(0, i_joelho, 7).astype(int)
        (i_perna, i_perna_quad, i_quad, i_quad_coxa, i_coxa, i_coxa_joelho, i_joelho_final) = upper_idx
        

        # região 2: panturrilha dividida em 2 pontos iguais entre joelho e tornozelo
        pant_idx = np.linspace(i_joelho, i_torn, 4).astype(int) #4 pontos
        i_gast, i_soleo = pant_idx[1], pant_idx[2]

        return (i_perna, i_perna_quad, i_quad, i_quad_coxa, i_coxa, i_coxa_joelho, i_joelho_final, i_gast, i_soleo, i_torn)

    landmarks_legs = {}
    if perims_lower:
        hs_legs   = np.array([p[0] for p in perims_lower])
        perms_esq = np.array([p[1] for p in perims_lower])
        perms_dir = np.array([p[2] for p in perims_lower])

        (i_perna_e, i_perna_quad_e, i_quad_e, i_quad_coxa_e, i_coxa_e,
        i_coxa_joelho_e, i_joe_e, i_pant1_e, i_pant2_e, i_torn_e) = _find_landmarks_leg(perms_esq, hs_legs)
        (i_perna_d, i_perna_quad_d, i_quad_d, i_quad_coxa_d, i_coxa_d,
        i_coxa_joelho_d, i_joe_d, i_pant1_d, i_pant2_d, i_torn_d) = _find_landmarks_leg(perms_dir, hs_legs)

        landmarks_legs_esq = {
            'pect_esq':          hs_legs[i_perna_e],
            'pect_quad_esq':     hs_legs[i_perna_quad_e],
            'quad_esq':          hs_legs[i_quad_e],
            'quad_coxa_esq':     hs_legs[i_quad_coxa_e],
            'coxa_esq':          hs_legs[i_coxa_e],
            'coxa_joelho_esq':   hs_legs[i_coxa_joelho_e],
            'joelho_esq':        hs_legs[i_joe_e],
            'gast_esq':          hs_legs[i_pant1_e],
            'soleo_esq':         hs_legs[i_pant2_e],
            'torn_esq':          hs_legs[i_torn_e],
        }
        
        landmarks_legs_dir = {
            'pect_dir':          hs_legs[i_perna_d],
            'pect_quad_dir':     hs_legs[i_perna_quad_d],
            'quad_dir':          hs_legs[i_quad_d],
            'quad_coxa_dir':     hs_legs[i_quad_coxa_d],
            'coxa_dir':          hs_legs[i_coxa_d],
            'coxa_joelho_dir':   hs_legs[i_coxa_joelho_d],
            'joelho_dir':        hs_legs[i_joe_d],
            'gast_dir':          hs_legs[i_pant1_d],
            'soleo_dir':         hs_legs[i_pant2_d],
            'torn_dir':          hs_legs[i_torn_d],
        }
    
    for name, h in landmarks_legs_esq.items():
        print(f"{name}: h={h:.2f}")

    for name, h in landmarks_legs_dir.items():
        print(f"{name}: h={h:.2f}")  

    print("\n==============================")
    print(f"h_arms_end={h_arms_end:.2f}")
    print(f"h_hip_end={h_hip_end:.2f}")
    print(f"h_legs_begin={h_legs_begin:.2f}")
    print(f"h_feet_begin={h_feet_begin}")
    print("==============================")

    return landmarks_legs_esq, landmarks_legs_dir, h_arms_end, h_hip_end


def detect_upper_trunk_heights(best_neck, best_shoulder):
    """Obtenção do ponto referente a altura do pescoço que são os perímetros com menores valores nos agrupamentos das fatias entre 
    a região do topo até o crescimento abrupto dos perímetros dos próximos agrupamentos dessas fatias.
    """
    trunk_upper_heights = np.linspace(best_neck['h'], best_shoulder['h'], 4)

    compress_frac = 0.7  # 60% do intervalo original — ajuste conforme necessário

    h_end_compressed = best_neck['h'] - compress_frac * (best_neck['h'] - best_shoulder['h'])

    trunk_upper_heights = np.linspace(best_neck['h'], h_end_compressed, 4)

    landmarks_upper_trunk = {
        'pescoco': trunk_upper_heights[0],
        'omb_sup': trunk_upper_heights[1],
        'omb_med': trunk_upper_heights[2],
        'omb_inf': trunk_upper_heights[3],
    }
    return landmarks_upper_trunk


def detect_lower_trunk_heights(h_arms_begin, h_hip_end):
    """Detecção dos pontos do tronco inferior (busto, cintura e quadril) que encontram-se entre o início dos braços (primeira fatia
    detectada com 3 agrupamentos) e o fim do quadril (última fatia detectada com 1 agrupamento). 
    """

    trunk_lower_heights = np.linspace(h_arms_begin, h_hip_end, 9)

    offset = 0.10 * abs(h_arms_begin - h_hip_end)  # 10% do intervalo total
    trunk_lower_heights += offset  # sobe todos os pontos

    landmarks_lower_trunk = {
        'bus_sup': trunk_lower_heights[0],
        'bus_med': trunk_lower_heights[1],
        'bus_inf': trunk_lower_heights[2],
        'cin_sup': trunk_lower_heights[3],
        'cin_med': trunk_lower_heights[4],
        'cin_inf': trunk_lower_heights[5],
        'quadl_sup': trunk_lower_heights[6],
        'quadl_med': trunk_lower_heights[7],
        'quadl_inf': trunk_lower_heights[8],
    }

    return landmarks_lower_trunk


def scan_slices_top_to_bottom(mesh: o3d.geometry.TriangleMesh, verts, tris, h_min, h_max, full_name, axis: int = 2):
    """Método unificado que contém os retornos dos métodos:
        - detect_preliminary_heights(): Alturas preliminares: Início dos braços, pescoço e início dos ombros.
        - detect_legs_heights(): Varredura desde o fim dos braços/mãos até os pés (ignorando esses pelos seus perímetros crescentes).
        - detect_upper_trunk_heights(): Transição entre pescoço e início dos braços:
                                         - busto superior: Ombros/trapézio superior, médio e inferior
        - detect_arms_heights(): regiões de transição entre o fim dos ombros que é o início dos 3 agrupamentos até o fim dos 3 
        agrupamentos.
        - detect_lower_trunk_heights(): Tronco inferior, regiões remanescentes caracterizadas pelas transições do início dos braços (3 agrupamentos)
        até o fim da cintura (final de um agrupamento)
        
        Seu objetivo consiste na varredura completa da malha do topo a base. Retorna as alturas de cada landmark, agrupamentos, fator de
        normalização, segmentos de retas entre os pontos das fatias.
    """
    begin1 = time.time()
    
    #Fatias da base
    h_mid = (h_max + h_min) / 2

    ############      MEDIÇÃO DA PARTE DE CIMA - ATÉ AS JUNTAS DOS BRAÇOS      ############
    h_arms_begin, best_neck, best_shoulder = detect_preliminary_heights(h_max, h_mid, verts, tris, axis)

    ############            DETECÇÃO DE LANDMARKS NA REGIÃO DAS PERNAS         ############
    landmarks_legs_esq, landmarks_legs_dir, h_arms_end, h_hip_end = detect_legs_heights(h_min, h_arms_begin, verts, tris, axis)

    ########### DETECÇÃO DE LANDMARKS NO PESCOÇO E OMBROS (3 PONTOS) ############
    landmarks_upper_trunk = detect_upper_trunk_heights(best_neck, best_shoulder)

    ###########          DETECÇÃO DE LANDMARKS NOS BRAÇOS           ############
    landmarks_arms_esq, landmarks_arms_dir = detect_arms_heights(verts, tris, axis, h_arms_begin, h_arms_end)

    ########### DETECÇÃO DE LANDMARKS NO BUSTO: CINTURA E QUADRIL (3 PONTOS CADA)  ############
    landmarks_lower_trunk = detect_lower_trunk_heights(h_arms_begin, h_hip_end)

    landmark_heights = {}
    
    landmark_heights.update(landmarks_upper_trunk)
    landmark_heights.update(landmarks_lower_trunk)
    landmark_heights.update(landmarks_arms_dir)
    landmark_heights.update(landmarks_arms_esq)
    landmark_heights.update(landmarks_legs_esq)
    landmark_heights.update(landmarks_legs_dir)

    heights = np.array(sorted(landmark_heights.values(), reverse=True))
    heights = np.unique(heights)[::-1]
     
    # print(f"verts min={verts.min():.4f}  max={verts.max():.4f}")
    # print(f"h_min={h_min:.4f}  h_max={h_max:.4f}")
    # print(f"heights[0]={heights[0]:.4f}  heights[-1]={heights[-1]:.4f}")
    
    print(f"total heights={len(heights)}")
    for k, h in enumerate(heights):
        print(f"  idx={k:02d}  h={h:.4f}  norm={((h-h_min)/(h_max-h_min))*100:.1f}%")
    
    num_slices     = len(heights)
    norm           = (heights - h_min) / (h_max - h_min)   # 1.0 = top, 0.0 = bottom
    n_clusters_arr = np.zeros(num_slices, dtype=int)
    clusters       = [[] for _ in range(len(heights))]
    mesh_slices    = []
    segments       = [[] for _ in range(num_slices)]
    
    print("heights gerados")
    for k, h in enumerate(heights):
        begin2 = time.time()
        print(f"\nSlice {k}/{num_slices}")
        print(f"h = {h:.4f}")
        
        segs, pts_2d = _slice_mesh(verts, tris, h, axis)
        # print("após slice mesh")   

        if len(segs) < 2:
            continue
        
        clusts    = _extract_slice_clusters(segs, 0.02)
        if not clusts:
            continue

        clusters[k]       = clusts
        n_clusters_arr[k] = len(clusts)
        mesh_slices.append(pts_2d)
        segments[k]       = segs
        
        end2 = time.time() - begin2
        print(f"idx={k:02d}  h={h:.4f}  n_segs={len(segs)}  n_clusters={len(clusts)}")
        print(f"scan_slices_top_to_bottom loop finishing time: {end2}")
                
    for k in range(len(mesh_slices)):
        print(
            f"{k:03d}",
            f"clusters={n_clusters_arr[k]}"
        )
        
    # mesh_slices: lista de np.arrays — funciona direto
    np.savez_compressed(f'./{full_name}/{full_name}_mesh_slices.npz', **{f'arr_{i}': a for i, a in enumerate(mesh_slices)})

    # segments: lista de listas de tuplas — usa pickle
    import pickle
    with open(f'./{full_name}/{full_name}_segments.pkl', 'wb') as f:
        pickle.dump(segments, f)

    # clusters: lista de listas de np.arrays — usa o padrão com chaves aninhadas
    save_dict = {}
    for k, cluster_list in enumerate(clusters):
        for c_idx, arr in enumerate(cluster_list):
            save_dict[f'slice{k:05d}_cluster{c_idx:02d}'] = arr
    np.savez_compressed(f'./{full_name}/{full_name}_clusters.npz', **save_dict)
    np.savez(f"./{full_name}/{full_name}_heights_and_norm.npz", heights=heights, norm=norm)
    
    
    end1 = time.time() - begin1
    print(f"scan_slices_top_to_bottom finishing time:{end1}")
    return (heights, norm, clusters, mesh_slices, segments, landmark_heights)


def compute_perimeters_from_landmarks(landmarks, clusters, heights, norm, full_name,
                                      *, verts=None, tris=None, axis=2,
                                      arm_window=5, threshold_ratio=0.15):
    """
    Calcula perímetro e hull por landmark usando o cluster correto (center para tronco/cabeça, left/right para braços e pernas), ademais,
    o seguinte método divide e fixa os agrupamentos encontrados em cada altura dos landmarks detectados nos lados esquerdo, central e
    direito e então retorna as medidas (perímetros) desses landmarks e a lista com os fechos convexos em cada landmark para posterior
    apresentação dessas medidas. 
    """
    left_lm  = {n for n in LANDMARK_STYLE if n.endswith('_esq')}
    right_lm = {n for n in LANDMARK_STYLE if n.endswith('_dir')}

    perimeters    = {}
    hull_pts_list = {}

    leg_chain = ['pect', 'pect_quad', 'quad', 'quad_coxa', 'coxa',
                 'coxa_joelho', 'joelho', 'gast', 'soleo', 'torn']

    arm_chain = ['bic_longo', 'bic_curto', 'braq', 'braquiorr',
                 'cot', 'ant', 'puls']

    arm_names = {f'{base}_{suffix}': side for base in arm_chain
                 for suffix, side in (('esq', 'left'), ('dir', 'right'))}

    planes = None
    if any(name in arm_names for name in landmarks):
        if axis != 2:
            raise ValueError('Esta integracao espera a varredura original em Z (axis=2).')
        if verts is None or tris is None:
            raise ValueError('Passe verts=verts e tris=tris para refatiar os bracos.')
        planes = _arm_final_planes(clusters, heights, arm_window, threshold_ratio)

    def _process_chain(chain):
        """Processa uma cadeia com tracking, retorna dict nome_completo -> cluster."""
        prev_left_center = None
        result = {}

        for base_name in chain:
            name_esq = f"{base_name}_esq"
            name_dir = f"{base_name}_dir"

            idx = landmarks.get(name_esq, landmarks.get(name_dir))
            if idx is None or idx < 0 or idx >= len(clusters):
                continue

            clusts = clusters[idx]
            if len(clusts) != 2:
                # fallback: classificação normal se não forem exatamente 2
                classified = _classify_clusters(clusts) or {}
                left, right  = classified.get('left'), classified.get('right')
            else:
                c0, c1 = clusts
                center0, center1 = c0.mean(axis=0), c1.mean(axis=0)

                if prev_left_center is None:
                    left, right = (c0, c1) if center0[0] < center1[0] else (c1, c0)
                else:
                    d0 = np.linalg.norm(center0-prev_left_center)
                    d1 = np.linalg.norm(center1-prev_left_center)
                    left, right = (c0, c1) if d0 < d1 else (c1, c0)
                prev_left_center = left.mean(axis=0)
            if name_esq in landmarks:
                result[name_esq] = left
            if name_dir in landmarks:
                result[name_dir] = right
        return result

    tracked_legs = _process_chain(leg_chain)

    for name, idx in landmarks.items():
        if idx is None or idx < 0 or idx >= len(clusters):
            continue
        if name in arm_names:
            # Cada lado usa seu proprio indice; nao reutilizar idx esquerdo no direito.
            plane = planes[arm_names[name]][idx]
            if plane is None:
                warnings.warn(f'{name}: falta sequencia de sondagens com 3 clusters em h={heights[idx]:.5f}.',
                            RuntimeWarning, stacklevel=2)
                continue
            hp, perim = _arm_final_hull(verts, tris, plane)
            if hp is None:
                warnings.warn(f'{name}: contorno do braco nao encontrado no plano calculado.',
                            RuntimeWarning, stacklevel=2)
                continue
            print(f"[arm] {name}: {'inclinado' if plane['tilted'] else 'horizontal'} "
                f"ratio={plane['ratio']:.4f}, normal={plane['normal']}")
        else:
            if name in tracked_legs:
                c = tracked_legs[name]
            else:
                clusts = clusters[idx]
                classified = _classify_clusters(clusts) or {}
                if name in left_lm:
                    c = classified.get('left')
                elif name in right_lm:
                    c = classified.get('right')
                elif name in ('bus', 'qua'):
                    c = max(clusts, key=len) if clusts else None
                else:
                    c = classified.get('center')
                    if c is None:
                        c = clusts[0] if clusts else None
            if c is None or len(c) < 3:
                continue
            hp, perim = _hull_perimeter_and_pts(c)
            if hp is None:
                continue
            
        perimeters[name]    = perim
        hull_pts_list[name] = hp #c
    
    # np.savez_compressed(f'./{full_name}/{full_name}_hull_pts_list_slices.npz', **{f'arr_{i}': a for i, a in enumerate(hull_pts_list)})
    # np.savez(f"./{full_name}/{full_name}_perimeters.npz", heights=heights, norm=norm)
    return perimeters, hull_pts_list


def plot_profile(heights, norm, perimeters, landmarks, scale=1.0, unit_label="units", out_path="circumference_profile.png"):
    """Método usado na versão desktop por enquanto, ainda avaliando a necessidade e viabilidade desse gráfico na interface web front-end"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from scipy.signal import savgol_filter
    except ImportError:
        print("[info] matplotlib/scipy not found – skipping profile plot.")
        return

    p = np.array([perimeters.get(name, 0.0) for name in landmarks.keys()])
    x = np.arange(len(p))

    if len(p) >= 5:
        win = max(5, len(p) // 3)
        win = win if win % 2 == 1 else win + 1
        win = min(win, len(p))
        smoothed = savgol_filter(p, win, 3)
    else:
        smoothed = p.copy()

    fig, ax = plt.subplots(figsize=(12, 7))

    ax2 = ax.twiny()
    ticks_idx = np.linspace(0, len(x) - 1, min(6, len(x))).astype(int)
    ax2.set_xticks(x[ticks_idx])
    ax2.set_xticklabels([f"{heights[list(landmarks.values())[i]]:.2f}" for i in ticks_idx])
    ax2.set_xlabel("Real height")

    ax.plot(x, p * scale, color='steelblue', alpha=0.25, linewidth=1.0, label='Raw profile')
    ax.plot(x, smoothed * scale, color='steelblue', linewidth=2.5, label='Smoothed profile')

    # ── landmarks — sem states, sem axvspan ───────────────────────────────
    landmark_patches = []
    for lname, idx in landmarks.items():
        if idx is None or idx >= len(smoothed):
            continue
        c     = LANDMARK_STYLE[lname][0]
        label = LANDMARK_STYLE[lname][1]
        value = smoothed[idx] * scale

        ax.axvline(x[idx], color=c, linestyle='--', linewidth=2)
        ax.scatter([x[idx]], [value], color=c, s=90, zorder=10)
        ax.text(x[idx], value, f" {label}", color=c, fontsize=10, weight='bold')
        landmark_patches.append(
            mpatches.Patch(color=c, label=f"{label}: {value:.1f} {unit_label}")
        )

    ax.set_xlabel('Slice index (0 = topo · N = base)', fontsize=11)
    ax.set_ylabel(f'Circumference ({unit_label})', fontsize=11)
    ax.set_title('Body Circumference Profile', fontsize=14)

    handles = ax.get_legend_handles_labels()[0]
    ax.legend(handles=[*handles[:2], *landmark_patches], fontsize=10, loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax2.set_xlim(ax.get_xlim())

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Profile plot saved -> {out_path}")
    plt.show()

    
#Usado somente na versão Desktop com o Open3D 
def visualize_landmarks(
        mesh,
        heights,
        hull_pts_list,
        landmarks,
        vertical_axis=2):
    """Descontinuado: Visualização dos arcos contornados nos landmarks, sua descontinuidade deve-se ao atual uso da interface web
    front end com FastAPI"""

    geometries = [mesh]
    
    colors = {
         'pescoco':         [1.00, 0.00, 0.00],

         # CENTRO (Amarelo Luminoso -> Âmbar / Ouro Escuro)
         'omb_sup':         [1.00, 0.95, 0.20],  # Amarelo claro
         'omb_med':         [1.00, 0.90, 0.10],
         'omb_inf':         [1.00, 0.85, 0.00],  # Amarelo ouro puro
         'bus_sup':         [1.00, 0.80, 0.00],
         'bus_med':         [1.00, 0.75, 0.00],
         'bus_inf':         [1.00, 0.70, 0.00],  # Âmbar claro
         'cin_sup':         [0.95, 0.65, 0.00],
         'cin_med':         [0.90, 0.60, 0.00],
         'cin_inf':         [0.85, 0.55, 0.00],  # Âmbar médio
         'quadl_sup':       [0.80, 0.50, 0.00],
         'quadl_med':       [0.75, 0.45, 0.00],
         'quadl_inf':       [0.70, 0.40, 0.00],  # Dourado bronze escuro
    
         # LADO ESQUERDO (Roxo Profundo -> Magenta -> Rosa Claro)
         'bic_longo_esq':   [0.55, 0.00, 0.60],  # Roxo profundo
         'bic_curto_esq':   [0.70, 0.00, 0.75],  # Violeta
         'braq_esq':        [0.85, 0.00, 0.85],  # Magenta vivo
         'braquiorr_esq':   [0.95, 0.10, 0.85],  # Fúcsia
         'cot_esq':         [1.00, 0.25, 0.85],  # Rosa choque
         'ant_esq':         [1.00, 0.45, 0.85],  # Rosa médio
         'puls_esq':        [1.00, 0.65, 0.90],  # Rosa claro pastel
    
         # LADO DIREITO (Azul Marinho -> Azul Elétrico -> Ciano Claro)
         'bic_longo_dir':   [0.00, 0.20, 0.60],  # Azul marinho
         'bic_curto_dir':   [0.00, 0.35, 0.75],  # Azul cobalto
         'braq_dir':        [0.00, 0.50, 0.90],  # Azul elétrico
         'braquiorr_dir':   [0.00, 0.65, 1.00],  # Azul celeste
         'cot_dir':         [0.00, 0.80, 1.00],  # Ciano vivo
         'ant_dir':         [0.20, 0.90, 1.00],  # Ciano piscina
         'puls_dir':        [0.50, 0.95, 1.00],  # Ciano gelo brilhante
         
          # PERNA LADO ESQUERDO: Família do Vermelho / Laranja
         'pect_esq':        [0.55, 0.00, 0.00],  # Vermelho escuro
         'pect_quad_esq':   [0.65, 0.00, 0.00],
         'quad_esq':        [0.75, 0.00, 0.00],
         'quad_coxa_esq':   [0.85, 0.00, 0.00],
         'coxa_esq':        [1.00, 0.00, 0.00],  # Vermelho puro
         'coxa_joelho_esq': [1.00, 0.15, 0.00],
         'joelho_esq':      [1.00, 0.25, 0.00],
         'gast_esq':        [1.00, 0.35, 0.00],
         'soleo_esq':       [1.00, 0.45, 0.00],
         'torn_esq':        [1.00, 0.55, 0.00],  # Laranja vivo
    
         # PERNA LADO DIREITO: Família do Verde
         'pect_dir':        [0.00, 0.40, 0.00],  # Verde floresta escuro
         'pect_quad_dir':   [0.00, 0.50, 0.00],
         'quad_dir':        [0.00, 0.60, 0.00],
         'quad_coxa_dir':   [0.00, 0.70, 0.05],
         'coxa_dir':        [0.00, 0.80, 0.10],
         'coxa_joelho_dir': [0.00, 0.90, 0.15],
         'joelho_dir':      [0.00, 1.00, 0.20],
         'gast_dir':        [0.15, 1.00, 0.30],
         'soleo_dir':       [0.30, 1.00, 0.45],
         'torn_dir':        [0.45, 1.00, 0.60],  # Verde menta brilhante
    }

    for name, idx in landmarks.items():

        hp = hull_pts_list.get(name)

        if hp is None or len(hp) < 3:
            continue
    
        print(f"name:{name}\nidx:{idx}")
        ring = make_ring_lineset(
            hp,
            heights[idx],
            vertical_axis,
            colors[name]
        )

        geometries.append(ring)

        print(
            f"{name}  "
            f"slice={idx}  "
            f"height={heights[idx]:.4f}"
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Landmarks",
        width=1200,
        height=900,
        mesh_show_back_face=True
    )
    
    return geometries


def align_to_z_and_convert_to_centimeters(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Alinhamento da malha no eixo z para fatiamento do topo da cabeça até os pés para a obtenção de medidas
    transversais com cortes horizontais."""
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    centered = vertices - center

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    body_axis = eigenvectors[:, np.argmax(eigenvalues)]

    if body_axis[2] < 0:
        body_axis = -body_axis

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(body_axis, z)
    s = np.linalg.norm(v)
    c = np.dot(body_axis, z)

    if s < EPS:
        return mesh

    vx = np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    rotated = (R @ centered.T).T + center
    mesh.vertices = o3d.utility.Vector3dVector(rotated)
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector((R @ normals.T).T)

    vertices = np.asarray(mesh.vertices)  # Open3d -> numpy
    vertices /= 10.0   # mm -> cm
    mesh.vertices = o3d.utility.Vector3dVector(vertices)  #numpy -> Open3d

    print(f"Eixo do corpo alinhado com Z (rotação aplicada)")
    return mesh, vertices


def display_mesh_info(vertical_axis, mesh):
    """Usado para depuração e visualização do progresso do processamento: Alturas máximas, mínimas e medianas dos eixos x,y e z"""
    vertices = np.asarray(mesh.vertices)
    h_min = vertices[:, vertical_axis].min()
    h_max = vertices[:, vertical_axis].max()
    
    print(f"         Vertices: {len(vertices):,}   |   Triangles: {len(mesh.triangles):,}")
    print(f"         Height range (axis {vertical_axis}): [{h_min:.4f}, {h_max:.4f}]")
    print("X range:", vertices[:, 0].min(), vertices[:, 0].max(), vertices[:, 0].max() - vertices[:, 0].min())
    print("Y range:", vertices[:, 1].min(), vertices[:, 1].max(), vertices[:, 1].max() - vertices[:, 1].min())
    print("Z range:", vertices[:, 2].min(), vertices[:, 2].max(), vertices[:, 2].max() - vertices[:, 2].min())
    print(f"Eixo vertical usado: {axis_name(vertical_axis)}")


def measuring_mesh(mesh, vertical_axis, vertices, full_name, scanning_performed = False):
    """Pipeline para detecção de landmarks e extração de medidas:
        1 - scanning_performed criado para fim de posterior debug afim de realizar diretamente a atribuição dos arcos nos landmarks
        e apresentar as medidas sem a necessidade de reobter todas as alturas do 0 caso já realizada uma iteração, se não foi realizada
        nenhuma iteração obtém-se somente as alturas dos landmarks.
        2 - Listam-se as alturas dos landmarks mais próximos.
        3 - Obtém-se as listas dos perímetros e fechos convexos.
        4 - As medidas são atribuídos em uma lista com seus valores junto com os nomes dos landmarks.
        5 - Geração dos arcos (linesets) de cada landmark encontrado para posterior apresentação.
        6 - Por fim retornam-se 5 e 6. 
        """
    begin = time.time()
    medidas = {}
    landmarks = {}

    if scanning_performed:
        # # mesh_slices
        # data = np.load(f'./{full_name}/{full_name}_mesh_slices.npz')
        # mesh_slices = [data[k] for k in sorted(data.files, key=lambda x: int(x.replace('arr_', '')))]
        
        # segments
        with open(f'./{full_name}/{full_name}_segments.pkl', 'rb') as f:
            segments = pickle.load(f)

        # clusters
        data = np.load(f'./{full_name}/{full_name}_clusters.npz')
        slice_map = {}
        for key in sorted(data.files):
            slice_key, c_idx = key.split('_cluster')
            if slice_key not in slice_map:
                slice_map[slice_key] = []
            slice_map[slice_key].append(int(c_idx))

        clusters = []
        for slice_key in sorted(slice_map.keys()):
            cluster_list = [data[f'{slice_key}_cluster{c_idx:02d}'] for c_idx in sorted(slice_map[slice_key])]
            clusters.append(cluster_list)

        #hull_pts_list
        data = np.load(f'./{full_name}/{full_name}_hull_pts_list.npz')
        hull_pts_list = [data[f'arr_{i}'] for i in range(len(data.files))]
        
        #heights and norm
        load_file= np.load(f"./{full_name}/{full_name}_heights_and_norm.npz")
        heights, norm = load_file["heights"], load_file["norm"]
        
        #perimeters
        load_file= np.load(f"./{full_name}/{full_name}_perimeters.npz", allow_pickle=True)
        perimeters = load_file["perimeters"]
    else:
        #Pega as alturas maior e menor no eixo Z
        h_min = vertices[:, vertical_axis].min()
        h_max = vertices[:, vertical_axis].max()
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        
        (heights, norm, clusters, mesh_slices, segments, landmark_heights) = scan_slices_top_to_bottom(mesh, verts, tris, h_min, h_max, full_name, axis=vertical_axis)

    def nearest_idx(h):
        return int(np.argmin(np.abs(heights - h)))

    for name, h in landmark_heights.items():
        landmarks[name] = nearest_idx(h)

    if not landmarks:
        sys.exit("[erro] Não foi possível detectar os landmarks.")

    for name, idx in landmarks.items():
        print(f"name:{name}\nidx:{idx}")

    perimeters, hull_pts_list = compute_perimeters_from_landmarks(landmarks, clusters, heights, norm, full_name, verts=verts, tris=tris, axis=vertical_axis)

    scale = 1.0
    unit_label = 'cm'
    
    print(" MEDIDAS CORPORAIS ")
    for name in LANDMARK_STYLE.keys():
        if name not in landmarks:
            print(f"  {LANDMARK_STYLE[name][1]} – NÃO DETECTADO")
            continue
        if name not in perimeters:
            print(f"  {LANDMARK_STYLE[name][1]} – SEM PERÍMETRO")
            continue
        idx  = landmarks[name]
        circ = perimeters[name] * scale
        medidas[LANDMARK_STYLE[name][1].strip()] = round(circ,1)
        print(f"  {LANDMARK_STYLE[name][1]}: {circ:6.1f} {unit_label}"
            f"   ({norm[idx]*100:.1f}% da altura)")
    print("═" * 40 + "\n")
        
    line_sets = []
    for name_upper,(name_lower, hp) in zip(medidas.keys(), hull_pts_list.items()):
        if hp is None or len(hp) < 3:
            print(f" [AVISO] Sem contorno válido para '{name_lower}' - anel ignorado.")
            continue
        idx = landmarks[name_lower]
        ls = make_ring_lineset(hp, heights[idx], vertical_axis, LANDMARK_STYLE[name_lower][0])
        line_sets.append((name_upper,ls))


    # plot_clusters(clusters, hull_pts_list, landmarks, full_name)

    end = time.time() - begin    
    print(f"measuring_mesh exec time:{end}")
    return line_sets, medidas
