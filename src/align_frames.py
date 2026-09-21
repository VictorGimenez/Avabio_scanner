import copy
import time
import os
import re
import json

import cv2 as cv
import open3d as o3d
import numpy as np

from pathlib import Path

class AlinhamentoRejeitado(RuntimeError):
    pass

def load_angles_and_paths(save_path):
    angles_json_path = Path(save_path) / "angles.json"

    if angles_json_path.exists():
        with open(angles_json_path) as f:
            angles_by_frame = json.load(f)
    else:
        print(
            "[WARN] angles.json nao encontrado, tentando angles.npy "
            "(posicional, sem garantia de sincronia)"
        )

        
        angles_list = np.load(f"{save_path}/angles.npy").tolist()
        paths_tmp = sorted(Path(save_path).glob("*_clean.ply"))
        angles_by_frame = {}

        for p, a in zip(paths_tmp, angles_list):
            m = re.search(r'(\d+)_clean\.ply', p.name)
            if m:
                angles_by_frame[m.group(1)] = a

    paths_all = sorted(Path(save_path).glob("*_clean.ply"))
    paths_valid, angles_deg = [], []

    for p in paths_all:
        m = re.search(r'(\d+)_clean\.ply', p.name)

        if not m:
            continue

        idx = m.group(1)

        if idx in angles_by_frame:
            paths_valid.append(p)
            angles_deg.append(angles_by_frame[idx])
        else:
            print(
                f"[WARN] Frame {idx} sem angulo correspondente -- descartado"
            )

    print(f"[INFO] {len(paths_valid)}/{len(paths_all)} frames com angulo valido")
    return paths_valid, angles_deg


def remove_cluster_artifacts(pcd):
    labels = np.array(
        pcd.cluster_dbscan(
            eps=30.0,
            min_points=50,
            print_progress=True
        )
    )

    maior_cluster = np.bincount(labels[labels >= 0]).argmax()
    indices = np.where(labels == maior_cluster)[0]
    pcd_clean = pcd.select_by_index(indices)

    return pcd_clean


def centralizar_nuvem(pcd):
    """Move a nuvem para que seu centroide fique na origem."""
    centroid = pcd.get_center()
    pcd.translate(-centroid)
    return pcd


def transformacao_inicial_turntable(angulo_graus, eixo='y'):
    """Cria matriz de rotacao inicial para turntable."""
    angulo_rad = np.radians(angulo_graus)

    if eixo == 'y':
        return np.array([
            [np.cos(angulo_rad), 0, np.sin(angulo_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(angulo_rad), 0, np.cos(angulo_rad), 0],
            [0, 0, 0, 1]
        ])

    elif eixo == 'z':
        return np.array([
            [np.cos(angulo_rad), -np.sin(angulo_rad), 0, 0],
            [np.sin(angulo_rad), np.cos(angulo_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


def pairwise_registration(
    source,
    target,
    transf_init,
    threshold_coarse,
    threshold_fine,
    calcular_information=True,
    max_iter_coarse=20,
    max_iter_fine=30
):
    begin = time.perf_counter()

    reg_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold_coarse,
        transf_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter_coarse
        )
    )

    reg_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold_fine,
        reg_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter_fine
        )
    )

    information = None

    if calcular_information:
        information = (
            o3d.pipelines.registration
            .get_information_matrix_from_point_clouds(
                source,
                target,
                threshold_fine,
                reg_fine.transformation
            )
        )

    print(
        f"[ICP] fitness={reg_fine.fitness:.4f} | "
        f"RMSE={reg_fine.inlier_rmse:.4f} | "
        f"tempo={time.perf_counter() - begin:.2f}s"
    )

    return reg_fine.transformation, information, reg_fine.fitness


def load_and_prepare(path, voxel_size=3.0, normal_radius=30.0):
    pcd = o3d.io.read_point_cloud(str(path))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd = remove_cluster_artifacts(pcd)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    pcd = centralizar_nuvem(pcd)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30
        )
    )

    return pcd


def select_and_build_pose_graph(
    paths_all,
    angles_all,
    threshold_coarse,
    threshold_fine,
    voxel_size=1.5,
    fitness_min=0.3000,
    max_gap_deg=40.0,
    loop_closure=True
):
    n = len(paths_all)

    if n != len(angles_all):
        raise ValueError("paths e angles fora de sincronia.")

    if n < 2:
        raise ValueError("Precisa de pelo menos duas nuvens.")

    if not np.isfinite(fitness_min) or not 0.0 < fitness_min <= 1.0:
        raise ValueError("fitness_min deve estar entre 0 e 1.")

    # Piso obrigatorio, inclusive se a chamada passar um valor menor.
    fitness_min = max(0.3000, fitness_min)

    if not 0.0 < max_gap_deg < 360.0:
        raise ValueError("max_gap_deg deve estar entre 0 e 360.")

    angles_all = np.asarray(angles_all, dtype=float)

    if not np.all(np.isfinite(angles_all)):
        raise ValueError("Existem angulos invalidos.")

    # Pressupoe ordem de captura, rotacao crescente e intervalos
    # menores que 180 graus entre capturas.
    angles_all = np.unwrap(
        angles_all,
        period=360.0
    ).tolist()

    if np.any(np.diff(angles_all) < -1e-6):
        raise ValueError(
            "Angulos fora da ordem crescente. "
            "Confira a associacao entre arquivos e angulos."
        )

    pcd_cache = {}

    def get_pcd(idx):
        if idx not in pcd_cache:
            begin = time.perf_counter()

            try:
                pcd = load_and_prepare(
                    paths_all[idx],
                    voxel_size
                )

                if pcd is None or pcd.is_empty():
                    raise ValueError("Nuvem vazia.")

                if not np.all(np.isfinite(np.asarray(pcd.points))):
                    raise ValueError("Coordenadas invalidas.")

                pcd_cache[idx] = pcd

            except (ValueError, RuntimeError) as e:
                print(
                    f"[WARN] Frame {idx} invalido: "
                    f"{paths_all[idx].name} | {e}"
                )
                pcd_cache[idx] = None

            print(
                f"[PREPARO] frame={idx} | "
                f"tempo={time.perf_counter() - begin:.2f}s"
            )

        return pcd_cache[idx]

    def try_pair(cur_pcd, idx, cur_angle):
        gap = angles_all[idx] - cur_angle
        pcd_cand = get_pcd(idx)

        if pcd_cand is None:
            return gap, None, None, -1.0

        init = transformacao_inicial_turntable(gap, eixo='y')

        transf, info, fitness = pairwise_registration(
            cur_pcd,
            pcd_cand,
            init,
            threshold_coarse,
            threshold_fine,
            calcular_information=False
        )

        return gap, transf, info, fitness

    cur_pcd = get_pcd(0)

    if cur_pcd is None:
        raise RuntimeError("A primeira nuvem nao pode ser utilizada.")

    selected_idx = [0]
    edges = []

    cur_angle = angles_all[0]
    i = 1

    loop_result = None

    while i < n:
        # Ignora capturas sem avanco angular.
        while i < n and angles_all[i] - cur_angle <= 1e-6:
            i += 1

        if i >= n:
            break

        # Processa apenas a primeira volta.
        if angles_all[i] - angles_all[0] >= 360.0 - 1e-6:
            break

        best = None

        # Registro por vizinhos.
        # As intermediarias ajudam a calcular as poses, mas nao
        # precisam entrar na composicao final.
        for cand in range(i, n):
            gap_c = angles_all[cand] - cur_angle

            if angles_all[cand] - angles_all[0] >= 360.0 - 1e-6:
                break

            if gap_c > max_gap_deg + 1e-6:
                break

            if gap_c <= 1e-6:
                continue

            gap_c, transf_c, info_c, fit_c = try_pair(
                cur_pcd,
                cand,
                cur_angle
            )

            if (
                transf_c is None
                or not np.isfinite(fit_c)
                or fit_c < fitness_min
                or not np.all(np.isfinite(transf_c))
            ):
                print(
                    f"[REJEITADO] {selected_idx[-1]} -> {cand} | "
                    f"gap={gap_c:.1f} graus | "
                    f"fitness={fit_c:.4f}"
                )
                continue

            best = (
                cand,
                gap_c,
                transf_c,
                info_c,
                fit_c
            )
            break

        if best is None:
            print("[ERRO] Nenhum candidato apresentou alinhamento aprovado.")
            return None, None, None

        j_sel, gap_sel, transf_sel, info_sel, fit_sel = best

        if not np.isfinite(fit_sel) or fit_sel < max(0.3000, fitness_min):
            print(
                f"[ERRO] Alinhamento rejeitado: fitness={fit_sel:.4f}. "
                "Reconstrução interrompida."
            )
            return None, None, None

        # Calcula information somente para a ligacao aprovada.
        # Reutiliza a transformacao, sem repetir o ICP.
        info_sel = (
            o3d.pipelines.registration
            .get_information_matrix_from_point_clouds(
                cur_pcd,
                get_pcd(j_sel),
                threshold_fine,
                transf_sel
            )
        )

        if not np.all(np.isfinite(info_sel)):
            raise RuntimeError(
                f"Matriz de informacao invalida: "
                f"{selected_idx[-1]} -> {j_sel}."
            )

        edges.append(
            (
                len(selected_idx) - 1,
                len(selected_idx),
                transf_sel,
                info_sel
            )
        )

        print(
            f"[REGISTRO] {selected_idx[-1]} -> {j_sel} | "
            f"gap={gap_sel:.1f} graus | "
            f"fitness={fit_sel:.4f}"
        )

        selected_idx.append(j_sel)

        cur_pcd = get_pcd(j_sel)
        cur_angle = angles_all[j_sel]
        i = j_sel + 1

        angulo_percorrido = cur_angle - angles_all[0]
        angle_gap = 360.0 - angulo_percorrido

        if (
            loop_closure
            and len(selected_idx) > 2
            and angulo_percorrido >= 330.0 - 1e-6
            and 0.0 <= angle_gap <= min(30.0, max_gap_deg) + 1e-6
        ):
            init = transformacao_inicial_turntable(
                angle_gap,
                eixo='y'
            )

            transf, info, fitness = pairwise_registration(
                cur_pcd,
                get_pcd(0),
                init,
                threshold_coarse,
                threshold_fine,
                calcular_information=False
            )

            if (
                np.isfinite(fitness)
                and fitness >= fitness_min
                and np.all(np.isfinite(transf))
            ):
                info = (
                    o3d.pipelines.registration
                    .get_information_matrix_from_point_clouds(
                        cur_pcd,
                        get_pcd(0),
                        threshold_fine,
                        transf
                    )
                )

                if np.all(np.isfinite(info)):
                    loop_result = (transf, info)

                    print(
                        f"[LOOP ACEITO] gap={angle_gap:.1f} graus | "
                        f"fitness={fitness:.4f}"
                    )
                    break

            print(
                f"[LOOP REJEITADO] fitness={fitness:.4f}. "
                "Continuando com os frames restantes."
            )

    if len(selected_idx) < 2:
        raise RuntimeError(
            "Nao foi possivel construir uma sequencia de registro."
        )

    pcds = [pcd_cache[idx] for idx in selected_idx]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(np.identity(4))
    )

    odometry = np.identity(4)

    for a, b, transf, info in edges:
        odometry = transf @ odometry

        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)
            )
        )

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                a,
                b,
                transf,
                info,
                uncertain=False
            )
        )

    if loop_result is not None:
        transf, info = loop_result

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(pcds) - 1,
                0,
                transf,
                info,
                uncertain=True
            )
        )

    elif loop_closure:
        print(
            "[WARN] Nenhum fechamento aprovado. "
            "O grafo permanece aberto."
        )

    print(
        f"[INFO] {len(selected_idx)}/{n} nuvens usadas no registro."
    )

    return pose_graph, pcds, selected_idx


def point_to_point(
    save_path,
    voxel_size=1.5,
    threshold_coarse=20.0,
    threshold_fine=10.0,
    fitness_min=0.3000,
    max_gap_deg=40.0,
    loop_closure=True,
    min_gap_deg=30.0
):
    if not 0.0 <= min_gap_deg < 180.0:
        raise ValueError(
            "min_gap_deg deve estar entre 0 e 180 graus."
        )

    begin = time.perf_counter()

    paths, angles = load_angles_and_paths(save_path)

    if len(paths) < 2:
        print("[WARN] Precisa de pelo menos 2 arquivos.")
        return None, None, None

    pose_graph, pcds, sel_idx = select_and_build_pose_graph(
        paths,
        angles,
        threshold_coarse,
        threshold_fine,
        voxel_size=voxel_size,
        fitness_min=fitness_min,
        max_gap_deg=max_gap_deg,
        loop_closure=loop_closure
    )

    if pose_graph is None:
        return None, None, None

    print(
        f"[TEMPO] Leitura, preparo e registro: "
        f"{time.perf_counter() - begin:.2f}s"
    )

    tempo = time.perf_counter()

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=threshold_fine,
        edge_prune_threshold=0.25,
        reference_node=0
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    print(
        f"[TEMPO] Otimizacao do grafo: "
        f"{time.perf_counter() - tempo:.2f}s"
    )

    tempo = time.perf_counter()

    angles_sel = np.unwrap(
        np.asarray(
            [angles[idx] for idx in sel_idx],
            dtype=float
        ),
        period=360.0
    )

    volta_completa = any(
        edge.source_node_id == len(pcds) - 1
        and edge.target_node_id == 0
        and edge.uncertain
        for edge in pose_graph.edges
    )

    # Selecao exclusiva para a composicao final.
    # Cada posicao continua associada ao seu no original no grafo.
    fusion_pos = [0]

    for pos in range(1, len(pcds)):
        gap = angles_sel[pos] - angles_sel[fusion_pos[-1]]

        if gap + 1e-6 < min_gap_deg:
            continue

        # Em uma volta fechada, evita uma contribuicao final
        # muito proxima da primeira.
        if volta_completa:
            angle_gap = 360.0 - (
                angles_sel[pos] - angles_sel[0]
            )

            if angle_gap + 1e-6 < min_gap_deg:
                continue

        fusion_pos.append(pos)

    combined = o3d.geometry.PointCloud()

    print(
        f"\n[INFO] {len(sel_idx)} nuvens no registro; "
        f"{len(fusion_pos)} nuvens na composicao final."
    )

    for pos in fusion_pos:
        idx = sel_idx[pos]
        nome = paths[idx].name
        angulo = angles[idx]

        print(
            f"[COMPOSICAO] frame idx={idx} | "
            f"arquivo={nome} | angulo={angulo:.2f} graus"
        )

        pcd_copy = copy.deepcopy(pcds[pos])
        pcd_copy.transform(pose_graph.nodes[pos].pose)
        combined += pcd_copy

    combined = combined.voxel_down_sample(voxel_size)

    print(
        f"[TEMPO] Composicao final: "
        f"{time.perf_counter() - tempo:.2f}s"
    )

    end = time.perf_counter() - begin
    print(f"[TEMPO] Total: {end:.2f}s")

    # sel_idx continua correspondendo aos nos do grafo.
    return combined, pose_graph, sel_idx