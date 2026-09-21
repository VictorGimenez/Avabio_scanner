"""Cortes locais adaptativos. Dependencias: numpy e scipy.

Reutiliza _slice_mesh e _slice_mesh_oriented do projeto, recebidos como
argumentos. Seus casos degenerados (faces coplanares, por exemplo) continuam
dependendo dessas funcoes. Nao detecta punho nem atribui marcos anatomicos.
"""
import numpy as np
from scipy.spatial import KDTree, ConvexHull, QhullError


def _components(segments, tol):
    """Conectividade por endpoints em 3D; preserva os segmentos originais."""
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if len(s) == 0:
        return []
    s = s[np.linalg.norm(s[:, 1] - s[:, 0], axis=1) > tol]
    if len(s) == 0:
        return []
    parent = np.arange(len(s))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in KDTree(s.reshape(-1, 3)).query_pairs(tol):
        a, b = find(a // 2), find(b // 2)
        parent[a] = b
    groups = {}
    for i in range(len(s)):
        groups.setdefault(find(i), []).append(i)
    return [s[ids] for ids in groups.values() if len(ids) >= 3]


def _center(s):
    # Centro do contorno ponderado por comprimento. Nao e centroide de area.
    # Subdividir um segmento nao muda esse centro.
    weights = np.linalg.norm(s[:, 1] - s[:, 0], axis=1)
    return np.average(s.mean(axis=1), axis=0, weights=weights)


def _horizontal_components(segments, axis, clusterer):
    """Usa o agrupador original em 2D e recupera seus segmentos em 3D.

    clusterer recebe pares de endpoints 2D e retorna listas de endpoints
    originais agrupados, como _extract_slice_clusters do projeto.
    Nao deve arredondar, simplificar ou deslocar os pontos retornados.
    """
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if len(s) == 0:
        return []
    xy = np.delete(s, axis, axis=2)
    clusters = clusterer([(a, b) for a, b in xy])
    # Associa endpoints por coordenada exata, pois o agrupador original
    # somente seleciona pontos. Nao une componentes proximos nesta etapa.
    groups = []
    for cluster in clusters:
        members = {tuple(p) for p in np.asarray(cluster)}
        mask = [tuple(a) in members and tuple(b) in members for a, b in xy]
        selected = s[np.asarray(mask, dtype=bool)]
        selected = selected[np.linalg.norm(selected[:, 1] - selected[:, 0], axis=1) > 0]
        if len(selected):
            groups.append(selected)
    return groups


def _basis(normal):
    ref = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def adaptive_arm_slices(
    verts, tris, h_start, h_end, axis, side,
    slice_horizontal, slice_oriented,
    n_slices=50, window=5, threshold_on=0.15, threshold_off=0.10,
    lateral_axis=None, seed_center=None, join_tol=None, max_center_step=None,
    horizontal_clusterer=None,
):
    """Retorna um registro por altura; cortes invalidos tem valid=False.

    side='left' significa menor coordenada lateral; 'right', maior. Isso
    depende da orientacao da malha e nao significa esquerda anatomica.
    Sondagens horizontais sem exatamente 3 componentes sao descartadas.
    Como no classificador original, o maior grupo por numero de segmentos
    e considerado tronco; os outros dois sao os bracos, ordenados lateralmente.
    seed_center e opcional e so seleciona um dos bracos na primeira sondagem
    aceita. Nao dispensa o requisito de 3 componentes.

    window: janela impar para regressao local dos centros horizontais.
    threshold_on/off: deslocamento lateral / deslocamento vertical.
    join_tol: tolerancia absoluta para unir endpoints; nao e raio do braco.
       horizontal_clusterer: passe seu agrupador original para preservar os
    clusters horizontais. Ex.: lambda s: _extract_slice_clusters(s, 0.02).
    Recebe segmentos projetados nos dois eixos nao verticais. Se fornecido,
    join_tol e usado apenas no agrupamento dos cortes inclinados.
    hull_2d, hull_3d, perimeter_hull e perimeter_segments.
    A origem e o centro da sondagem horizontal, nao um centroide refinado.
    """
    verts = np.asarray(verts, dtype=float)
    tris = np.asarray(tris, dtype=int)
    if axis not in (0, 1, 2) or side not in ('left', 'right'):
        raise ValueError('axis deve ser 0, 1 ou 2; side deve ser left ou right.')
    if n_slices < 3 or window < 3 or window % 2 != 1 or h_start == h_end:
        raise ValueError('Use n_slices>=3, window impar>=3 e alturas distintas.')
    if not 0 <= threshold_off < threshold_on:
        raise ValueError('Exige 0 <= threshold_off < threshold_on.')
    axes = [a for a in range(3) if a != axis]
    lateral_axis = axes[0] if lateral_axis is None else lateral_axis
    if lateral_axis not in axes:
        raise ValueError('lateral_axis deve ser diferente do eixo vertical.')
    scale = np.linalg.norm(np.ptp(verts, axis=0))
    if scale <= 0:
        raise ValueError('Malha sem extensao espacial.')
    tol = scale * 1e-7 if join_tol is None else float(join_tol)
    if tol <= 0:
        raise ValueError('join_tol deve ser positivo.')
    heights = np.linspace(h_start, h_end, n_slices)
    step = abs(heights[1] - heights[0])
    max_step = 3 * step if max_center_step is None else float(max_center_step)
    if max_step <= 0:
        raise ValueError('max_center_step deve ser positivo.')
    vertical = np.eye(3)[axis]

    # 1. Sondagem horizontal para obter uma trajetoria de centros comparaveis.
    centers = np.full((n_slices, 3), np.nan)
    horizontal_groups = [None] * n_slices
    reasons = [None] * n_slices
    component_counts = np.zeros(n_slices, dtype=int)
    seed_used = False
    for i, h in enumerate(heights):
        segs, _ = slice_horizontal(verts, tris, float(h), axis)
        groups = (_horizontal_components(segs, axis, horizontal_clusterer)
                  if horizontal_clusterer is not None else _components(segs, tol))
        component_counts[i] = len(groups)
        if len(groups) != 3:
            reasons[i] = f'Sondagem horizontal com {len(groups)} componentes; esperado: 3.'
            continue
        cc = np.array([_center(g) for g in groups])
        trunk = max(range(3), key=lambda k: len(groups[k]))
        arms = sorted((k for k in range(3) if k != trunk),
                      key=lambda k: cc[k, lateral_axis])
        k = arms[0 if side == 'left' else 1]
        if seed_center is not None and not seed_used:
            target = np.asarray(seed_center, dtype=float)
            k = min(arms, key=lambda a: np.linalg.norm(cc[a, axes] - target[axes]))
            if np.linalg.norm(cc[k, axes] - target[axes]) > max_step:
                reasons[i] = 'Bracos distantes do seed_center informado.'
                continue
            seed_used = True
        elif i > 0 and horizontal_groups[i - 1] is not None:
            if np.linalg.norm(cc[k, axes] - centers[i - 1, axes]) > max_step:
                reasons[i] = 'Salto entre centros: janela local interrompida.'
                continue
        centers[i] = cc[k]
        horizontal_groups[i] = groups[k]

    results = []
    tilted = False
    # 2. Direcao LOCAL: dC/dh; a normal acompanha o eixo local do braco.
    for i, h in enumerate(heights):
        result = {'height': float(h), 'valid': False,
                  'probe_components': int(component_counts[i])}
        if horizontal_groups[i] is None:
            tilted = False
            result['reason'] = reasons[i]
            results.append(result)
            continue
        # A janela termina na primeira lacuna de cada lado.
        lo, hi = i, i + 1
        while lo > max(0, i - window // 2) and horizontal_groups[lo - 1] is not None:
            lo -= 1
        while hi < min(n_slices, i + window // 2 + 1) and horizontal_groups[hi] is not None:
            hi += 1
        if hi - lo < 2:
            tilted = False
            result['reason'] = 'Fatia isolada: sem vizinha valida para estimar inclinacao.'
            results.append(result)
            continue
        local_h = heights[lo:hi]
        dh = local_h - local_h.mean()
        local_c = centers[lo:hi]
        tangent = (dh[:, None] * (local_c - local_c.mean(axis=0))).sum(axis=0)
        tangent /= np.dot(dh, dh)
        tangent[axis] = 1.0
        ratio = float(np.linalg.norm(tangent[axes]))
        # Histerese evita alternancia por ruido em torno de um unico limiar.
        if tilted:
            tilted = ratio > threshold_off
        else:
            tilted = ratio > threshold_on
        normal = tangent / np.linalg.norm(tangent) if tilted else vertical.copy()
        origin = centers[i].copy()
        result.update(origin=origin, normal=normal, tilted=tilted, drift_ratio=ratio)

        # 3. Refaz a intersecao com a malha; nao gira o contorno anterior.
        if tilted:
            segs, _ = slice_oriented(verts, tris, origin, normal)
            groups = _components(segs, tol)
            if not groups:
                result['reason'] = 'Plano inclinado sem contorno.'
                results.append(result)
                continue
            distance = [np.linalg.norm(_center(g) - origin) for g in groups]
            k = int(np.argmin(distance))
            # Rejeita um componente distante em vez de escolher outro membro.
            probe = horizontal_groups[i].reshape(-1, 3)
            radius = np.max(np.linalg.norm(probe - origin, axis=1))
            if distance[k] > max(radius, 10 * tol):
                result['reason'] = 'Contorno inclinado distante do braco sondado.'
                results.append(result)
                continue
            selected = groups[k]
        else:
            selected = horizontal_groups[i]

        u, v = _basis(normal)
        points = selected.reshape(-1, 3)
        xy = np.column_stack(((points - origin) @ u, (points - origin) @ v))
        try:
            hull = ConvexHull(xy)
        except QhullError:
            result['reason'] = 'Contorno degenerado.'
            results.append(result)
            continue
        hp = xy[hull.vertices]
        result.update(
            valid=True, u=u, v=v, segments_3d=selected,
            hull_2d=hp, hull_3d=origin + hp[:, :1] * u + hp[:, 1:] * v,
            perimeter_hull=float(np.linalg.norm(np.roll(hp, -1, axis=0) - hp, axis=1).sum()),
            perimeter_segments=float(np.linalg.norm(selected[:, 1] - selected[:, 0], axis=1).sum()),
        )
        results.append(result)
    return results