# Copyright 2026 Avanutri.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#                                           /\\
#                                                      /  \\               
#                                                     /    /               
#                                                    /    /                
#                     __                            /    /  __              
#                    /  \\                         /    / /   \\             
#                   /    \\    /\\                /    / /     \\            
#                  /      \\  /  \\              /    / /       \\           
#                 /   /\\   \\ \\   \\          /    / /   /\\   \\          
#                /   /  \\   \\ \\   \\        /    / /   /  \\   \\         
#               /   /    \\   \\ \\   \\      /    / /   /    \\   \\        
#              /   /      \\   \\ \\   \\    /    / /   /      \\   \\       
#             /   /        \\   \\ \\   \\  /    / /   /        \\   \\     
#            /   /          \\   \\ \\   \\/    / /   /          \\   \\    
#           /   /            \\   \\ \\       / /   /            \\   \\    
#           \\  /              \\  /  \\     /  \\  /              \\  /    
#            \\/                \\/    \\___/    \\/                \\/     
                                                                            
#        A D V A N C E D    R E C O V E R Y   F O R   A T H L E T E S       

# ###########################################################################################################################
# ###################################################   Main_Web_Version  ###################################################
# ###########################################################################################################################

# Versão: 1.0.0

# Módulo interno a ser incluso na aplicação de bioimpedância:

# Escopo da Prova de Conceito (Tarefas principais) proposto por Rodrigo Santana:
# • Captura de dados: Utilizar o SDK da Orbbec (ou outra câmera de profundidade similar, como Intel RealSense) para capturar a nuvem 
# de pontos 3D de um objeto simples (ex: um cilindro) e, posteriormente, de um membro humano (ex: braço ou perna).
# • Processamento da nuvem de pontos: Limpar e processar a nuvem de pontos capturada para remover ruídos e isolar o objeto de 
# interesse.
# • Cálculo de circunferência: Desenvolver um algoritmo em Python para "fatiar" o modelo 3D em um ponto específico e calcular 
# a circunferência dessa seção transversal.


# Desenvolvedor: Victor Borghi Gimenez - Email: victor.gimenez@ufabc.edu.br, victor.gimenez@gmx.es
# Cliente: Rodrigo Izidoro Santana


import uvicorn, time, os, sys, threading, tempfile, traceback
os.environ['MKL_NUM_THREADS'] = '14'      # número de cores
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import cv2 as cv
import copy
import numpy as np
import open3d as o3d
import tempfile
import psycopg2

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
from queue import Queue

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import DeclarativeBase, Session
from pydantic import BaseModel
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from src.depth_color_overlay import display_camera, apply_thresh_filter
from src.pipeline_settings import CAPTURE_PROFILES, generate_pipeline
from src.file_handling import check_and_mkdir
from src.data_acquisition import rotate_complete
from src.generate_data import frames_pipeline, process_npz
from src.align_frames import point_to_point, AlinhamentoRejeitado
from src.generate_mesh import generate_mesh
from src.gui import clear_screen, print_header, select_option, set_window_attributes, monitor_information, display_content, focus_main_window
from src.measuring_mesh import measuring_mesh, align_to_z_and_convert_to_centimeters, display_mesh_info
from src.Turntable import Turntable

#Diretório raiz para salvar informações dos registros de nuvens de pontos
dir_name = "scans"
save_points_dir = os.path.join(os.getcwd(), dir_name)

#Misc...
global current_date, t_rotate, camera_in_vertical, is_rotating, countdown_active, pac, t1, measurement_pronto, pcd_pronto, ply_pronto, origin_reached, path, nome, medidas_global, fmin, voxel_size, max_gap_deg, turntable_attached
vertical_axis  = 2
current_date   = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamps     = {}
begin_t        = [0]
save_queue     = Queue()
origin_reached = threading.Event()
frame_queue    = Queue(maxsize=2)

DB_NAME = "avanutri"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

#Conexão com bd postgres
conn = psycopg2.connect(dbname="postgres",user=DB_USER,password=DB_PASSWORD,host=DB_HOST,port=DB_PORT)

conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

cursor = conn.cursor()

cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s",(DB_NAME,))

exists = cursor.fetchone()

if not exists:
    cursor.execute(
        f"CREATE DATABASE {DB_NAME}"
    )
    print(f"Banco '{DB_NAME}' criado!")
else:
    print(f"Banco '{DB_NAME}' já existe.")

# Fecha aqui
cursor.close()
conn.close()

# database connection
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?client_encoding=utf8" #Windows
# engine = create_engine(DATABASE_URL, connect_args={"client_encoding": "utf8"})
engine = create_engine(DATABASE_URL)

MIN_DEPTH = 900
MAX_DEPTH = 1800
COLOR_TO_REMOVE = np.array([128, 0, 0], dtype=np.uint8)

#For testing purposes
scanning_performed = False
turntable_attached = True

#Para efeitos de teste/debug quando a base giratória não estiver ligada
point_clouds_generated = False
point_clouds_aligned = False
mesh_generated = False

turntable_event = threading.Event()
countdown_active = False
t_rotate = None
path = None
camera_in_vertical = True
is_rotating = False

point_cloud = None
mesh = None
line_sets = []
medidas_global = {}

running = False
error = None
ply_pronto = False
pcd_pronto = False
measurement_pronto = False

fmin = 0.5
voxel_size = 4
print(f"voxel_size:{voxel_size}")
max_gap_deg=40.0

capture_thread = None
stop_capture = threading.Event()

_latest_rgb      = None
_latest_filtered = None
_frame_lock      = threading.Lock()


LANDMARKS_MAP = {
    "Pescoço":               "pescoco",
    "Ombros Superior":       "omb_sup",
    "Ombros Médio":          "omb_med",
    "Ombros Inferior":       "omb_inf",
    "Busto Superior":        "bus_sup",
    "Busto":                 "bus_med",
    "Busto Inferior":        "bus_inf",
    "Cintura Superior":      "cin_sup",
    "Cintura":               "cin_med",
    "Cintura Inferior":      "cin_inf",
    "Quadril Superior":      "quadl_sup",
    "Quadril":               "quadl_med",
    "Quadril Inferior":      "quadl_inf",

    "Bíceps Longo Esq.":     "bic_longo_esq",
    "Bíceps Curto Esq.":     "bic_curto_esq",
    "Braquial Esq.":         "braq_esq",
    "Braquiorradial Esq.":   "braquiorr_esq",
    "Cotovelo Esq.":         "cot_esq",
    "Antebraço Esq.":        "ant_esq",
    "Pulso Esq.":            "puls_esq",

    "Bíceps Longo Dir.":     "bic_longo_dir",
    "Bíceps Curto Dir.":     "bic_curto_dir",
    "Braquial Dir.":         "braq_dir",
    "Braquiorradial Dir.":   "braquiorr_dir",
    "Cotovelo Dir.":         "cot_dir",
    "Antebraço Dir.":        "ant_dir",
    "Pulso Dir.":            "puls_dir",

    "Pectíneo Esq.":             "pect_esq",
    "Pectíneo-Quadríceps Esq.":  "pect_quad_esq",
    "Quadríceps Esq.":           "quad_esq",
    'Quadríceps-Coxa Esq.':      "quad_coxa_esq",
    "Coxa Esq.":                 "coxa_esq",
    "Coxa-Joelho Esq.":          "coxa_joelho_esq",
    "Joelho Esq.":               "joelho_esq",
    "Gastrocnêmio Esq.":         "gast_esq",
    "Sóleo Esq.":                "soleo_esq",     
    "Panturrilha Esq.":          "panturrilha_esq",
    "Tornozelo Esq.":            "tornozelo_esq",

    "Pectíneo Dir.":             "pect_dir",
    "Pectíneo-Quadríceps Dir.":  "pect_quad_dir",
    "Quadríceps Dir.":           "quad_dir",
    'Quadríceps-Coxa Dir.':      "quad_coxa_dir",
    "Coxa Dir.":                 "coxa_dir",
    "Coxa-Joelho Dir.":          "coxa_joelho_dir",
    "Joelho Dir.":               "joelho_dir",
    "Gastrocnêmio Dir.":         "gast_dir",
    "Sóleo Dir.":                "soleo_dir",     
    "Panturrilha Dir.":          "panturrilha_dir",
    "Tornozelo Dir.":            "tornozelo_dir",
}

class Base(DeclarativeBase):
    pass

class Paciente(Base):
    __tablename__ = "pacientes"
    id     = Column(Integer, primary_key=True, index=True)
    nome   = Column(String(100))
    altura = Column(Float)
    peso   = Column(Float)
    idade  = Column(Integer)

class PacienteIn(BaseModel):
    nome:   str
    altura: float
    peso:   float
    idade:  int

class Medicao(Base):
    __tablename__ = "medicoes"
    id                   = Column(Integer, primary_key=True, index=True)
    paciente_id          = Column(Integer, ForeignKey("pacientes.id"))
    data                 = Column(DateTime, default=datetime.utcnow)                    # frame_queue = Queue(maxsize=2)

    # tronco
    pescoco              = Column(Float, nullable=True)
    ombros               = Column(Float, nullable=True)
    busto_superior       = Column(Float, nullable=True)
    busto_medio          = Column(Float, nullable=True)
    busto_inferior       = Column(Float, nullable=True)
    cintura_superior     = Column(Float, nullable=True)
    cintura_medio        = Column(Float, nullable=True)
    cintura_inferior     = Column(Float, nullable=True)
    quadril_superior     = Column(Float, nullable=True)
    quadril_medio        = Column(Float, nullable=True)
    quadril_inferior     = Column(Float, nullable=True)
    # braço esquerdo
    biceps_longo_esq     = Column(Float, nullable=True)
    biceps_curto_esq     = Column(Float, nullable=True)
    braquial_esq         = Column(Float, nullable=True)
    braquiorradial_esq   = Column(Float, nullable=True)
    cotovelo_esq         = Column(Float, nullable=True)
    antebraco_esq        = Column(Float, nullable=True)
    pulso_esq            = Column(Float, nullable=True)
    # braço direito
    biceps_longo_dir     = Column(Float, nullable=True)
    biceps_curto_dir     = Column(Float, nullable=True)
    braquial_dir         = Column(Float, nullable=True)
    braquiorradial_dir   = Column(Float, nullable=True)
    cotovelo_dir         = Column(Float, nullable=True)
    antebraco_dir        = Column(Float, nullable=True)
    pulso_dir            = Column(Float, nullable=True)
    # perna esquerda
    pectinio_esq         = Column(Float, nullable=True)
    quadriceps_esq       = Column(Float, nullable=True)
    coxa_esq             = Column(Float, nullable=True)
    joelho_esq           = Column(Float, nullable=True)
    panturrilha_esq      = Column(Float, nullable=True)
    tibia_esq            = Column(Float, nullable=True)
    tornozelo_esq        = Column(Float, nullable=True)
    # perna direita
    pectinio_dir         = Column(Float, nullable=True)
    quadriceps_dir       = Column(Float, nullable=True)
    coxa_dir             = Column(Float, nullable=True)
    joelho_dir           = Column(Float, nullable=True)
    panturrilha_dir      = Column(Float, nullable=True)
    tibia_dir            = Column(Float, nullable=True)
    tornozelo_dir        = Column(Float, nullable=True)

# class MedicaoIn(BaseModel):

Base.metadata.create_all(bind=engine)

app = FastAPI()

def serializar_linesets(line_sets_o3d: list, nomes: list = None) -> list:
    """
        Método responsável pela listagem de dicionários de landmarks, o seguinte dicionário retornado é enviado ao método: servir_lineset()
        que envia a lista estruturada para a rota /lineset que carrega seu conteúdo na última tela de geração de medidas.
    
        Argumentos de entrada:
            line_sets_o3d: Lista formada por tuplas com cada tupla composta por um par: ('Nome do landmark', Lineset) onde
            Lineset é uma instância de um objeto do tipo: <class 'open3d.cpu.pybind.geometry.LineSet'> composto por conjuntos
            de pontos, linhas e cores ex:
            
            [('Bíceps Longo Esq.', LineSet with 197 lines.), 
            ('Bíceps Curto Esq.', LineSet with 208 lines.), ineSet with 167 lines.), ..., ('Pulso Dir.', LineSet with 78 lines.)]
            
            nomes: Lista com os nomes das medidas encontradas.

        Retorno:
            Lista de dicionários onde cada dicionário é formado por 4 chaves que são os atributos (pontos, linhas, cores e nome) e os valores
            sendo 'points' (pontos) -> numpy.array, 'lines' (linhas) -> list, 'colors' (cores) -> list e 'nome': str. Considere i como sendo um número 
            inteiro qualquer:
            resultado[i]['points']: [[-1.8865334140157328, 25.19831603732486, 20.54117232181245],
                        [-1.6839799385215515, 25.146851563514637, 20.566356059433353], ...,  [-2.1347364399862974, 25.343976906394936, 20.550267093681747]]
            resultado[i]['lines']: [[0, 1], [1, 2],..., [195, 196], [196, 0]]
            resultado[i]['colors']:[[0.0, 0.2, 0.6], [0.0, 0.2, 0.6],..., [0.0, 0.2, 0.6]]
            resultado[i]['nome']: 'Bíceps Longo Esq.'
            
            O tamanho (len) de cada dicionário da lista são 4 itens como já mencionados acima e o tamanho len da lista resultado é a quantidade
            de landmarks como mencionados em LANDMARK_STYLE dicionário pertinente ao arquivo measuring_mesh.py e em LANDMARKS_MAP nesse mesmo arquivo.
    """
    resultado = []
    for i, (_, ls) in enumerate(line_sets_o3d):
        item = {
            "points": np.asarray(ls.points).tolist(),
            "lines": np.asarray(ls.lines).tolist(),
            "colors": np.asarray(ls.colors).tolist(),
        }
        if nomes is not None and i < len(nomes):
            item["nome"] = nomes[i]
        resultado.append(item)
    return resultado

def capture_loop():
    """Parse a string representing a color to an RGBA tuple.
    
        Possible formats for the input string include:
    
        * named color, see `COLORS_BY_NAME`
        * hex short eg. `<prefix>fff` (prefix can be `#`, `0x` or nothing)
        * hex long eg. `<prefix>ffffff` (prefix can be `#`, `0x` or nothing)
        * `rgb(<r>, <g>, <b>)`
        * `rgba(<r>, <g>, <b>, <a>)`
    
        Args:
            value: A string representing a color.
    
        Returns:
            An `RGBA` tuple parsed from the input string.
    
        Raises:
            ValueError: If the input string cannot be parsed to an RGBA tuple.
    """

    global _latest_rgb, _latest_filtered, _color_alert, depth_frame, pipeline, has_color_sensor, sdk_filters
    global align_filter, pcd_pronto, ply_pronto, measurement_pronto, origin_reached, path, line_sets
    global mesh_final, mesh_smoothed, paciente_id, medidas_global, t_rotate, point_cloud, running
    global error, point_cloud, mesh, line_sets, medidas_global, t1

    try:
        pipeline, has_color_sensor, sdk_filters, align_filter, _ = generate_pipeline()
        if pipeline is None:
            return
    except Exception as e:
        print(f"[ERRO] Pipeline: {e}")
        return

    running = True
    pcd_pronto = False
    ply_pronto = False
    measurement_pronto = False
    error = None
    line_sets = []
    medidas = {}
    
    try:
        for frame, depth_frame, frames in display_camera(pipeline, has_color_sensor, sdk_filters, align_filter):
            if frame is None or depth_frame is None:
                continue

            # frame bruto
            raw = frame.copy()

            # frame com threshold + RGB
            filtered, color_detected = apply_thresh_filter(frame, frames, align_filter, COLOR_TO_REMOVE, MIN_DEPTH, MAX_DEPTH, ALERT_COLOR=(0, 0, 125))

            if camera_in_vertical:
                raw      = cv.rotate(raw,      cv.ROTATE_90_CLOCKWISE)
                filtered = cv.rotate(filtered, cv.ROTATE_90_CLOCKWISE)

            with _frame_lock:
                _latest_rgb      = raw
                _latest_filtered = filtered
                _color_alert     = color_detected

            is_rotating = turntable_event.is_set()

            if is_rotating and not origin_reached.is_set():
                if not frame_queue.full():
                    print("if not frame_queue.full():")
                    frame_queue.put(frames)
            if origin_reached.is_set():
                with Session(engine) as session:
                    print("Ply pronto")
                    pac = session.get(Paciente, paciente_id)
                    os.chdir(save_points_dir)
                    path = os.path.join(os.path.join(os.getcwd(), save_points_dir), pac.nome + "_" + current_date)

                    process_npz(path)

                    point_cloud, _, _ = point_to_point(
                        path,
                        voxel_size=voxel_size,
                        threshold_coarse=20.0,
                        threshold_fine=10.0,
                        fitness_min=0.3000,
                        max_gap_deg=40.0,
                        loop_closure=True,
                        min_gap_deg=30.0
                    )

                    if point_cloud is None:
                        print("[ERRO] Alinhamento falhou. A malha não será gerada.")
                        _reiniciar_processo()
                        return
                    else:
                        o3d.io.write_point_cloud(f"{path}/pcd_with_fmin{fmin}_and_voxel_size{voxel_size}.ply", point_cloud)
                        print(f"[DONE] Nuvem final salva com {len(point_cloud.points)} pontos")
                        
                    pcd_pronto = True

                    #Usado para debugging... (descomentar para uso)
                    # point_cloud = o3d.io.read_point_cloud(os.path.join(path,f"{pac.nome}_reconstructed_pcd_with_p2p_and_p2p_voxel1_and3_non_complete_6to84_1_12_1.ply"))

                    _, _, mesh_smoothed = generate_mesh(path, pac.nome, point_cloud, voxel_size)

                    # if mesh_final is not None:
                    #     o3d.io.write_triangle_mesh(os.path.join(path, f"{pac.nome}_mesh_with_fmin{fmin}_and_voxel_size{voxel_size}.ply"), mesh_final)
                    #     print("[WARN] Final mesh generated")

                    # if mesh_without_holes is not None:
                    #     o3d.io.write_triangle_mesh(os.path.join(path, f"{pac.nome}_mesh_without_holes_with_fmin{fmin}_and_voxel_size{voxel_size}.ply"), mesh_without_holes)
                    #     print("[WARN] Final mesh without holes generated")

                    if mesh_smoothed is not None:
                        o3d.io.write_triangle_mesh(os.path.join(path, f"{pac.nome}_smoothed_mesh_with_fmin{fmin}_and_voxel_size{voxel_size}.ply"), mesh_smoothed)
                        print("[WARN] Final mesh smoothed was generated")

                    #Usado para debugging... (descomentar para uso)
                    #mesh_smoothed = o3d.io.read_triangle_mesh(os.path.join(path, f"{pac.nome}_smoothed_mesh_with_fmin{fmin}_and_voxel_size{voxel_size}.ply"))

                    ply_pronto = True
                    mesh = mesh_smoothed

                    if len(mesh.vertices) == 0:
                        sys.exit("[Erro] Nenhum vértice encontrado.")

                    # opcional: garantir normais
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()

                    #Pre-processing mesh
                    mesh.remove_duplicated_vertices()
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_non_manifold_edges()

                    mesh, vertices = align_to_z_and_convert_to_centimeters(mesh)
                    pts = np.asarray(mesh.vertices)

                    print(f"[MESH salvo] X: {pts[:,0].min():.3f} a {pts[:,0].max():.3f}")
                    print(f"[MESH salvo] Y: {pts[:,1].min():.3f} a {pts[:,1].max():.3f}")
                    print(f"[MESH salvo] Z: {pts[:,2].min():.3f} a {pts[:,2].max():.3f}")

                    mesh.orient_triangles()
                    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
                    mesh.compute_vertex_normals()

                    o3d.io.write_triangle_mesh(os.path.join(path, f"aligned_mesh_for_debug_open_arms_{pac.nome}.ply"), mesh)

                    line_sets_o3d, medidas = measuring_mesh(mesh, vertical_axis, pts, pac.nome + "_" + current_date, scanning_performed)
                    salvar_medicao(paciente_id, medidas)
                    nomes_medidas = list(medidas.keys())

                    line_sets = serializar_linesets(line_sets_o3d, nomes_medidas)
                    medidas = {k: float(v) for k, v in medidas.items()}

                    medidas_global = medidas

                    measurement_pronto = True
                    return
    except Exception as e:
        import traceback
        print(f"[ERRO] Captura: {e}")
        traceback.print_exc()
        error = str(e)
    finally:
        print("Finalizado!")
        if turntable_attached:
            t1.close_turntable_connection()
        if t_rotate is not None and t_rotate.is_alive():
            print("[WARN] Waiting capture be finished")
        pipeline.stop()
        print("Pipeline finished")
        running = False


@app.on_event("startup")
def startup():
    global t1, capture_thread
    print("[WARN] Inicializando base giratória")
    if turntable_attached:
        # Base giratória (Abaixo descomentar a linha de acordo caso você tenha ou não uma base disponível e conectada no PC)
        t1 = Turntable()  #Essa linha executa a base giratória diretamente
        t1.open_turntable_connection()  #Se fora do modo simulação descomentar essa linha
    # Modo simulação
    # t1 = Turntable(arquivo_simulacao="./turntable_simulation3.txt")   #Essa linha executa um arquivo que simula os estados de rotação da base giratória BKL sem a necessidade da base
    # t1.send_message(b'\xFF\xFF\xD2\x04\x01\xD7\xFF\xFE')   #setting gear 2 (medium rotation)
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()


def gen_rgb_frames():
    while True:
        with _frame_lock:
            frame = _latest_rgb
        if frame is None:
            continue
        ret, buf = cv.imencode(".jpg", frame)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")


def gen_rgb_filtered_frames():
    while True:
        with _frame_lock:
            frame = _latest_filtered
        if frame is None:
            continue
        ret, buf = cv.imencode(".jpg", frame)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")


@app.get("/color_status")
def get_status():
    with _frame_lock:
        alerta = _color_alert
    return {"alert": alerta}


@app.get("/origin_reached")
def estado():
    global measurement_pronto, pcd_pronto, ply_pronto
    return {"measurement_pronto": measurement_pronto, "pcd_pronto": pcd_pronto, "ply_pronto": ply_pronto}


@app.get("/paciente_nome")
def paciente_nome():
    return {"paciente_nome": pac.nome}


@app.get("/data")
def data():
    return {"data": current_date}


@app.get("/stream/rgb")
def stream_rgb():
    return StreamingResponse(gen_rgb_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/filtered_rgb")
def stream_filtered_rgb():
    return StreamingResponse(gen_rgb_filtered_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/medidas")
def servir_medidas():
    return JSONResponse(medidas_global)


def _reiniciar_processo():
    time.sleep(0.3)
    print("[RESET] encerrando processo para reinício controlado pelo launcher...")
    os._exit(99)  # precisa bater com RESTART_CODE do launcher.py
    # script = os.path.abspath(__file__)
    # os.chdir(os.path.dirname(script))
    # os.execv(sys.executable, [sys.executable, script])


@app.post("/reset_estado")
def reset_estado():
    threading.Thread(target=_reiniciar_processo, daemon=False).start()
    return JSONResponse({"status": "reiniciando"})


@app.get("/ply")
def servir_ply():
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = tmp.name
 
    if (ply_pronto or measurement_pronto) and mesh is not None:
        o3d.io.write_triangle_mesh(tmp_path, mesh)
    elif pcd_pronto and point_cloud is not None:
        o3d.io.write_point_cloud(tmp_path, point_cloud)
    else:
        os.unlink(tmp_path)
        return JSONResponse({"erro": "nenhum .ply disponível ainda"}, status_code=404)
 
    def iterfile():
        with open(tmp_path, "rb") as f:
            yield from f
        os.unlink(tmp_path)
 
    return StreamingResponse(iterfile(), media_type="application/octet-stream")


@app.get("/lineset")
def servir_lineset():
    return JSONResponse({"linesets": line_sets})
 
 
@app.get("/debug_centro")
def debug_centro():
    pts_mesh = np.asarray(mesh.vertices)
    centro_mesh = pts_mesh.mean(axis=0)

    centros_ls = []
    for line_set in line_sets:
        if isinstance(line_set, o3d.geometry.LineSet):
            pts = np.asarray(line_set.points)
            centros_ls.append(pts.mean(axis=0).tolist())

    return {
        "centro_mesh": centro_mesh.tolist(),
        "centros_linesets": centros_ls
    }

def salvar_medicao(paciente_id: int, medidas: dict):
    with Session(engine) as session:
        medicao = Medicao(paciente_id=paciente_id)

        for nome_origem, valor in medidas.items():
            coluna = LANDMARKS_MAP.get(nome_origem)
            if coluna is None:
                print(f"[WARN] chave '{nome_origem}' sem mapeamento, ignorada")
                continue
            setattr(medicao, coluna, float(valor))

        session.add(medicao)
        session.commit()
        session.refresh(medicao)
        print(f"[DB] Medição salva, id={medicao.id}, paciente_id={paciente_id}")
        return medicao.id
    
@app.post("/start")
def start(paciente: PacienteIn):
    print(f"[START] recebido: {paciente}")
    global countdown_active, paciente_id

    countdown_active = True

    with Session(engine) as session:
        pac = Paciente(
            nome   = paciente.nome,
            altura = paciente.altura,
            peso   = paciente.peso,
            idade  = paciente.idade,
        )

        session.add(pac)
        session.commit()
        session.refresh(pac)

        paciente_id = pac.id
        print(f"[DB] Paciente salvo: {paciente.nome}")
        print(f"[DB] ID gerado: {pac.id}")

    return {"id": paciente_id, "nome": pac.nome}


@app.post("/countdown_done")
def countdown_done():
    global countdown_active, current_date, t_rotate, measurement_pronto, path, paciente_id
    countdown_active = False
    if turntable_attached == False:
        print("Rotação encerrada")
        origin_reached.set()

    with Session(engine) as session:
        print("[WARN] Start rotation")
        pac = session.get(Paciente, paciente_id)
        while origin_reached.is_set() == False:
            if t_rotate is None or not t_rotate.is_alive():
                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)
                os.chdir(save_points_dir)
                curdir = check_and_mkdir(pac.nome + "_" + current_date)
                path = os.path.join(os.path.join(os.getcwd(), save_points_dir), curdir)
                turntable_event.set()

                begin_t[0] = time.time()

                t_rotate = threading.Thread(target=rotate_complete, args=(turntable_event, t1, timestamps, begin_t, origin_reached, path), daemon=False)
                t_get_frames = threading.Thread(target=frames_pipeline, args=(depth_frame, timestamps, begin_t, frame_queue, save_queue, save_points_dir, path, pipeline, has_color_sensor, origin_reached, align_filter, sdk_filters), daemon=False)
                t_rotate.start()
                t_get_frames.start()
            else:
                print("[thread] Threading already in action")
                break
    return {"status": "ok"}


@app.get("/rgb_feed")
def rgb_feed():
    def gerar():
        while True:
            with _frame_lock:
                frame = _latest_rgb
            if frame is None:
                time.sleep(0.03)
                continue
            ok, buf = cv.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(gerar(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/rgb_filtered_feed")
def rgb_filtered_feed():
    def gerar():
        while True:
            with _frame_lock:
                frame = _latest_filtered
            if frame is None:
                time.sleep(0.03)
                continue
            ok, buf = cv.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return StreamingResponse(gerar(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/visualizar", response_class=HTMLResponse)
def visualizar():
    return """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="utf-8">
        <title>Visualização 3D</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
    
            body {
                background: #0f1117;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                height: 100vh;
                overflow: hidden;
            }
    
            #canvas-container {
                flex: 1;
                position: relative;
            }
    
            .painel {
                width: 300px;
                background: #1a1d27;
                border-left: 1px solid #2a2d3a;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
    
            .painel-header {
                padding: 16px;
                border-bottom: 1px solid #2a2d3a;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
    
            .painel-header h2 {
                font-size: 0.95rem;
                color: #7eb8f7;
                letter-spacing: 1px;
            }
    
            .painel-header .subtitulo {
                font-size: 0.75rem;
                color: #666;
            }
    
            .btn-voltar {
                background: #2a2d3a;
                color: #e0e0e0;
                border: 1px solid #3a3d4a;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 0.85rem;
                cursor: pointer;
                transition: background 0.2s;
                text-align: center;
                text-decoration: none;
                display: block;
            }
    
            .btn-voltar:hover { background: #3a3d4a; }
    
            .aviso {
                padding: 12px;
                font-size: 0.8rem;
                color: #aaa;
                line-height: 1.5;
                border-bottom: 1px solid #2a2d3a;
            }
            .aviso strong { color: #7eb8f7; }
    
            .medidas-lista {
                flex: 1;
                overflow-y: auto;
                padding: 12px;
            }
    
            .medida-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 6px 8px;
                border-radius: 6px;
                margin-bottom: 4px;
                background: #0f1117;
            }
    
            .medida-item:hover { background: #2a2d3a; }
    
            .medida-nome { font-size: 0.78rem; color: #aaa; }
            .medida-valor { font-size: 0.85rem; font-weight: 600; color: #7eb8f7; white-space: nowrap; }
            .medida-nao-detectado { color: #555; font-style: italic; }

             /* destaque quando o mouse passa perto da linha correspondente em 3D */
            .medida-destacada {
                background: #2a2d3a !important;
                box-shadow: inset 3px 0 0 #7eb8f7;
            }
            .medida-destacada .medida-nome { color: #e0e0e0; }

            .secao-titulo {
                font-size: 0.7rem;
                color: #555;
                text-transform: uppercase;
                letter-spacing: 1px;
                padding: 8px 8px 4px;
                margin-top: 4px;
            }
    
            .medidas-lista::-webkit-scrollbar { width: 4px; }
            .medidas-lista::-webkit-scrollbar-track { background: #0f1117; }
            .medidas-lista::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 2px; }
    
            /* banner de nova etapa disponível */
            #banner-atualizar {
                position: absolute;
                top: 16px; left: 50%;
                transform: translateX(-50%);
                display: none;
                align-items: center; gap: 10px;
                background: #1a1d27;
                border: 1px solid #7eb8f7;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 0.8rem;
                z-index: 10;
            }
            #banner-atualizar.show { display: flex; }
        </style>
        <script type="importmap">
        {
            "imports": {
                "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
                "three/examples/jsm/loaders/PLYLoader.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/PLYLoader.js",
                "three/examples/jsm/controls/OrbitControls": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js",
                "three/examples/jsm/lines/LineSegmentsGeometry.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/lines/LineSegmentsGeometry.js",
                "three/examples/jsm/lines/LineMaterial.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/lines/LineMaterial.js",
                "three/examples/jsm/lines/LineSegments2.js": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/lines/LineSegments2.js"
            }
        }
        </script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="banner-atualizar">
                <span>Uma nova etapa foi concluída.</span>
            </div>
        </div>
    
        <div class="painel">
            <div class="painel-header">
                <h2 id="painel-titulo">Carregando…</h2>
                <span class="subtitulo" id="painel-subtitulo"></span>
                <a href="/" class="btn-voltar" id="btn-voltar">← Tentar novamente</a>
            </div>
    
            <div class="aviso" id="aviso" style="display:none;"></div>
            <div class="medidas-lista" id="medidas-lista"></div>
        </div>
    
        <script type="module">
            import * as THREE from 'three';
            import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
            import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
            import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
            import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
            import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';

            const container = document.getElementById("canvas-container");
            let camera, scene, renderer, controls;
            let grupo = null;
            let animating = true;
            let estagioAtual = null; // "pcd" | "mesh" | "medidas"

            const SECOES = {
                "Pescoço":["Pescoço"],
                "Tronco": ["Ombros Superior","Ombros Médio", "Ombros Inferior",
                           "Busto Superior","Busto Médio","Busto Inferior",
                           "Cintura Superior","Cintura Médio","Cintura Inferior",
                           "Quadril Superior","Quadril Médio","Quadril Inferior"],
                "Braço Esq.": ["Bíceps Longo Esq.","Bíceps Curto Esq.","Braquial Esq.",
                               "Braquiorradial Esq.","Cotovelo Esq.","Antebraço Esq.","Pulso Esq."],
                "Braço Dir.": ["Bíceps Longo Dir.","Bíceps Curto Dir.","Braquial Dir.",
                               "Braquiorradial Dir.","Cotovelo Dir.","Antebraço Dir.","Pulso Dir."],
                "Perna Esq.": ["Pectíneo Esq.","Pectíneo-Quadríceps Esq.","Quadríceps Esq.",
                               "Quadríceps-Coxa Esq.","Coxa Esq.","Coxa-Joelho Esq.",
                               "Joelho Esq.","Gastrocnêmio Esq.","Sóleo Esq.","Tornozelo Esq."],
                "Perna Dir.": ["Pectíneo Dir.","Pectíneo-Quadríceps Dir.","Quadríceps Dir.",
                               "Quadríceps-Coxa Dir.","Coxa Dir.","Coxa-Joelho Dir.",
                               "Joelho Dir.","Gastrocnêmio Dir.","Sóleo Dir.","Tornozelo Dir."],
            };
    
            // ---- decide o estágio mais avançado disponível a partir do /api/status ----
            function determinarEstagio(s) {
                if (s.measurement_pronto) return "medidas";
                if (s.ply_pronto) return "mesh";
                if (s.pcd_pronto) return "pcd";
                return null;
            }
    
            function configurarPainel(estagio, s) {
                const aviso = document.getElementById("aviso");
                const subtitulo = document.getElementById("painel-subtitulo");
    
                if (estagio === "medidas") {
                    document.getElementById("painel-titulo").textContent = "Medidas Corporais";
                    subtitulo.textContent = s.vertices ? `${s.vertices} vértices` : "";
                    aviso.style.display = "none";
                } else if (estagio === "mesh") {
                    document.getElementById("painel-titulo").textContent = "Malha Reconstruída";
                    subtitulo.textContent = s.vertices ? `${s.vertices} vértices` : "";
                    aviso.style.display = "block";
                    aviso.innerHTML = "<strong>Malha construída.</strong><br>As medidas corporais ainda estão sendo calculadas.";
                } else if (estagio === "pcd") {
                    document.getElementById("painel-titulo").textContent = "Alinhamento das Nuvens";
                    subtitulo.textContent = s.pontos ? `${s.pontos} pontos` : "";
                    aviso.style.display = "block";
                    aviso.innerHTML = "<strong>Alinhamento concluído.</strong><br>A malha triangular ainda não foi construída.";
                }
            }
    
            async function carregarMedidas() {
                const res = await fetch("/medidas");
                const medidas = await res.json();
                const lista = document.getElementById("medidas-lista");
                lista.innerHTML = "";
    
                for (const [secao, nomes] of Object.entries(SECOES)) {
                    const titulo = document.createElement("div");
                    titulo.className = "secao-titulo";
                    titulo.textContent = secao;
                    lista.appendChild(titulo);
    
                    for (const nome of nomes) {
                        const item = document.createElement("div");
                        item.className = "medida-item";
                        item.dataset.nome = nome; // usado pelo hover 3D -> lista
    
                        // efeito recíproco: passar o mouse na lista destaca a
                        // linha 3D correspondente
                        item.addEventListener("mouseenter", () => definirDestaque(nome));
                        item.addEventListener("mouseleave", () => definirDestaque(null));
    
                        const nomeEl = document.createElement("span");
                        nomeEl.className = "medida-nome";
                        nomeEl.textContent = nome;
    
                        const valorEl = document.createElement("span");
                        valorEl.className = "medida-valor";
    
                        if (medidas[nome] !== undefined) {
                            valorEl.textContent = `${medidas[nome]} cm`;
                        } else {
                            valorEl.textContent = "–";
                            valorEl.classList.add("medida-nao-detectado");
                        }
    
                        item.appendChild(nomeEl);
                        item.appendChild(valorEl);
                        lista.appendChild(item);
                    }
                }
            }
    
            function limparThree() {
                animating = false;
                if (controls) controls.dispose();
                if (renderer) { renderer.dispose(); renderer.forceContextLoss(); }
                if (scene) {
                    scene.traverse((obj) => {
                        if (obj.geometry) obj.geometry.dispose();
                        if (obj.material) {
                            if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                            else obj.material.dispose();
                        }
                    });
                }
            }
    
            function iniciarCena() {
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.01, 2000);
                camera.position.set(0, 0, 10);
    
                scene = new THREE.Scene();
                scene.add(new THREE.AxesHelper(5));
                scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    
                const dirLight = new THREE.DirectionalLight(0xffffff, 1);
                dirLight.position.set(0, 1, 1);
                scene.add(dirLight);
    
                renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(renderer.domElement);
    
                controls = new OrbitControls(camera, renderer.domElement);
                controls.update();
    
                grupo = new THREE.Group();
                grupo.rotateX(-Math.PI / 2);
                scene.add(grupo);
            }
    
            function carregarMesh() {
                return new Promise((resolve, reject) => {
                    const plyLoader = new PLYLoader();
                    plyLoader.load("/ply", function (geometry) {
                        geometry.computeVertexNormals();
                        const material = new THREE.MeshStandardMaterial({
                            vertexColors: true,
                            side: THREE.DoubleSide
                        });
                        const meshObj = new THREE.Mesh(geometry, material);
                        grupo.add(meshObj);
                        resolve(meshObj);
                    }, null, reject);
                });
            }
    
            // nuvem de pontos ainda não tem faces, então renderiza como THREE.Points
            function carregarNuvemDePontos() {
                return new Promise((resolve, reject) => {
                    const plyLoader = new PLYLoader();
                    plyLoader.load("/ply", function (geometry) {
                        if (!geometry.attributes.color) {
                            geometry.computeBoundingBox();
                            const { min, max } = geometry.boundingBox;
                            const range = Math.max(max.y - min.y, 1e-6);
                            const pos = geometry.attributes.position;
                            const colors = new Float32Array(pos.count * 3);
                            const low = new THREE.Color(0x1a1d27);
                            const high = new THREE.Color(0x7eb8f7);
                            const c = new THREE.Color();
                            for (let i = 0; i < pos.count; i++) {
                                c.copy(low).lerp(high, (pos.getY(i) - min.y) / range);
                                colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
                            }
                            geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
                        }
                        const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
                        const pontos = new THREE.Points(geometry, material);
                        grupo.add(pontos);
                        resolve(pontos);
                    }, null, reject);
                });
            }

            // linhas 3D que têm uma medida associada (usadas no raycasting de hover)
            const linhasComMedida = [];
            // materiais das fat lines — precisam ter a resolution atualizada no resize
            const materiaisLinha = [];
    
            const ESPESSURA_PADRAO = 3; // pixels de tela
            const ESPESSURA_DESTAQUE = 5;
    
            // estado compartilhado de destaque — usado tanto pelo hover no 3D
            // (raycaster) quanto pelo hover na lista lateral (efeito recíproco)
            let nomeDestaqueAtual = null;
            let itemDestacadoDOM = null;
    
            function limparDestaqueTudo() {
                if (itemDestacadoDOM) {
                    itemDestacadoDOM.classList.remove("medida-destacada");
                    itemDestacadoDOM = null;
                }
                for (const linha of linhasComMedida) {
                    if (linha.userData.coresOriginais) {
                        // LineSegmentsGeometry guarda cor em atributos de instância
                        // (instanceColorStart/End), não num "color" simples — por
                        // isso a forma correta de trocar cor é chamar setColors()
                        // de novo, não mexer direto no buffer.
                        linha.geometry.setColors(linha.userData.coresOriginais);
                        linha.material.linewidth = ESPESSURA_PADRAO;
                        linha.material.depthTest = true;
                        linha.renderOrder = 0;
                    }
                }
                nomeDestaqueAtual = null;
            }
    
            function definirDestaque(nome) {
                if (nome === nomeDestaqueAtual) return;
                limparDestaqueTudo();
                if (!nome) return;
    
                nomeDestaqueAtual = nome;
    
                const item = document.querySelector(`.medida-item[data-nome="${CSS.escape(nome)}"]`);
                if (item) {
                    item.classList.add("medida-destacada");
                    item.scrollIntoView({ block: "nearest", behavior: "smooth" });
                    itemDestacadoDOM = item;
                }
    
                for (const linha of linhasComMedida) {
                    if (linha.userData.nomeMedida !== nome) continue;
                    const branco = new Array(linha.userData.coresOriginais.length).fill(1);
                    linha.geometry.setColors(branco);
                    linha.material.linewidth = ESPESSURA_DESTAQUE;
                    linha.material.depthTest = false; // aparece mesmo "dentro" da malha
                    linha.renderOrder = 999;
                }
            }

            async function carregarLineset() {
                const res = await fetch("/lineset");
                const data = await res.json();
    
                for (const ls of data.linesets) {
                    const positions = [];
                    const lineColors = [];
    
                    for (const [i, j] of ls.lines) {
                        const pi = ls.points[i];
                        const pj = ls.points[j];
                        positions.push(pi[0], pi[1], pi[2]);
                        positions.push(pj[0], pj[1], pj[2]);
                        lineColors.push(ls.colors[i][0], ls.colors[i][1], ls.colors[i][2]);
                        lineColors.push(ls.colors[j][0], ls.colors[j][1], ls.colors[j][2]);
                    }
    
                    const geometry = new LineSegmentsGeometry();
                    geometry.setPositions(positions);
                    geometry.setColors(lineColors);
    
                    const material = new LineMaterial({
                        vertexColors: true,
                        linewidth: ESPESSURA_PADRAO, // em pixels de tela (worldUnits: false)
                        resolution: new THREE.Vector2(container.clientWidth, container.clientHeight),
                    });
    
                    const lineSegments = new LineSegments2(geometry, material);
    
                    if (ls.nome) {
                        lineSegments.userData.nomeMedida = ls.nome;
                        // guarda a cor "de fábrica" pra poder restaurar depois
                        // que geometry.setColors() sobrescrever pro branco
                        lineSegments.userData.coresOriginais = lineColors.slice();
                        linhasComMedida.push(lineSegments);
                    }
    
                    materiaisLinha.push(material);
                    grupo.add(lineSegments);
                }
            }

            // raycasting: quando o mouse passa perto de uma linha em 3D, destaca
            // o item correspondente em .medidas-lista (via data-nome)
            function configurarHoverMedidas() {
                if (linhasComMedida.length === 0) return;
    
                const raycaster = new THREE.Raycaster();
                // fat lines (LineSegments2) usam o namespace "Line2" pro threshold
                // do raycast, não "Line". Valor em unidades de mundo (cm, no seu
                // caso) — ajuste se estiver difícil/fácil demais de acertar.
                raycaster.params.Line2 = { threshold: 1.0 };
                const mouse = new THREE.Vector2();
    
                renderer.domElement.addEventListener("mousemove", (event) => {
                    const rect = renderer.domElement.getBoundingClientRect();
                    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
                    raycaster.setFromCamera(mouse, camera);
                    const acertos = raycaster.intersectObjects(linhasComMedida, false);
                    definirDestaque(acertos.length > 0 ? acertos[0].object.userData.nomeMedida : null);
                });
    
                renderer.domElement.addEventListener("mouseleave", () => definirDestaque(null));
            }
    
            // "de frente" depende de como sua captura orienta o corpo no espaço.
            // Se o enquadramento sair de lado/de trás/de cima em vez de frontal,
            // troque este valor: "z" | "-z" | "x" | "-x"
            const EIXO_FRENTE = "z";
            const OFFSETS_FRENTE = {
                "z":  new THREE.Vector3(0, 0, 1),
                "-z": new THREE.Vector3(0, 0, -1),
                "x":  new THREE.Vector3(1, 0, 0),
                "-x": new THREE.Vector3(-1, 0, 0),
            };
    
            // calcula a bounding box de tudo que está no grupo (mesh/pontos/lineset)
            // e posiciona a câmera na frente, a uma distância que enquadra o objeto
            // inteiro — sem isso a câmera ficava fixa em (0,0,10), dentro da malha
            function enquadrarCamera() {
                if (!grupo || grupo.children.length === 0) return;
    
                const box = new THREE.Box3().setFromObject(grupo);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
    
                const maiorDimensao = Math.max(size.x, size.y, size.z) || 1;
                const fovRad = camera.fov * (Math.PI / 180);
                const distancia = (maiorDimensao / 2) / Math.tan(fovRad / 2) * 1.5; // 1.5 = margem
    
                const direcao = OFFSETS_FRENTE[EIXO_FRENTE].clone();
                camera.position.copy(center).add(direcao.multiplyScalar(distancia));
                camera.near = Math.max(distancia / 100, 0.01);
                camera.far = distancia * 10;
                camera.updateProjectionMatrix();
    
                controls.target.copy(center);
                controls.minDistance = distancia * 0.1;
                controls.maxDistance = distancia * 5;
                controls.update();
            }
    
            function animate() {
                if (!animating) return;
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
                controls.update();
            }
    
            async function carregarEstagio(estagio, s) {
                iniciarCena();
                configurarPainel(estagio, s);
    
                if (estagio === "medidas") {
                    await carregarMesh();
                    await carregarLineset();
                    await carregarMedidas();
                } else if (estagio === "mesh") {
                    await carregarMesh();
                } else if (estagio === "pcd") {
                    await carregarNuvemDePontos();
                }
    
                enquadrarCamera();
                animate();
            }
    
            // ---- checa /api/status periodicamente e recarrega sozinho assim que
            //      uma etapa mais avançada terminar (pcd -> mesh -> medidas) ----
            let atualizando = false;
            async function checarNovaEtapa() {
                if (atualizando) return;
                try {
                    const res = await fetch("/api/status", { cache: "no-store" });
                    const s = await res.json();
                    const estagio = determinarEstagio(s);
                    if (estagio && estagio !== estagioAtual) {
                        atualizando = true;
                        document.getElementById("banner-atualizar").classList.add("show");
                        document.querySelector("#banner-atualizar span").textContent =
                            "Nova etapa concluída — atualizando…";
                        setTimeout(() => window.location.reload(), 900);
                    }
                } catch (e) {
                    console.error("Falha ao consultar /api/status:", e);
                }
            }
    
            document.getElementById("btn-voltar").addEventListener("click", async (e) => {
                e.preventDefault();
                limparThree();
    
                const btn = e.currentTarget;
                btn.textContent = "Reiniciando…";
                btn.style.pointerEvents = "none";
    
                try {
                    await fetch("/reset_estado", { method: "POST" });
                } catch (err) {
                    // esperado: a conexão pode cair no instante em que o
                    // processo morre pra reiniciar — segue o fluxo mesmo assim
                }
    
                // o processo inteiro está sendo substituído (os.execv), então
                // a porta fica momentaneamente indisponível. Em vez de navegar
                // direto pra "/" e arriscar cair num erro de conexão recusada,
                // espera o servidor voltar a responder antes de navegar.
                async function aguardarServidorVoltar() {
                    for (let tentativa = 0; tentativa < 30; tentativa++) {
                        await new Promise(r => setTimeout(r, 500));
                        try {
                            const res = await fetch("/api/status", { cache: "no-store" });
                            if (res.ok) {
                                window.location.href = "/";
                                return;
                            }
                        } catch (err) {
                            // servidor ainda de pé caindo/subindo, tenta de novo
                        }
                    }
                    // depois de ~15s sem resposta, navega mesmo assim
                    window.location.href = "/";
                }
                aguardarServidorVoltar();
            });
    
            window.addEventListener("resize", () => {
                if (!camera || !renderer) return;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
    
                // fat lines (LineMaterial) calculam espessura em pixels de tela,
                // então precisam saber o tamanho atual do viewport pra desenhar
                // certo — sem isso a linha fica com espessura errada após resize
                for (const mat of materiaisLinha) {
                    mat.resolution.set(container.clientWidth, container.clientHeight);
                }
            });
    
            async function main() {
                const res = await fetch("/api/status", { cache: "no-store" });
                const s = await res.json();
                estagioAtual = determinarEstagio(s);
    
                if (!estagioAtual) {
                    document.getElementById("painel-titulo").textContent = "Nada disponível ainda";
                    document.getElementById("aviso").style.display = "block";
                    document.getElementById("aviso").innerHTML =
                        "<strong>Nenhum resultado pronto.</strong><br>Volte e aguarde a captura terminar.";
                    return;
                }
    
                await carregarEstagio(estagioAtual, s);
                setInterval(checarNovaEtapa, 2000);
            }
            main();
        </script>
    </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Avanutri - Sistema de Avaliação Corporal</title>
        <style>
            * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            }

            body {
            background: #0f1117;
            color: #008000;
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
            padding: 24px;
            }

            h1 {
            text-align: center;
            font-size: 1.3rem;
            color: #7eb8f7;
            margin-bottom: 24px;
            letter-spacing: 1px;
            }

            .layout {
            display: flex;
            gap: 32px;
            align-items: flex-start;
            justify-content: center;
            }

            /* ── Câmeras ── */
            .cameras {
            display: flex;
            gap: 24px;
            }

            .cam-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            }

            .cam-box span {
            font-size: 0.78rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            }

            .cam-box img {
            width: 380px;
            border-radius: 10px;
            border: 2px solid #2a2d3a;
            background: #1a1d27;
            }

            /* ── Painel lateral ── */
            .panel {
            background: #1a1d27;
            border: 1px solid #2a2d3a;
            border-radius: 12px;
            padding: 24px;
            width: 260px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            }

            .panel h2 {
            font-size: 0.9rem;
            color: #7eb8f7;
            border-bottom: 1px solid #2a2d3a;
            padding-bottom: 10px;
            }

            .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
            }

            .field label {
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            }

            .field input {
            background: #0f1117;
            border: 1px solid #2a2d3a;
            border-radius: 6px;
            color: #e0e0e0;
            padding: 8px 10px;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s;
            }

            .field input:focus {
            border-color: #7eb8f7;
            }

            .btn-init {
            margin-top: 4px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 11px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            }

            .btn-init:hover { background: #1d4ed8; }

            .status {
            font-size: 0.78rem;
            text-align: center;
            color: #888;
            min-height: 18px;
            }
        </style>
        <script>
            async function iniciar() {
                console.log("iniciando iniciar")

                const nome   = document.getElementById("nome").value;
                const altura = document.getElementById("altura").value;
                const peso   = document.getElementById("peso").value;
                const idade  = document.getElementById("idade").value;

                // await fetch("/countdown_done", { method: "POST" });

                console.log("dados nome, altura, peso e idade inicializados")

                if (!nome || !altura || !peso || !idade ) {
                    document.getElementById("status").textContent = "Preencha os campos obrigatórios.";
                    return;
                }

                console.log("chamando /start com:", { nome, altura, peso, idade });

                document.getElementById("status").textContent = "Iniciando captura...";

                // POST /start virá aqui quando implementar
                // fetch("/start", { method: "POST", ... })

                // Chamada no Python
                const res = await fetch("/start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        nome,
                        altura: parseFloat(altura),
                        peso:   parseFloat(peso),
                        idade:  parseInt(idade)
                    })
                });

                if (!res.ok) {
                    console.log("erro no /start:", res.status);
                    return;
                }

                console.log("resposta status:", res.status);

                console.log("nome:",nome)

                const data = await res.json();
                document.getElementById("status").textContent = data.mensagem;

                iniciarCountdown(5);
            }

            async function verificarStatus() {
                try {
                    const resp = await fetch('/color_status');
                    const dados = await resp.json();
                    document.getElementById('alerta').style.display = dados.alert ? 'block' : 'none';
                } catch (e) {
                    console.error('Erro ao consultar /color_status:', e);
                }
            }

            function iniciarCountdown(secs) {
                const overlay = document.getElementById("countdown-overlay");
                const number  = document.getElementById("countdown-number");

                overlay.style.display = "flex";  // mostra o overlay
                number.textContent = secs;
                const interval = setInterval(async () => {
                    secs--;
                    if (secs > 0) {
                        number.textContent = secs;
                    } else {
                        clearInterval(interval);
                        overlay.style.display = "none";
                        document.getElementById("status").textContent = "Capturando...";
                        await fetch("/countdown_done", { method: "POST" });
                        iniciarPolling();
                    }
                }, 1000);
            }

            function iniciarPolling() {
                console.log("[POLLING] iniciado");
                const interval = setInterval(async () => {
                    const res  = await fetch("/origin_reached");
                    const data = await res.json();
                    console.log("[POLLING] measurement_pronto:", data.measurement_pronto);

                    if (data.measurement_pronto || data.ply_pronto || data.pcd_pronto) {
                        clearInterval(interval);
                        window.location.href = "/visualizar";
                    }
                }, 1000);
            }

            document.getElementById("btn-init").addEventListener("click", iniciar);

            window.addEventListener("pageshow", () => {
                location.reload();
            });
        setInterval(verificarStatus, 500);
        </script>
    </head>
    <body>
        <h1>3DBody — Avaliação de Circunferências Corporais</h1>
        
        <div id="video-container" style="position: relative; display: inline-block;">
            <!-- seu <img>/<video> do stream já existente entra aqui dentro -->
            <div id="alerta" style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                padding: 16px 32px;
                font-size: 2rem;
                font-weight: bold;
                color: #fff;
                background: rgba(200, 0, 0, 0.85);
                border-radius: 8px;
                display: none;
                pointer-events: none;
            ">
                Mantenha-se nessa posição!
            </div>
        </div>

        <div id="countdown-overlay" style="
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.75);
            justify-content: center;
            align-items: center;
            z-index: 999;
        ">
            <div style="text-align: center;">
            <div id="countdown-number" style="
                font-size: 10rem;
                font-weight: bold;
                color: #7eb8f7;
                line-height: 1;
            "></div>
            <div style="color: #aaa; font-size: 1.2rem; margin-top: 16px;">
                Prepare-se...
            </div>
            </div>
        </div>

        <div class="layout">
            <!-- Câmeras -->
            <div class="cameras">
                <div class="cam-box">
                    <img src="/stream/filtered_rgb" alt="RGB+Filtered">
                    <!--  <span>RGB+Filtered</span> -->
                </div>
                <div class="cam-box">
                    <img src="/stream/rgb" alt="RGB">
                    <!-- <span>Depth</span>-->
                </div>
            </div>

            <!-- Painel de cadastro -->
            <div class="panel">
            <h2>Cadastro de Paciente</h2>

            <div class="field">
                <label>Nome</label>
                <input type="text" id="nome" placeholder="Nome completo">
            </div>

            <div class="field">
                <label>Altura (cm)</label>
                <input type="number" id="altura" placeholder="Ex: 175">
            </div>

            <div class="field">
                <label>Peso (kg)</label>
                <input type="number" id="peso" placeholder="Ex: 70">
            </div>

            <div class="field">
                <label>Idade</label>
                <input type="number" id="idade" placeholder="Ex: 30">
            </div>

            <button class="btn-init" onclick="iniciar()">Iniciar Captura ▶</button>

            <div class="status" id="status">Aguardando início...</div>
        </div>

        <!-- overlay countdown -->
        <div id="countdown-overlay" style="display:none; position:fixed; inset:0;
            background:rgba(0,0,0,0.75); justify-content:center; align-items:center; z-index:999;">
            <div style="text-align:center;">
                <div id="countdown-numero" style="font-size:10rem; font-weight:bold; color:#7eb8f7;">5</div>
                <div style="color:#aaa; font-size:1.2rem; margin-top:16px;">Prepare-se...</div>
            </div>
        </div>
    </body>
    </html>
    """
    
@app.get("/api/status")
def status():
    return JSONResponse({
        "running": running,
        "pcd_pronto": pcd_pronto,
        "ply_pronto": ply_pronto,
        "measurement_pronto": measurement_pronto,
        "error": error,
        #"pontos": n_pontos,
        #"vertices": n_vertices,
        #"triangulos": n_triangulos,
    })

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()