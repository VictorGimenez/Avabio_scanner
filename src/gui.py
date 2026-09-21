import os
import time
import copy
import ctypes
import time
import platform
import subprocess

import open3d as o3d
import numpy as np
import cv2 as cv

from screeninfo import get_monitors


sem = {0:"Segunda", 1:"Terça", 2:"Quarta", 3:"Quinta", 4:"Sexta", 5:"Sábado", 6:"Domingo"}
calend = {1:"Janeiro",   \
          2:"Fevereiro", \
          3:"Março", \
          4:"Abril", \
          5:"Maio",   \
          6:"Junho", \
          7:"Julho", \
          8:"Agosto", \
          9:"Setembro",   \
          10:"Outubro", \
          11:"Novembro", \
          12:"Dezembro"}


def clear_screen():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

def ava_logo():
    return "                                            /\\                \n" \
           "                                           /  \\               \n" \
           "                                          /    /               \n" \
           "                                         /    /                \n" \
           "              __                        /    / __              \n" \
           "             /  \\                      /    / /  \\             \n" \
           "            /    \\    /\\              /    / /    \\            \n" \
           "           /      \\  /  \\            /    / /      \\           \n" \
           "          /   /\\   \\ \\   \\          /    / /   /\\   \\          \n" \
           "         /   /  \\   \\ \\   \\        /    / /   /  \\   \\         \n" \
           "        /   /    \\   \\ \\   \\      /    / /   /    \\   \\        \n" \
           "       /   /      \\   \\ \\   \\    /    / /   /      \\   \\       \n" \
           "      /   /        \\   \\ \\   \\  /    / /   /        \\   \\      \n" \
           "     /   /          \\   \\ \\   \\/    / /   /          \\   \\     \n" \
           "    /   /            \\   \\ \\       / /   /            \\   \\    \n" \
           "    \\  /              \\  /  \\     /  \\  /              \\  /    \n" \
           "     \\/                \\/    \\___/    \\/                \\/     \n" \
           "                                                                    \n" \
           " A D V A N C E D    R E C O V E R Y   F O R   A T H L E T E S        "

def print_header():
    from datetime import datetime, date
    
    print(ava_logo())
    dia = date.today().weekday()
    mes = date.today().month
    print("\n")
    print(datetime.now().strftime(f"Hoje é {sem[dia]}, %d de {calend[mes]} de %Y - %H:%M:%S"))
    print("\n")
    

def select_option():
    print("[1] - Gerar Reconstrução 3D com câmera")
    print("[Outra opção] - Sair")
    option = int(input("Selecione a opção desejada:"))
    return option


def set_window_attributes(win_name, mv_width, mv_height, res_width, res_height, kp_rt = cv.WINDOW_KEEPRATIO, property = cv.WND_PROP_TOPMOST):
    cv.namedWindow(win_name, kp_rt)
    cv.moveWindow(win_name, mv_width, mv_height)
    cv.resizeWindow(win_name, res_width, res_height)
    cv.setWindowProperty(win_name, property, 1)


def monitor_information():
    for i, m in enumerate(get_monitors()):
        print(f"Monitor {i}: {m.width}x{m.height} — {m.name}")

    monitor = get_monitors()[0]
    screen_w, screen_h = monitor.width, monitor.height
    return screen_w, screen_h


def focus_main_window(win_name):

    if os.name == "nt":
        import win32gui
        import win32con
        import ctypes
        
        import pygetwindow as gw
        
        hwnd = win32gui.FindWindow(None, win_name)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)

        windows = gw.getWindowsWithTitle(win_name)
        if windows:
            try:
                windows[0].minimize()
                time.sleep(0.1)
                windows[0].restore()
            except:
                pass
    else:
        import subprocess
        
        hwnd = subprocess.run(["xdotool", "search", "--name", win_name], capture_output=True, text=True).stdout.strip().split('\n')[0]
        if hwnd:
            subprocess.run(["xdotool", "windowmap", hwnd])
            subprocess.run(["xdotool", "windowactivate", hwnd])
            subprocess.run(["xdotool", "windowfocus",    hwnd])
            subprocess.run(["xdotool", "windowraise",    hwnd])

def display_content(mesh, geometries, title1, title2):

    def move_window_by_title(title2, x, y, w, h):
        # hwnd = user32.FindWindowW(None, title2)
        
        # if hwnd:
        #     SWP_NOZORDER = 0x0004
        #     ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)
        # else:
        #     print(f"Janela '{title2}' não encontrada")
        
        if platform.system() == "Windows":
            import ctypes
            user32 = ctypes.windll.user32
            ctypes.windll.user32.SetProcessDPIAware()
            hwnd = user32.FindWindowW(None, title2)
            if hwnd:
                SWP_NOZORDER = 0x0004
                user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)
        elif platform.system() == "Linux":
            subprocess.run(["wmctrl", "-r", title2, "-e", f"0,{x},{y},{w},{h}"])

    # Declaração
    vis1 = o3d.visualization.Visualizer()
    vis2 = o3d.visualization.Visualizer()

    vis1.create_window(window_name=title1, width=800, height=600)
    vis1.get_render_option().mesh_show_back_face = True

    vis2.create_window(window_name=title2, width=800, height=600)
    vis2.get_render_option().mesh_show_back_face = True

    mesh = copy.deepcopy(mesh)
    
    if platform.system() == "Windows":
        ctypes.windll.user32.SetForegroundWindow(
            user32.FindWindowW(None, title1)
        )
        ctypes.windll.user32.SetForegroundWindow(
            user32.FindWindowW(None, title2)
        )
    else:
        subprocess.run(["wmctrl", "-a", title1])
        subprocess.run(["wmctrl", "-a", title2])

    mesh.compute_vertex_normals()

    # Limpeza e correção de normais — aqui
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[:, ::-1]
    )

    mesh.orient_triangles()

    vis1.add_geometry(mesh)
    vis2.add_geometry(mesh)
    #Desenhando as linhas em cada "fatia" da mesh
    for geom in geometries:
        if isinstance(geom, o3d.geometry.TriangleMesh):
            if not geom.has_vertex_colors():
                geom.paint_uniform_color([0.7, 0.7, 0.7])  # cinza
            if not geom.has_vertex_normals():
                geom.compute_vertex_normals()
        elif isinstance(geom, o3d.geometry.LineSet):
            if not geom.has_colors():
                geom.paint_uniform_color([1.0, 0.0, 0.0])  # vermelho
        vis2.add_geometry(geom)

    move_window_by_title(title1,   0, 0, 700, 500)
    move_window_by_title(title2, 820, 0, 700, 500)

    while True:
        if not vis1.poll_events() or not vis2.poll_events():
            break
        vis1.update_renderer()
        vis2.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()