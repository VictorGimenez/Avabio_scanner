import open3d as o3d
import numpy as np

from meshlib import mrmeshpy as mm
from pathlib import Path

def normals_estimating(file_location):
    mesh = o3d.io.read_triangle_mesh(file_location)
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])

def main():
    dir = input("Selecione o caminho do arquivo")
    file = input("Selecione o arquivo")
    desired_name_file = input("Selecione o nome desejado do arquivo")
    file_loc = input("Selecione o local da malha")

    wdir = Path(dir).parent
    pc = mm.loadPoints(wdir / file)
    fragment_mesh = mm.triangulatePointCloud(pc)
    mm.saveMesh(fragment_mesh, wdir / desired_name_file)
    
    #normals_estimating(file_loc)

if __name__ == "__main__":
    main()