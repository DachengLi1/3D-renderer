import numpy as np
import open3d as o3d
import cv2
import os
from math import sin,cos,radians
from PIL import Image
import trimesh
import torchvision
from math import sin,cos,radians
from utils import get_pixel,rotate,get_neighbors,sample_surface,get_rotation_matrix
from collections import defaultdict
import copy
import time
import xml.etree.ElementTree as ET
import kdtree
from math import radians

# camera and image plane parameter
resolution = 500
sensor_width = 0.05
cam_focal = 0.05
cam_dist = 2.0
# pre-defined rotation angles in rx, ry, rz order
good_angles = [(-90,0,-5),(-90,0,-10),(-90,0,-15),(-90,0,-20),(-90,0,-25),(-90,0,-30),
               (-90,0,-35),(-90,0,-40),(-90,0,-45),(-90,0,-50),(-90,0,-55),(-90,0,-60),
               (-90,0,5),(-90,0,10),(-90,0,15),(-90,0,20),(-90,0,25),(-90,0,30),
               (-90,0,35),(-90,0,40),(-90,0,45),(-90,0,50),(-90,0,55),(-90,0,60)]

output_dir = "."
start = time.time()

# load pre-genrated mesh and accurate color files
npz = np.load("./materials/texture.npz")
rgb = npz["pred_sph_texture"]
colored_points = npz['pred_sph_texture_224']
mesh = trimesh.load("./materials/mesh.obj")

sample_density = 300000  #number of points to sample for point cloud

rgb_dict = defaultdict(int)
for i in range(len(rgb)):
    rgb_dict[tuple(colored_points[i])] = rgb[i]
tree = kdtree.create(list(colored_points))
vertices = mesh.vertices

# assign the color of the closest point to each point, in terms of L2 distance
for i in range(len(vertices)):
    node,_ = tree.search_nn(vertices[i])
    cur_color = np.append(rgb_dict[tuple(node.data)],255) # initialize with white pixels
    tree.add(vertices[i])
    rgb_dict[tuple(vertices[i])] = rgb_dict[tuple(node.data)]
    mesh.visual.vertex_colors[i] = cur_color

# sample points from the current colored mesh
mesh.visual.face_colors = trimesh.visual.color.vertex_to_face_color(mesh.visual.vertex_colors,mesh.faces)
sil_points,sil_color = sample_surface(mesh,sample_density)
points = np.concatenate((colored_points,sil_points),axis=0)
colors = np.concatenate((rgb,sil_color),axis=0)

num = 0
for i in range(len(good_angles)):
    try:
        rx,ry,rz = good_angles[i]
        num += 1

        image = np.ones((resolution,resolution,3)) * 255
        image_dist = np.full((resolution,resolution),fill_value=np.inf)
        y,x,dist,color = get_pixel(points,cam_focal,cam_dist,rx,ry,rz,resolution,sensor_width)
        num_pts_render = y.shape[0]
        num_color = len(colored_points)
        
        # take the color of the closest point from the mesh to the image plane
        for i in range(num_pts_render):
            y_ = int(y[i])
            x_ = int(x[i])
            if(image_dist[y_][x_] > dist[i]):
                if(i >= len(colored_points) and image_dist[y_][x_] != np.inf):
                    continue
                image[y_][x_] = colors[i]
                image_dist[y_][x_] = dist[i]
        
        # mask the non-white aera
        mask = np.sum(image,axis=2)
        mask[np.where(mask!=765)] = 0

        Image.fromarray(mask.astype(np.uint8),"L").save(os.path.join("./output","view_"+str(num)+"_mask.png"),"png")
        Image.fromarray(image.astype(np.uint8),"RGB").save(os.path.join("./output","view_"+str(num)+".png"),"png")
    except:
        print("some errors.")
        pass
end = time.time()
time_pass = end - start
print("done in " + str(time_pass) + " seconds.")