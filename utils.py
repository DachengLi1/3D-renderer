import numpy as np
import os
from PIL import Image
from collections import defaultdict
from scipy.io import loadmat
import torchvision
from math import sin,cos,radians
pad = torchvision.transforms.functional.pad
crop = torchvision.transforms.functional.crop

# rotate X (a point cloud matrix) by rx, ry and rz
def rotate(X,rx,ry,rz):
    Rx = np.array([[1, 0, 0],
          [0 ,np.cos(rx), -np.sin(rx)],
          [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry),0,np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                  [np.sin(rz),np.cos(rz),0],
                  [0, 0, 1]])
    
    X = np.matmul(Rx,X.T)
    X = np.matmul(Ry,X)
    X = np.matmul(Rz,X)
    X = X.T
    return X

# get rotation matrix by rx, ry, rz
def get_rotation_matrix(rx,ry,rz):
    Rx = np.array([[1, 0, 0],
          [0 ,np.cos(rx), -np.sin(rx)],
          [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                  [np.sin(rz),np.cos(rz),0],
                  [0, 0, 1]])
    rot = np.matmul(Rz,Ry)
    rot = np.matmul(rot,Rx)
    return np.array([rot[0][0],rot[0][1],rot[0][2],0],
                [rot[1][0],rot[1][1],rot[1][2],0],
                [rot[2][0],rot[2][1],rot[2][2],0],
                [0,0,0,1])

# render a point cloud x into an image. dist, focal and sensor_width is the 
# virtual cameral/projection plane parameter. Can change these three to adjust 
# the output size inthe final projection image.rx, ry, rz supports a rotation on x.
# normal 
def get_pixel(x,focal,dist,rx,ry,rz,resolution,sensor_width,normal=None):
    num_pts = x.shape[0]
    cam_location = np.array([dist,0,0])
    rx = radians(rx)
    ry = radians(ry)
    rz = radians(rz)
    n_1,n_2,n_3 = 1,0,0 # normal direction
    c_1,c_2,c_3 = cam_location
    x_rotate = rotate(x,rx,ry,rz)
    x_1 = x_rotate[:,0]
    x_2 = x_rotate[:,1]
    x_3 = x_rotate[:,2]
    # intersection of the lines and the image plane
    i_x,i_y,i_z = dist - focal, 0, 0
    
    # n1ix + n2iy + n3iz + d = 0
    d = dist - focal 
    ratio = focal / (dist-x_1)
    y_new = x_2 * ratio
    z_new = x_3 * ratio
    
    # convert 3d points to 2d coordinate
    x_2d = y_new
    y_2d = z_new
    
    pix_size = sensor_width / resolution
    x_final = np.round(np.round(x_2d / pix_size) + resolution / 2)
    y_final = np.round(np.round(y_2d / pix_size) + resolution / 2)
    
    # whether this point should have color, not used in the final version
    color = [True] * num_pts
    if(normal is not None):
        point_to_cam = np.array([c_1-x_1,c_2-x_2,c_3-x_3])
        dot_normal = point_to_cam.dot(normal)
        dot_normal = np.sum(point_to_cam,axis=0)
        color = dot_normal > 0
    
    dist = i_x - x_1
    return y_final, x_final, dist, color

# get a list of neighbor pixels in an image
def get_neighbors(y,x,resolution):
    neighbors = []
    if(y>0):
        neighbors.append((y-1,x))
    if(x>0):
        neighbors.append((y,x-1))
    if(y<resolution-1):
        neighbors.append((y+1,x))
    if(x<resolution-1):
        neighbors.append((y,x+1))
    return neighbors

# resize an image with mask with a specific ratio
# We found that if we separately resize them,
# they will not align because of interpolation in the margin.
# This resize is accurate
def resize_with_mask(img, mask, ratio):
    w,h = mask.size
    mask = np.asarray(mask)
    img = np.asarray(img)
    h_,w_ = int(h*ratio),int(w*ratio)
    mask = cv2.resize(mask.astype(np.float32),(h_,w_))
    img = cv2.resize(img.astype(np.float32),(h_,w_))
    
    new_img = np.zeros((h_,w_,3))
    new_img[np.where(mask==0)] = img[np.where(mask==0)]
    return Image.fromarray(new_img.astype(np.uint8),'RGB')

# sample point clouds from a mesh
# count is the number of points to be sampled
# modified from TriMesh
def sample_surface(mesh, count):
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    samples = sample_vector + tri_origins

    color = mesh.visual.face_colors
    colored = np.zeros((len(face_index),3))
    for i in range(len(face_index)):
        colored[i] = (color[face_index[i]])[:3]
    return samples, colored