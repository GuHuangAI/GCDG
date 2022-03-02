# -*- coding: utf-8 -*-
# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys, os
import json
import torch
import math
import cv2
import trimesh
import pickle
import imageio
from utils import ffmpeg
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely

import numpy as np
import pyrender
from pyrender.camera import PerspectiveCamera
from pyrender.constants import RenderFlags
from OpenGL import platform, _configflags
from vedo import *

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, faces, resolution=(224,224), wireframe=False,light_type='Point', ambient_light=(0.3, 0.3, 0.3), light_intensity=1):
        self.resolution = resolution

        self.faces = faces
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=ambient_light)

        if light_type=='Point':
            light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
        elif light_type=='Directional':
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
        else:
            raise Exception('Light type is not recognized. Currently, only Point or Directional light is supported!')

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def __call__(self, verts, faces=None, cam=[0.8,0.8,0,0], camera_pose=np.eye(4), fov=np.radians(50),angle=None, axis=None, mesh_filename=None, persp=False, color=[1.0, 1.0, 0.9]):
        # print(verts.shape)
        verts[:,0] *= -1 
        verts[:,2] *= -1 
        faces = self.faces if faces is None else faces
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        if persp:
            camera = PerspectiveCamera(yfov=fov)
        else:
            sx, sy, tx, ty = cam
            camera = WeakPerspectiveCamera(
                scale=[sx, sy],
                translation=[tx, ty],
                zfar=10000.)
            
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0))

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        image, _ = self.renderer.render(self.scene, flags=render_flags)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def get_renderer(smpl_model_path, vertices, save_type, save_path, i, test=False, model_type='smplx', resolution = (512,512,3), part_segment=False, **kwargs):
    import imutils
    model = pickle.load(open(os.path.join(smpl_model_path, 'smpl/SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
    faces = model['f']
    renderer = Renderer(faces, resolution=resolution[:2], **kwargs)
    if test:
        save_dir = os.path.join(save_path, 'mesh_frames', save_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result = renderer(vertices)
        result = imutils.rotate(result, 180)
        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i)), result.astype(np.uint8))

def get_tex_renderer(smpl_model_path, vertices, save_type, save_path, i, test=False, model_type='smplx', resolution = (512,512,3), part_segment=False, **kwargs):
    import imutils
    model = pickle.load(open(os.path.join(smpl_model_path, 'smpl/SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
    faces = model['f']
    # renderer = Renderer(faces, resolution=resolution[:2], **kwargs)
    visulizer = vedo_visualizer('.')
    if test:
        save_dir = os.path.join(save_path, 'mesh_frames', save_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result = visulizer.run(vertices)
        print(vertices.shape)
        print(result.shape)
        result = imutils.rotate(result, 180)
        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i)), result.astype(np.uint8))

def create_gif(image_path, save_path, gif_name, duration = 1./60, to_video=None, audio_path=None):
    frames = []
    image_list = os.listdir(image_path)
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(image_path, image_name)))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imageio.mimsave(os.path.join(save_path, str(gif_name)+'.gif'), frames, 'GIF', duration = duration)

    if to_video:
        ffmpeg.save_to_movie(
            os.path.join(save_path, str(gif_name) + "_mesh.mp4"),
            os.path.join(image_path, '%d.png'))
        print('video saved success')

    if audio_path:
        ffmpeg.attach_audio_to_movie(
            os.path.join(save_path, str(gif_name) + "_mesh.mp4"),
            audio_path,
            os.path.join(save_path, str(gif_name) + "_mesh_audio.mp4")
        )

class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))

class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s

class vedo_visualizer(object):
    def __init__(self, smpl_model_path):
        smpl_param_dict = pickle.load(open(os.path.join(smpl_model_path, 'smpl', 'SMPL_NEUTRAL.pkl'), 'rb'),
                                      encoding='latin1')
        self.smpl_model_path = smpl_model_path
        self.faces = smpl_param_dict['f']
        # self.verts_mean = smpl_param_dict['v_template']
        # self.load_smpl_tex()
        # self.load_smpl_vtk()
        smpl_uvmap = os.path.join(smpl_model_path, 'smpl', 'uv_table.npy')
        self.uv_map = np.load(smpl_uvmap)
        self.texture_file = os.path.join(smpl_model_path, 'smpl', 'SMPL_sampleTex_m.jpg')
        # if args.webcam_mesh_color == 'female_tex':
        #     self.uv_map = np.load(smpl_uvmap)
        #     self.texture_file = smpl_female_texture
        # elif args.webcam_mesh_color == 'male_tex':
        #     self.uv_map = np.load(args.smpl_uvmap)
        #     self.texture_file = args.smpl_male_texture
        # else:
        #     self.mesh_color = np.array(constants.mesh_color_dict[args.webcam_mesh_color]) / 255.

        # self.mesh = self.create_single_mesh(self.verts_mean)
        self.mesh_smoother = OneEuroFilter(4.0, 0.0)
        # self.vp = Plotter(title='Predicted 3D mesh', interactive=0)  #
        # self.vp_2d = Plotter(title='Input frame', interactive=0)
        # show(self.mesh, axes=1, viewup="y", interactive=0)

    def load_smpl_tex(self):
        import scipy.io as sio
        UV_info = sio.loadmat(os.path.join(self.smpl_model_path, 'smpl', 'UV_Processed.mat'))
        self.vertex_reorder = UV_info['All_vertices'][0] - 1
        self.faces = UV_info['All_Faces'] - 1
        self.uv_map = np.concatenate([UV_info['All_U_norm'], UV_info['All_V_norm']], 1)

    def run(self, verts):
        verts[:, 1:] = verts[:, 1:] * -1
        verts = self.mesh_smoother.process(verts)
        # verts = verts[self.vertex_reorder]
        # self.mesh.points(verts)
        mesh = self.create_single_mesh(verts)
        # self.vp.show(mesh, viewup=np.array([0, -1, 0]))
        # self.vp_2d.show(Picture(frame))
        # return False
        return mesh

    def create_single_mesh(self, verts):
        mesh = Mesh([verts, self.faces])
        mesh.texture(self.texture_file, tcoords=self.uv_map)
        mesh = self.collapse_triangles_with_large_gradient(mesh)
        mesh.computeNormals()
        return mesh

    def collapse_triangles_with_large_gradient(self, mesh, threshold=4.0):
        points = mesh.points()
        new_points = np.array(points)
        mesh_vtk = Mesh(os.path.join(self.smpl_model_path, 'smpl', 'smpl_male.vtk'), c='w').texture(
            self.texture_file).lw(0.1)
        grad = mesh_vtk.gradient("tcoords")
        ugrad, vgrad = np.split(grad, 2, axis=1)
        ugradm, vgradm = mag(ugrad), mag(vgrad)
        gradm = np.log(ugradm * ugradm + vgradm * vgradm)

        largegrad_ids = np.arange(mesh.N())[gradm > threshold]
        for f in mesh.faces():
            if np.isin(f, largegrad_ids).all():
                id1, id2, id3 = f
                uv1, uv2, uv3 = self.uv_map[f]
                d12 = mag(uv1 - uv2)
                d23 = mag(uv2 - uv3)
                d31 = mag(uv3 - uv1)
                idm = np.argmin([d12, d23, d31])
                if idm == 0:  # d12, collapse segment to pt3
                    new_points[id1] = new_points[id3]
                    new_points[id2] = new_points[id3]
                elif idm == 1:  # d23
                    new_points[id2] = new_points[id1]
                    new_points[id3] = new_points[id1]
        mesh.points(new_points)
        return mesh

def write_to_html(img_names, plot_dict, vis_cfg):
    containers = []
    raw_layout = Layout(overflow_x='scroll',border='2px solid black',width='1800px',height='',
                    flex_direction='row',display='flex')
    for inds, img_name in enumerate(img_names):
        Hboxes = []
        for item in list(plot_dict.keys()):
            fig = plot_dict[item]['figs'][inds]
            fig['layout'] = {"title":{"text":img_name.replace(args.dataset_rootdir, '')}}
            Hboxes.append(go.FigureWidget(fig))
        containers.append(HBox(Hboxes,layout=raw_layout))
    all_figs = VBox(containers)
    save_name = os.path.join(vis_cfg['save_dir'],vis_cfg['save_name']+'.html')
    embed_minimal_html(save_name, views=[all_figs], title=vis_cfg['save_name'], drop_defaults=True)
    ipywidgets.Widget.close_all()
    del all_figs, containers, Hboxes

if __name__ == '__main__':
    get_renderer(test=True,model_type='smpl')
