from vedo import *
import pickle
import os
import numpy as np
import cv2
import imageio
from utils import ffmpeg

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

def get_tex_renderer(smpl_model_path, vertices, save_type, save_path, i, test=False, model_type='smplx', resolution = (512,512,3), part_segment=False, **kwargs):
    # import imutils
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


def create_gif(ver_path, smpl_model_path, image_path, save_path, gif_name,
               duration=1. / 60, to_video=None, audio_path=None):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    vertices = np.load(ver_path)
    visulizer = vedo_visualizer(smpl_model_path)
    print(vertices.shape[0])
    for i in range(vertices.shape[0]):
        if os.path.exists(os.path.join(image_path, '{}.png'.format(i))):
            continue
        vertice = vertices[i]
        result = visulizer.run(vertice)
        plt = Plotter(offscreen=True)
        plt.show(result, interactive=False, axes=0, resetcam=True, camera={'pos': (0, 0, -4), 'viewup': [0, -1, 0], })
        screenshot(os.path.join(image_path, '{}.png'.format(i)))

    frames = []
    image_list = os.listdir(image_path)
    image_list = sorted(image_list, key=lambda x: int(x[:-4]))
    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(image_path, image_name)))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imageio.mimsave(os.path.join(save_path, str(gif_name) + '.gif'), frames, 'GIF', duration=duration)

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

if __name__ == '__main__':
    smpl_model_path = r'G:\database\ROMP\models'
    ver_path = r'G:\dance2motion\pred_vertices\1.npy'
    image_path = r'G:\dance2motion\pred_gif'
    save_path = r'G:\dance2motion'
    gif_name = 'pred_1'
    create_gif(ver_path, smpl_model_path, image_path, save_path, gif_name,
               duration=1. / 60, to_video=None, audio_path=None)

    # smpl_model_path = r'G:\database\ROMP\models'
    ver_path = r'G:\dance2motion\pred_vertices\2.npy'
    image_path = r'G:\dance2motion\pred_gif_2'
    save_path = r'G:\dance2motion'
    gif_name = 'pred_2'
    create_gif(ver_path, smpl_model_path, image_path, save_path, gif_name,
               duration=1. / 60, to_video=None, audio_path=None)

    ver_path = r'G:\dance2motion\gt_vertices\1.npy'
    image_path = r'G:\dance2motion\gt_gif_1'
    save_path = r'G:\dance2motion'
    gif_name = 'gt_1'
    create_gif(ver_path, smpl_model_path, image_path, save_path, gif_name,
               duration=1. / 60, to_video=None, audio_path=None)

    ver_path = r'G:\dance2motion\gt_vertices\2.npy'
    image_path = r'G:\dance2motion\gt_gif_2'
    save_path = r'G:\dance2motion'
    gif_name = 'gt_2'
    create_gif(ver_path, smpl_model_path, image_path, save_path, gif_name,
               duration=1. / 60, to_video=None, audio_path=None)

    # vertices = np.load(ver_path)
    # visulizer = vedo_visualizer(smpl_model_path)
    # vertice = vertices[0]
    # result = visulizer.run(vertice)
    # plt = Plotter(offscreen=False)
    # plt.show(result, interactive=True, axes=0, resetcam=True, camera={'pos':(0,0,-4), 'viewup':[0, -1, -0],})
    # screenshot(r'./test.png')
    pass