import open3d
import numpy as np
import matplotlib.pyplot as plt

from utils.pointcloud import estimate_normal
from utils.pointcloud import make_point_cloud

import copy

class VisOpen3D:

    def __init__(self, width=1920, height=1080, visible=True, back_face = True):
        # self.__vis = open3d.visualization.Visualizer()
        self.__vis = open3d.visualization.VisualizerWithKeyCallback()

        self.__vis.create_window(width=width, height=height, visible=visible)
        self.__width = width
        self.__height = height
        self.__vis.register_key_callback(75, self.call_back_save_pose) # 75 is key 'k'

        opt = self.__vis.get_render_option()
        # opt.light_on = True
        if back_face == True:
            opt.mesh_show_back_face = True  
        opt.mesh_show_wireframe = True

        if visible:
            self.poll_events()
            self.update_renderer()
        
        self.save_pose_counter = 0

    def call_back_save_pose(self, vis):
        #Your update routine
        self.save_view_point("/home/ubuntu/Pictures/view_point_2.json")
        # self.save_pose_counter += 1
        print("call back triggered")
        self.run()
        
    def __del__(self):
        self.__vis.destroy_window()

    def render(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()

    def poll_events(self):
        self.__vis.poll_events()

    def update_renderer(self):
        self.__vis.update_renderer()

    def run(self):
        self.__vis.run()

    def destroy_window(self):
        self.__vis.destroy_window()

    def add_geometry(self, data):
        self.__vis.add_geometry(data)

    def update_view_point(self, intrinsic, extrinsic):
        ctr = self.__vis.get_view_control()
        param = self.convert_to_open3d_param(intrinsic, extrinsic)
        # import ipdb; ipdb.set_trace()
        print("!!", ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True))
        self.__vis.update_renderer()

    def get_view_point_intrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        return intrinsic

    def get_view_point_extrinsics(self):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = param.extrinsic
        return extrinsic

    def get_view_control(self):
        return self.__vis.get_view_control()

    def save_view_point(self, filename):
        ctr = self.__vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        param = open3d.io.read_pinhole_camera_parameters(filename)
        # param = open3d.io.read_pinhole_camera_parameters(filename)
        # intrinsic = param.intrinsic.intrinsic_matrix
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        extrinsic = extrinsic.copy()

        # self.draw_camera(intrinsic, extrinsic)

        # extrinsic[0,3] = 10

        print("load_view_point: ", intrinsic, extrinsic)
        # self.draw_camera(intrinsic, extrinsic)
                    
        self.update_view_point(intrinsic, extrinsic)

    def convert_to_open3d_param(self, intrinsic, extrinsic):
        param = open3d.camera.PinholeCameraParameters()
        param.intrinsic = open3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = extrinsic
        return param

    def capture_screen_float_buffer(self, show=False):
        image = self.__vis.capture_screen_float_buffer(do_render=True)

        if show:
            plt.imshow(image)
            plt.show()

        return image

    def capture_screen_image(self, filename):
        self.__vis.capture_screen_image(filename, do_render=True)

    def capture_depth_float_buffer(self, show=False):
        depth = self.__vis.capture_depth_float_buffer(do_render=True)

        if show:
            plt.imshow(depth)
            plt.show()

        return depth

    def capture_depth_image(self, filename):
        self.__vis.capture_depth_image(filename, do_render=True)

        # to read the saved depth image file use:
        # depth = open3d.io.read_image(filename)
        # import ipdb; ipdb.set_trace()
        # plt.imshow(depth)
        # plt.show()

    def draw_camera(self, intrinsic, extrinsic, scale=1, color=None):
        # intrinsics
        K = intrinsic

        # convert extrinsics matrix to rotation and translation matrix
        extrinsic = np.linalg.inv(extrinsic)
        R = extrinsic[0:3,0:3]
        t = extrinsic[0:3,3]

        width = self.__width
        height = self.__height

        geometries = draw_camera(K, R, t, width, height, scale, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_points3D(self, points3D, color=None):
        geometries = draw_points3D(points3D, color)
        for g in geometries:
            self.add_geometry(g)

    def draw_registration_result(self, source, target, transformation, line_set = None, vis_file_name=None):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        if not source_temp.has_normals():
            estimate_normal(source_temp)
            estimate_normal(target_temp)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        # o3d.visualization.draw_geometries([line_set, source_temp, target_temp])


        # vis.add_geometry(line_set)
        self.__vis.add_geometry(source_temp)
        self.__vis.add_geometry(target_temp)
        if line_set is not None:
            self.__vis.add_geometry(line_set)
        # self.load_view_point(vis_file_name)
        self.load_view_point(vis_file_name)
        self.__vis.run()
        

        # vis.destroy_window()
        # vis.draw_geometries([source_temp, target_temp])
        # o3d.visualization.draw_geometries([source_temp])
        # o3d.visualization.draw_geometries([target_temp])

#
# Auxiliary funcions
#
def draw_camera(K, R, t, width, height, scale=1, color=None):
    """ Create axis, plane and pyramid geometries in Open3D format
    :   param K     : calibration matrix (camera intrinsics)
    :   param R     : rotation matrix
    :   param t     : translation
    :   param width : image width
    :   param height: image height
    :   param scale : camera model scale
    :   param color : color of the image plane and pyramid lines
    :   return      : camera model geometries (axis, plane and pyramid)
    """

    # default color
    if color is None:
        color = [0.8, 0.2, 0.8]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array([[K[0, 0] * s,            0, K[0,2]],
                   [          0,  K[1, 1] * s, K[1,2]],
                   [          0,            0, K[2,2]]])
    Kinv = np.linalg.inv(Ks)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = create_coordinate_frame(T, scale=scale*0.5)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [width, 0, 1],
        [0, height, 1],
        [width, height, 1],
    ]

    # pixel to camera coordinate system
    points = [scale * Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.transform(T)
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


def create_coordinate_frame(T, scale=0.25):
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    frame.transform(T)
    return frame


def draw_points3D(points3D, color=None):
    # color: default value
    if color is None:
        color = [0.8, 0.2, 0.8]

    geometries = []
    for pt in points3D:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01,
                                                            resolution=20)
        sphere.translate(pt)
        sphere.paint_uniform_color(np.array(color))
        geometries.append(sphere)

    return geometries


