import numpy as np
import math
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random
import pyk4a
from pyk4a import Config, PyK4A
import multiprocessing as mp

from .depth_process import *
import uoais.tools.detect_img as uoais
from .point_cloud_process import *


def get_plane_kcc(pc_img, rgb_img, obj_mask, downsample_rate=5, crop_radius=25, visualize=False):
    center = [pc_img.shape[0] // 2, pc_img.shape[1] // 2]

    # pc img 2d image perspective y, x -> 3d perspective x, y, z
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    pc_img_pruned, rgb_img_pruned = remove_background(pc_img.copy(), rgb_img.copy(), obj_mask)
    obj_surface = pc_img_pruned.reshape(-1, 3)

    point_downsample_size = obj_surface.shape[0] / downsample_rate
    print(obj_surface.shape)
    uniform_sampling = np.array([int(i * downsample_rate) for i in range(int(point_downsample_size - 1))])
    obj_surface_downsample = obj_surface[uniform_sampling]
    print(f"{obj_surface.shape} downsampled to {obj_surface_downsample.shape}")

    plt.figure()
    ax = plt.subplot(1, 2, 1, projection='3d')
    xs = obj_surface_downsample[:, 0]
    ys = obj_surface_downsample[:, 1]
    zs = obj_surface_downsample[:, 2]
    ax.scatter(xs, ys, zs, s=10, color='grey')

    # Get center of the object and evaluate the initial center point
    # pc_center, center_circle_idx, center_surface_inliers, center_normal, center_norm_degree = calculate_voxel_norm_center_info(pc_img[center[0], center[1]], obj_surface_downsample)
    # center_circle_idx, center_surface_inliers, center_normal, center_norm_degree = calculate_ranch_center_info(pc_img[center[0], center[1]], obj_surface_downsample)
    pc_center, center_circle_idx, center_surface_inliers, center_normal, center_norm_degree = calculate_optimal_center_info(pc_img[center[0], center[1]], obj_surface_downsample)
    ax.scatter(pc_center[0], pc_center[1], pc_center[2], s=300, color='g')

    # Activate optimal suction point search algorithm
    # opt_plane, inliners_idx, opt_crop_idx, iteration = optimize_plane(obj_surface_downsample, plane, pc_center, rad_crop_idx)
    # opt_inliners_idx, obj_surface_inliers, opt_normal, opt_norm_degree, iteration = ranch_plane(obj_surface_downsample)
    opt_inliners_idx, opt_surface_inliers, opt_normal, opt_norm_degree, iteration = ransun(obj_surface_downsample.copy(), pc_center)

    # opt_crop_inliers = obj_surface_downsample[opt_crop_idx]
    ax.scatter(opt_surface_inliers[:, 0], opt_surface_inliers[:, 1], opt_surface_inliers[:, 2], s=30, color='b')
    if iteration != 0:
        opt_plane_center = np.average(opt_surface_inliers, axis=0) # ransac

        # ax.scatter(opt_crop_inliers[:, 0], opt_crop_inliers[:, 1], opt_crop_inliers[:, 2], s=60, color='g')
        # opt_plane_center = np.average(opt_crop_inliers, axis=0)

        ax.scatter(opt_plane_center[0], opt_plane_center[1], opt_plane_center[2], s=300, color='orange')
    else:
        opt_plane_center = pc_img[center[0], center[1]]

    if center_normal[2] > 0:
        center_normal = -center_normal
    if opt_normal[2] > 0:
        opt_normal = -opt_normal

    print("Initial center point, normal and degree= ", len(center_circle_idx), center_normal, np.rad2deg(center_norm_degree))
    print("Optimal suction point, normal and degree = ", len(opt_inliners_idx), opt_normal, np.rad2deg(opt_norm_degree))
    print(f"center pc: {pc_center}, opt pc: {opt_plane_center}")

    if False:
        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]), np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                # Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
                Z[r, c] = (-opt_plane[0] * X[r, c] - opt_plane[1] * Y[r, c] - opt_plane[3]) / opt_plane[2]
        # ax.plot_wireframe(X, Y, Z, color='k')
        ax.plot_surface(X, Y, Z, alpha=0.2)

        # tune threshold
        # thresh_test_line = obj_surface[np.where(obj_surface[:, 0] == pc_center[0])]
        # a, b, c, d = -fit[0], -fit[1], 1, -fit[2]
        # distance = np.absolute((a * thresh_test_line[:, 1] + b * thresh_test_line[:, 0] + c * thresh_test_line[:, 2] + d) / np.sqrt(
        #     a ** 2 + b ** 2 + c ** 2))
        # thresh_test_line = thresh_test_line[np.where(distance < 3)[1]]
        # ax.scatter(thresh_test_line[:, 1], thresh_test_line[:, 0], thresh_test_line[:, 2], s=300, color='b')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-150, 80)
        ax.set_ylim(-130, 100)
        ax.set_zlim(800, 1050)
        plt.show()

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_img_pruned.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(rgb_img_pruned.reshape(-1, 3) / 255.0)

        pcd_center = o3d.geometry.PointCloud()
        pcd_center.points = o3d.utility.Vector3dVector(center_surface_inliers.reshape(-1, 3))
        pcd_center.paint_uniform_color([1, 0, 0])
        pcd_opt = o3d.geometry.PointCloud()
        pcd_opt.points = o3d.utility.Vector3dVector(opt_surface_inliers.reshape(-1, 3))
        pcd_opt.paint_uniform_color([0, 0, 1])
        pcd_center.scale(2, center=pcd_center.get_center())
        pcd_opt.scale(2, center=pcd_opt.get_center())

        plane_model, inliers = pcd.segment_plane(distance_threshold=3, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pcd.select_by_index(inliers)
        plane_mesh = o3d.geometry.TriangleMesh.create_box(200, 200, 1)
        plane_mesh.translate(inlier_cloud.get_center(), relative=False)
        R = vec2rot_mat(np.array([0, 0, 1]), normalize_vector(plane_model))
        plane_mesh.rotate(R, center=inlier_cloud.get_center())
        plane_mesh = make_transparent(plane_mesh, "plane_mesh", [0.1, 0.1, 0.9], 0.5)

        hull, _ = pcd.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color([1, 0, 0])

        # pcd.uniform_down_sample(128)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=40, max_nn=50))
        # opt_center_argmin = np.abs(pc_img - opt_plane_center).sum(axis=2).argmin()
        # opt_2D_center = np.unravel_index(opt_center_argmin, pc_img.shape[:2])
        # print(np.asarray(pcd.points)[1000], center, opt_2D_center)
        # print(np.asarray(pcd.normals)[(center[0] * center[1]) // 128], np.asarray(pcd.normals)[(opt_2D_center[0] * opt_2D_center[1]) // 128])

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(4)
        center_sphere.translate(pc_center, relative=False)
        center_sphere.paint_uniform_color([1, 0, 0])
        opt_sphere = o3d.geometry.TriangleMesh.create_sphere(4)
        opt_sphere.translate(opt_plane_center, relative=False)
        opt_sphere.paint_uniform_color([0, 1, 1])

        center_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1, cone_radius=1.5, cylinder_height=20, cone_height=15)
        center_arrow = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
        center_arrow.translate(pc_center, relative=False)
        R = vec2rot_mat(np.array([0, 0, 1]), center_normal)
        center_arrow.rotate(R, center=pc_center)
        # center_arrow.paint_uniform_color([0, 1, 0])
        opt_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1, cone_radius=1.5, cylinder_height=20, cone_height=15)
        opt_arrow = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
        opt_arrow.translate(opt_plane_center, relative=False)
        R = vec2rot_mat(np.array([0, 0, 1]), opt_normal)
        opt_arrow.rotate(R, center=opt_plane_center)
        # opt_arrow.paint_uniform_color([0, 1, 0])

        center_circle = o3d.geometry.TriangleMesh.create_cylinder(radius=crop_radius, height=3)
        center_circle.translate(pc_center, relative=False)
        R = vec2rot_mat(np.array([0, 0, 1]), center_normal)
        center_circle.rotate(R, center=pc_center)
        center_circle = make_transparent(center_circle, "center_circle", [0.3, 0.3, 0.3])
        opt_circle = o3d.geometry.TriangleMesh.create_cylinder(radius=crop_radius, height=3, create_uv_map=True)
        opt_circle.translate(opt_plane_center, relative=False)
        R = vec2rot_mat(np.array([0, 0, 1]), opt_normal)
        opt_circle.rotate(R, center=opt_plane_center)
        opt_circle = make_transparent(opt_circle, "opt_circle", [0.3, 0.3, 0.3])

        # plane_mesh, hull_ls, pcd_center, pcd_opt
        geoms = [pcd, center_sphere, opt_sphere, center_arrow, opt_arrow, center_circle, opt_circle]
        o3d.visualization.draw(geoms, show_skybox=False)

        # o3d.visualization.draw_geometries([pcd, center_sphere, opt_sphere, center_arrow, opt_arrow, center_circle, opt_circle])
        # o3d.io.write_point_cloud("point_cloud_module/pc_fig.ply", pcd)

        if opt_normal[2] < 0:
            return -opt_normal, opt_plane_center
        else:
            return opt_normal, opt_plane_center

if __name__ == "__main__":
    # Azure Kinect
    y_start = 60
    x_start = 200
    height = 600
    width = 900
    camera = True

    if camera:
        mp.set_start_method("spawn", force=True)
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                synchronized_images_only=True,
                camera_fps=pyk4a.FPS.FPS_5,
            ))
        k4a.start()
    else:
        rgb_img = cv2.imread("/home/car/PycharmProjects/ICRA2024/result/episode1004/raw/rgb/0.png")
        depth = cv2.imread("/home/car/PycharmProjects/ICRA2024/result/episode1004/raw/depth/0.png")
        pcd = o3d.io.read_point_cloud("/home/car/PycharmProjects/ICRA2024/result/episode1004/raw/point_cloud/0.ply")
        pcd_np = np.asarray(pcd.points)
        pc_img = pcd_np.reshape((height, width, 3))[y_start:y_start + height, x_start:x_start + width]


    while(True):
        if camera:
            capture = k4a.get_capture()
            rgb_img = capture.color[:, :, :3]
            depth = capture.transformed_depth
            pc_img = capture.transformed_depth_point_cloud[y_start:y_start + height, x_start:x_start + width]

        rgb_img = rgb_img[y_start:y_start + height, x_start:x_start + width]
        depth = depth[y_start:y_start + height, x_start:x_start + width]
        depth_img = normalize_depth(depth)
        # # depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        uoais_img = uoais.detect(rgb_img, depth, width, height, pc_img)

        cv2.imshow('vis_all_img venv3.8', uoais_img)

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    k4a.close()