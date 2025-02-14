#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import open3d as o3d
def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_set(WXYZ,scene,model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    all_depths = []
    all_rgb = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(WXYZ,view, gaussians, pipeline, background)
        rendering =render_pkg["render"]
        depth = render_pkg["depth"].cpu().detach().numpy()
        all_depths.append(depth)
        all_rgb.append(rendering.cpu().detach().numpy())
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    print("准备开始TSDF融合")
    all_depths = np.array(all_depths)
    all_rgb = np.array(all_rgb)


    voxel_size = 3.4643626845036104e-05 * 40
    sdf_trunc = 0.0001732181 * 40
    depth_trunc = 0.03547507388931697 * 50
    num_cluster = 50

    # import open3d as o3d
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    width, height =640,480
    fx,fy =520.9,521.0
    cx,cy = 325.1,249.7



    intrinsic = o3d.camera.PinholeCameraIntrinsic(width = width, height = height, fx = fx, fy = fy, cx = cx, cy = cy)
    for img_idx in tqdm(range(np.array(all_depths).shape[0]),desc="TSDF integration progress"):
        rgb = all_rgb[img_idx]
        depth = all_depths[img_idx]
        # depth[depth == 15] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(
                np.asarray(np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(np.transpose(depth, (1, 2, 0)), order="C")),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            depth_scale=1.0
        )
        # print()

        volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=views[img_idx].world_view_transform.T.cpu().numpy())
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(os.path.join(model_path, name, "ours_{}".format(iteration), "fuse.ply"), mesh)
    print("mesh saved at {}".format(os.path.join(model_path, name, "ours_{}".format(iteration), "fuse.ply")))
    mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
    o3d.io.write_triangle_mesh((os.path.join(model_path, name, "ours_{}".format(iteration), "fuse_post.ply")), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(model_path, name, "ours_{}".format(iteration), "fuse_post.ply")))

def render_sets(WXYZ,dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(WXYZ,scene,dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(WXYZ,scene,dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    WXYZ = True
    render_sets(WXYZ,model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)