import numpy as np
import os
import subprocess
import sys
import imageio


def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    colmap_exe = '/local/scratch/pmh64/software/usr/local/bin/colmap'

    feature_extractor_args = [
        colmap_exe, 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    matcher_args = [
        colmap_exe, match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        colmap_exe, 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    print('Sparse reconstruction done')

    p = os.path.join(basedir, 'dense')
    if not os.path.exists(p):
        os.makedir(p)

    undistorter_args = [
        colmap_exe, 'image_undistorter',
            '--image_path', os.path.join(basedir, 'images'),
            '--input_path', os.path.join(basedir, 'sparse', '0'),
            '--output_path', os.path.join(basedir, 'dense'), # --export_path changed to --output_path in colmap 3.6
            '--output_type', 'COLMAP',
    ]

    undistort_output = ( subprocess.check_output(undistorter_args, universal_newlines=True) )
    logfile.write(undistort_output)

    patchmatch_args = [
        colmap_exe, 'patch_match_stereo',
            '--workspace_path', os.path.join(basedir, 'dense'),
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', 'true',
    ]

    patchmatch_output = ( subprocess.check_output(patchmatch_args, universal_newlines=True) )
    logfile.write(patchmatch_output)
    logfile.close()
    print('Dense reconstruction done')

    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))

    if cam.model == 'SIMPLE_RADIAL':
        h, w, f = cam.height, cam.width, cam.params[0]
        # w, h, f = factor * w, factor * h, factor * f
        hwf = np.array([h,w,f]).reshape([3,1])
        cxcys = np.array(cam.params[1:]).reshape([3,1])
    elif cam.model == 'PINHOLE':
        assert cam.params[0] == cam.params[1], cam.params[0] - cam.params[1]
        h, w, f = cam.height, cam.width, cam.params[0]
        hwf = np.array([h,w,f]).reshape([3,1])
        cxcys = np.array(cam.params[1:] + [0]).reshape([3,1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    poses = np.concatenate([poses, np.tile(cxcys[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :], poses[:, 5:6, :]], 1)
    
    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    cam_valid_z = pts_arr[:,2]
    print( 'Cam depth stats', cam_valid_z.min(), cam_valid_z.max(), cam_valid_z.mean() )
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def load_data(basedir, index=None):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 6])
    bds = poses_arr[:, -2:]

    imgdir = os.path.join(basedir, 'images')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[0] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape
    assert (poses[0, :2, 4].astype(np.int) == np.array(sh[:2])).all()

    if index is not None:
        img = imageio.imread(imgfiles[index], ignoregamma=True)[...,:3].astype(np.float32)/255
        depth = read_depth(basedir, index=index)
        return poses[index], bds[index], img, depth

    imgs = [imageio.imread(f, ignoregamma=True)[...,:3].astype(np.float32)/255 for f in imgfiles]
    imgs = np.stack(imgs, 0)

    depths = np.stack(read_depth(basedir), 0)
    
    return poses, bds, imgs, depths


def gen_poses(basedir, match_type, format='.bin'):
    
    files_needed = [f + format for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    poses, pts3d, perm = load_colmap_data(basedir)    
    save_poses(basedir, poses, pts3d, perm)
    print( 'Done with imgs2poses' )


def read_depth(root, index=None):
    root = os.path.join(root, 'stereo/depth_maps')
    depth_paths = []
    for f in sorted(os.listdir(root)):
        if f.endswith('geometric.bin'):
            depth_paths.append(os.path.join(root, f))

    if index is not None:
        path = depth_paths[index]
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                    usecols=(0, 1, 2), dtype=int)
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        array = np.transpose(array, (1, 0, 2)).squeeze()
        return array

    depth_maps = []
    for path in depth_paths:
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                    usecols=(0, 1, 2), dtype=int)
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        array = np.transpose(array, (1, 0, 2)).squeeze()
        depth_maps.append(array)
    return depth_maps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='A wrapper to parse COLMAP outputs')
    parser.add_argument('colmap_dir', help='path to COLMAP folder')
    parser.add_argument('--input-format', choices=['.bin', '.txt'],
                        help='input model format', default='.bin')
    parser.add_argument('--matcher', help='COLMAP matcher to use', default='sequential_matcher')
    args = parser.parse_args()

    import colmap_read_model as read_model

    gen_poses(args.colmap_dir, args.matcher, format=args.input_format)

