#%%
import argparse, os, copy
import numpy as np, cv2, HDRutils
import matplotlib.pyplot as plt
from poses.pose_utils import load_data
%matplotlib inline

np.set_printoptions(suppress=True, precision=2)
EPS = 1e-6


colmap = '/anfs/gfxdisp/video/adobe240f/hr_frames_noisy/IMG_0177_binning_4x/dense'
poses, bds, imgs, depths = load_data(colmap)

#%% Preprocess camera matrices and obtain data
def unpack_matrices(pose_vec):
    K = np.eye(4, dtype=np.float32)
    K[0,0] = pose_vec[0,2,4]
    K[1,1] = pose_vec[0,2,4]
    K[0,2] = pose_vec[0,0,5]
    K[1,2] = pose_vec[0,1,5]
    cam2world = np.zeros((pose_vec.shape[0], 4, 4), dtype=np.float32)
    cam2world[:,:3,:] = pose_vec[...,:4]
    cam2world[:,3,3] = 1
    return K, np.linalg.inv(K), cam2world, np.linalg.inv(cam2world)

def pack_coords_grid(N, h, w, near, far, l=3):
    '''
    In camera space, set (u,v,1,1/z)
    dims: frames, height, width, frustum_pts, homogenous_coords
    '''
    out_pts = np.ones((N,h,w,8,4), dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    # Top-left
    out_pts[...,0,0] = np.maximum(yy - l//2, 0)
    out_pts[...,0,1] = np.maximum(xx - l//2, 0)
    out_pts[...,4,0] = np.maximum(yy - l//2, 0)
    out_pts[...,4,1] = np.maximum(xx - l//2, 0)
    # Top-right
    out_pts[...,1,0] = np.maximum(yy - l//2, 0)
    out_pts[...,1,1] = np.minimum(xx + l//2, w-1)
    out_pts[...,5,0] = np.maximum(yy - l//2, 0)
    out_pts[...,5,1] = np.minimum(xx + l//2, w-1)
    # Bottom-left
    out_pts[...,2,0] = np.minimum(yy + l//2, h-1)
    out_pts[...,2,1] = np.maximum(xx - l//2, 0)
    out_pts[...,6,0] = np.minimum(yy + l//2, h-1)
    out_pts[...,6,1] = np.maximum(xx - l//2, 0)
    # Bottom-right
    out_pts[...,3,0] = np.minimum(yy + l//2, h-1)
    out_pts[...,3,1] = np.minimum(xx + l//2, w-1)
    out_pts[...,7,0] = np.minimum(yy + l//2, h-1)
    out_pts[...,7,1] = np.minimum(xx + l//2, w-1)

    out_pts[...,:4,3] = 1/near
    out_pts[...,4:,3] = 1/far

    return out_pts

print(poses.shape, depths.shape, imgs.shape)
K, K_inv, c2w, w2c = unpack_matrices(poses)

near, far = bds.min(), bds.max()
N, h, w = depths.shape
n = 3**2
cam_out_pts = pack_coords_grid(N, h, w, near, far, int(n**0.5))

# %% Get 3D frustum of first frame and project to all other frames
scene_pts = np.concatenate((near * c2w[0] @ K_inv @ cam_out_pts[0,...,:4,:,None],
                            far * c2w[0] @ K_inv @ cam_out_pts[0,...,4:,:,None]), axis=-3).squeeze()
cam_in_pts = np.concatenate((1/near * K @ w2c[:,None,None,None] @ scene_pts[None,...,4:,:,None],
                             1/far * K @ w2c[:,None,None,None] @ scene_pts[None,...,4:,:,None]), axis=-3).squeeze()

# Ideally the extra dimension should be the last one
# cam_in_pts = np.round(cam_in_pts / cam_in_pts[...,2,None]).astype(np.uint16)
cam_in_pts = np.round(cam_in_pts / cam_in_pts[...,2,None]).astype(np.int16)
cam_in_pts[...,0] = cam_in_pts[...,0].clip(0, h-1)
cam_in_pts[...,1] = cam_in_pts[...,1].clip(0, w-1)

# %% Collect all points in the input views
def gather(boxes):
    N, h, w, _ = boxes.shape
    m = (boxes[...,2] - boxes[...,0]).max()
    n = (boxes[...,3] - boxes[...,1]).max()
    grid = np.mgrid[:m,:n].transpose(1,2,0)[None,None,None].astype(np.float32)
    pts = grid.repeat(w, axis=2).repeat(h, axis=1).repeat(N, axis=0)
    pts[...,0] += boxes[...,None,None,0]
    pts[...,1] += boxes[...,None,None,1]
    for i in range(m):
        pts[boxes[...,2] - boxes[...,0] == i,i:] = -10000
    for j in range(n):
        pts[boxes[...,3] - boxes[...,1] == j,:,j:] = -10000
    pts = np.concatenate((pts, np.ones_like(pts)), axis=-1)
    return pts.reshape(N, h, w, m*n, 4)

bounding_boxes = np.stack((cam_in_pts[...,0].min(axis=-1),
                           cam_in_pts[...,1].min(axis=-1),
                           cam_in_pts[...,0].max(axis=-1),
                           cam_in_pts[...,1].max(axis=-1)), axis=-1)
gathered_in_pts = gather(bounding_boxes)
print(f'Found NaNs in {np.count_nonzero(gathered_in_pts == -10000) / gathered_in_pts.size * 2 * 100:0.2f}% of the data')

invalid = gathered_in_pts[...,0] == -10000
gathered_in_pts[invalid] = 1
# Can this loop be eliminated?
for cc in range(N):
    coords = gathered_in_pts[cc].astype(np.int16)
    gathered_in_pts[cc,...,:3] *= depths[cc,coords[...,0],coords[...,1],None]

invalid += (gathered_in_pts[...,0] == 0)
print(f'{np.count_nonzero(invalid) / invalid.size * 100:0.2f}% of the data is invalid')
gathered_in_pts[invalid] = 1
valid = np.logical_not(invalid)
# gathered_in_pts[valid,3] = 1/gathered_in_pts[valid,3]
gathered_in_pts[valid].max(axis=0)

# %%
gathered_scene_pts = ((c2w @ K_inv)[:,None,None,None] @ gathered_in_pts[:,...,None]).squeeze()
gathered_scene_pts[valid].max(axis=0)
gathered_scene_pts[invalid] = 0

normal1 = np.cross(scene_pts[...,2,:3] - scene_pts[...,0,:3], scene_pts[...,4,:3] - scene_pts[...,0,:3])
offset1 = -np.sum(normal1 * scene_pts[...,0,:3], axis=-1)
normal2 = np.cross(scene_pts[...,3,:3] - scene_pts[...,1,:3], scene_pts[...,5,:3] - scene_pts[...,1,:3])
offset2 = -np.sum(normal2 * scene_pts[...,1,:3], axis=-1)
normal3 = np.cross(scene_pts[...,1,:3] - scene_pts[...,0,:3], scene_pts[...,5,:3] - scene_pts[...,0,:3])
offset3 = -np.sum(normal3 * scene_pts[...,0,:3], axis=-1)
normal4 = np.cross(scene_pts[...,3,:3] - scene_pts[...,2,:3], scene_pts[...,6,:3] - scene_pts[...,2,:3])
offset4 = -np.sum(normal4 * scene_pts[...,2,:3], axis=-1)

# %% Filter out points that lie outside the frustum
filter = (((gathered_scene_pts[...,:3] * normal1[None,...,None,:]).sum(axis=-1) \
           + offset1[None,...,None]) * \
          ((gathered_scene_pts[...,:3] * normal2[None,...,None,:]).sum(axis=-1) \
           + offset2[None,...,None]) >= 0) + \
         (((gathered_scene_pts[...,:3] * normal3[None,...,None,:]).sum(axis=-1) \
           + offset3[None,...,None]) * \
          ((gathered_scene_pts[...,:3] * normal4[None,...,None,:]).sum(axis=-1) \
           + offset4[None,...,None]) >= 0)
print(f'{np.count_nonzero(filter + invalid) / invalid.size * 100:0.2f}% of the data is invalid')

# %%
valid = np.logical_not(filter + invalid)
rgb_xyz_t = np.ones((N,h,w,n,7), dtype=np.float32)
for cc in range(N):
    coords = (gathered_in_pts[cc,...,:2] * valid[cc,...,None] / gathered_in_pts[cc,...,2,None]).astype(np.int16)
    rgb_xyz_t[cc,...,:3] = imgs[cc,coords[...,0],coords[...,1]]
    rgb_xyz_t[cc,...,3:6] = gathered_scene_pts[cc,...,:3]
    rgb_xyz_t[cc,...,6] = cc

# %%
ref = np.concatenate((imgs[0], gathered_scene_pts[0,...,n//2,:3], np.zeros((h,w,1), dtype=np.float32)), axis=-1)
sigma_inv = np.diag(1/np.array([40, 40, 40, 10, 10 , 10, 6], dtype=np.float32))
weights = -(ref[None,...,None,:] - rgb_xyz_t)**2 / 2 @ sigma_inv**2
weights = np.exp(weights) * valid[...,None]
# weights = np.exp(weights.sum(axis=-1)[...,None]) * valid[...,None].repeat(7, axis=-1)
res = np.ma.average(rgb_xyz_t, weights=weights, axis=(0,3))
res[res.mask] = ref[res.mask]
fig, ax = plt.subplots(2,1)
ax[0].imshow(ref[...,:3])
ax[1].imshow(res[...,:3])

# %%
# HDRutils.imwrite('ref.exr', ref[...,:3].astype(np.float16))
HDRutils.imwrite('res_new.exr', res[...,:3].astype(np.float16))
# HDRutils.imwrite('depth.exr', depths[0].astype(np.float16))
# %%
