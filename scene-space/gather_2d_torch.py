import numpy as np, HDRutils
import matplotlib.pyplot as plt
import torch
# %matplotlib inline

from data import ImagesData
import gfxdisp
v = gfxdisp.pfs.pfs()

colmap = '/anfs/gfxdisp/video/adobe240f/hr_frames_noisy/IMG_0177_binning_4x/dense'
N = 100
side = 25
n = side**2
ref = 5

def pack_coords(h, w, near, far, l=3):
    '''
    In camera space, set (u,v,1,1/z)
    dims: frames, height, width, frustum_pts, homogenous_coords
    '''
    out_pts = torch.ones((h,w,8,4))
    xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w))

    out_pts[...,:2,0] = torch.clamp(yy - l//2, min=0).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,4:6,0] = torch.clamp(yy - l//2, min=0).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,:3:2,1] = torch.clamp(xx - l//2, min=0).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,4:7:2,1] = torch.clamp(xx - l//2, min=0).unsqueeze(-1).repeat(1,1,2)

    out_pts[...,1:4:2,1] = torch.clamp(xx + l//2, max=w-1).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,5:8:2,1] = torch.clamp(xx + l//2, max=w-1).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,2:4,0] = torch.clamp(yy + l//2, max=h-1).unsqueeze(-1).repeat(1,1,2)
    out_pts[...,6:8,0] = torch.clamp(yy + l//2, max=h-1).unsqueeze(-1).repeat(1,1,2)

    # First 4 pts on near plane, next 4 on far plane
    out_pts[...,:4,3] = 1/near
    out_pts[...,4:,3] = 1/far

    return out_pts

video = ImagesData(colmap, N)
ref_c2w, ref_bds, ref_img, ref_depth = video[ref]

ref_near, ref_far = ref_bds.min(), ref_bds.max()
h, w = ref_depth.shape

# Generate 2d coordinate grid and get corresponding 3d points
ref_2d = pack_coords(h, w, ref_near, ref_far, side)
ref_3d = torch.cat((ref_c2w @ video.K_inv @ ref_2d[...,:4,:,None],
                    ref_c2w @ video.K_inv @ ref_2d[...,4:,:,None]), dim=-3).squeeze()

# Collect all points within convex hull in the input views
# def points_in_convex_polygon(points, polygon, clockwise=True):
#     """check points is in convex polygons. may run 2x faster when write in
#     cython(don't need to calculate all cross-product between edge and point)
#     Args:
#         points: [num_points, 2] array.
#         polygon: [num_polygon, num_points_of_polygon, 2] array.
#         clockwise: bool. indicate polygon is clockwise.
#     Returns:
#         [num_points, num_polygon] bool array.
#     """
#     # first convert polygon to directed lines
#     num_lines = polygon.shape[1]
#     polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
#     if clockwise:
#         vec1 = (polygon - polygon_next)[np.newaxis, ...]
#     else:
#         vec1 = (polygon_next - polygon)[np.newaxis, ...]
#     vec2 = polygon[np.newaxis, ...] - points[:, np.newaxis, np.newaxis, :]
#     # [num_points, num_polygon, num_points_of_polygon, 2]
#     cross = np.cross(vec1, vec2)
#     return np.all(cross > 0, axis=2)

# For now, work with a simple bounding box
def gather(boxes):
    boxes = boxes.numpy()
    h, w, _ = boxes.shape
    m = (boxes[...,2] - boxes[...,0]).max()
    n = (boxes[...,3] - boxes[...,1]).max()
    grid = np.mgrid[:m,:n].transpose(1,2,0)[None,None]
    pts = grid.repeat(w, axis=1).repeat(h, axis=0).astype(np.float32)
    pts[...,0] += boxes[...,None,None,0]
    pts[...,1] += boxes[...,None,None,1]
    for i in range(m):
        pts[boxes[...,2] - boxes[...,0] == i,i:] = -10000
    for j in range(n):
        pts[boxes[...,3] - boxes[...,1] == j,:,j:] = -10000
    pts = np.concatenate((pts, np.ones_like(pts)), axis=-1)
    return pts.reshape(h, w, m*n, 4)

imgs = []
for c2w, _, img, depth in video:
    w2c = torch.inverse(c2w)
    # Project 3d points to all other frames
    frame_2d = ((video.K @ w2c)[None,None,None] @ ref_3d[...,None]).squeeze()
    frame_2d = (frame_2d / frame_2d[...,2,None]).round().int()
    frame_2d[...,0] = frame_2d[...,0].clamp(0, h-1)
    frame_2d[...,1] = frame_2d[...,1].clamp(0, w-1)

    # Use bounding boxes at most 5x the neighborhood
    bounding_boxes = torch.stack((frame_2d[...,0].min(dim=-1)[0],
                                  frame_2d[...,1].min(dim=-1)[0],
                                  frame_2d[...,0].max(dim=-1)[0],
                                  frame_2d[...,1].max(dim=-1)[0]), dim=-1)

    bb_center = torch.stack(((bounding_boxes[...,2] + bounding_boxes[...,0]) / 2,
                           (bounding_boxes[...,3] + bounding_boxes[...,1]) / 2), dim=-1)
    print(bb_center[0].shape, bounding_boxes[...,::2].shape)
    # bounding_boxes[...,:2] = torch.max(bounding_boxes[...,:2], bb_center - side*5)
    # bounding_boxes[...,2:] = torch.min(bounding_boxes[...,2:], bb_center + side*5)
    print((bounding_boxes[...,2] - bounding_boxes[...,0]).max(), (bounding_boxes[...,3] - bounding_boxes[...,1]).max())
    # print(np.argmax(bounding_boxes[...,2] - bounding_boxes[...,0]), np.argmax(bounding_boxes[...,3] - bounding_boxes[...,1]))
    img[bounding_boxes[509,317,0]:bounding_boxes[509,317,2],bounding_boxes[509,317,1]:bounding_boxes[509,317,3]] = 0
    imgs.append(img)
    # gathered_in_pts = gather(bounding_boxes)
    # print(gathered_in_pts.shape)

v.view(*imgs)

# print(f'Found NaNs in {np.count_nonzero(gathered_in_pts == -10000) / gathered_in_pts.size * 2 * 100:0.2f}% of the data')

# invalid = gathered_in_pts[...,0] == -10000
# gathered_in_pts[invalid] = 1
# # Can this loop be eliminated?
# for cc in range(N):
#     coords = gathered_in_pts[cc].astype(np.int16)
#     gathered_in_pts[cc,...,:3] *= depths[cc,coords[...,0],coords[...,1],None]

# invalid += (gathered_in_pts[...,0] == 0)
# print(f'{np.count_nonzero(invalid) / invalid.size * 100:0.2f}% of the data is invalid')
# gathered_in_pts[invalid] = 1
# valid = np.logical_not(invalid)
# # gathered_in_pts[valid,3] = 1/gathered_in_pts[valid,3]
# gathered_in_pts[valid].max(axis=0)

# # %%
# gathered_scene_pts = ((c2w @ K_inv)[:,None,None,None] @ gathered_in_pts[:,...,None]).squeeze()
# gathered_scene_pts[valid].max(axis=0)
# gathered_scene_pts[invalid] = 0

# normal1 = np.cross(scene_pts[...,2,:3] - scene_pts[...,0,:3], scene_pts[...,4,:3] - scene_pts[...,0,:3])
# offset1 = -np.sum(normal1 * scene_pts[...,0,:3], axis=-1)
# normal2 = np.cross(scene_pts[...,3,:3] - scene_pts[...,1,:3], scene_pts[...,5,:3] - scene_pts[...,1,:3])
# offset2 = -np.sum(normal2 * scene_pts[...,1,:3], axis=-1)
# normal3 = np.cross(scene_pts[...,1,:3] - scene_pts[...,0,:3], scene_pts[...,5,:3] - scene_pts[...,0,:3])
# offset3 = -np.sum(normal3 * scene_pts[...,0,:3], axis=-1)
# normal4 = np.cross(scene_pts[...,3,:3] - scene_pts[...,2,:3], scene_pts[...,6,:3] - scene_pts[...,2,:3])
# offset4 = -np.sum(normal4 * scene_pts[...,2,:3], axis=-1)

# # %% Filter out points that lie outside the frustum
# filter = (((gathered_scene_pts[...,:3] * normal1[None,...,None,:]).sum(axis=-1) \
#            + offset1[None,...,None]) * \
#           ((gathered_scene_pts[...,:3] * normal2[None,...,None,:]).sum(axis=-1) \
#            + offset2[None,...,None]) >= 0) + \
#          (((gathered_scene_pts[...,:3] * normal3[None,...,None,:]).sum(axis=-1) \
#            + offset3[None,...,None]) * \
#           ((gathered_scene_pts[...,:3] * normal4[None,...,None,:]).sum(axis=-1) \
#            + offset4[None,...,None]) >= 0)
# print(f'{np.count_nonzero(filter + invalid) / invalid.size * 100:0.2f}% of the data is invalid')

# # %%
# valid = np.logical_not(filter + invalid)
# rgb_xyz_t = np.ones((N,h,w,n,7), dtype=np.float32)
# for cc in range(N):
#     coords = (gathered_in_pts[cc,...,:2] * valid[cc,...,None] / gathered_in_pts[cc,...,2,None]).astype(np.int16)
#     rgb_xyz_t[cc,...,:3] = imgs[cc,coords[...,0],coords[...,1]]
#     rgb_xyz_t[cc,...,3:6] = gathered_scene_pts[cc,...,:3]
#     rgb_xyz_t[cc,...,6] = cc

# # %%
# ref = np.concatenate((imgs[0], gathered_scene_pts[0,...,n//2,:3], np.zeros((h,w,1), dtype=np.float32)), axis=-1)
# sigma_inv = np.diag(1/np.array([40, 40, 40, 10, 10 , 10, 6], dtype=np.float32))
# weights = -(ref[None,...,None,:] - rgb_xyz_t)**2 / 2 @ sigma_inv**2
# weights = np.exp(weights) * valid[...,None]
# # weights = np.exp(weights.sum(axis=-1)[...,None]) * valid[...,None].repeat(7, axis=-1)
# res = np.ma.average(rgb_xyz_t, weights=weights, axis=(0,3))
# res[res.mask] = ref[res.mask]
# fig, ax = plt.subplots(2,1)
# ax[0].imshow(ref[...,:3])
# ax[1].imshow(res[...,:3])

# # %%
# # HDRutils.imwrite('ref.exr', ref[...,:3].astype(np.float16))
# HDRutils.imwrite('res_new.exr', res[...,:3].astype(np.float16))
# # HDRutils.imwrite('depth.exr', depths[0].astype(np.float16))
# # %%
