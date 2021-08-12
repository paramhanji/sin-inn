import os, argparse, struct, numpy as np, collections
import HDRutils as io

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D

def read_model(path, ext=""):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D" + ext))
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D" + ext))
    return cameras, images, points3D


def unpack_camera_matrices(colmap_dir, format='.bin', save_file=None):
    assert format in ('.bin', '.txt')
    cameras, images, points3D = read_model(path=colmap_dir, ext=format)

    # Read the single intrinsic matrix
    assert len(cameras) == 1, 'Specify single camera with flag "--ImageReader.single_camera 1" during feature extraction'
    K = np.eye(3)
    cam = cameras[next(iter(cameras))]
    if cam.model == 'SIMPLE_RADIAL':
        K[0,0] = cam.params[0]
        K[1,1] = cam.params[0]
        K[0,2] = cam.params[1]
        K[1,2] = cam.params[2]
        K[0,1] = cam.params[3]
    elif cam.model == 'PINHOLE':
        raise NotImplementedError
        K[0,0] = cam.params[0]
        K[1,1] = cam.params[1]
        K[0,2] = cam.params[2]
        K[1,2] = cam.params[3]
    else:
        raise NotImplementedError

    # Read extrinsic matrices
    extrinsics = np.empty((len(images),3,4))
    for i, id in enumerate(images):
        R = images[id].qvec2rotmat()
        T = images[id].tvec[:,None]
        extrinsics[i] = np.hstack((R,T))

    pts = []
    for p in points3D:
        pts.append(points3D[p].xyz)
    pts = np.array(pts)
    z = np.sum(-(pts[:,None,:].transpose(2,0,1) - extrinsics[:3,3:4])*extrinsics[:3,2:3], 0)
    print(z.max(), z.mean(), z.min())

    if save_file:
        # Pack everything into a single np array
        intrinsics = np.hstack((K, np.array([cam.height, cam.width, 0])[:,None]))
        combined = np.concatenate((intrinsics[None], extrinsics))
        np.save(save_file, combined)
    else:
        return K, extrinsics, (cam.height, cam.width)


def get_camera_matrices(colmap_dir=None, format='.bin', file=None):
    assert file is None or file.endswith('.npy')

    if file is not None and os.path.exists(file):
        print(f'Loading matrices from: {file}')
        matrices = np.load(file)
        K = matrices[0][:3,:3]
        extrinsics = matrices[1:]
    else:
        print('Reading matrices from colmap output')
        assert colmap_dir is not None
        colmap_dir = os.path.join(colmap_dir, 'sparse/0')
        K, extrinsics, _ = unpack_camera_matrices(colmap_dir, format=format, save_file=file)

    return K, extrinsics


def read_depth(root):
    root = os.path.join(root, 'dense/stereo/depth_maps')
    depth_maps = []
    for f in os.listdir(root):
        print(f)
        if f.endswith('geometric.bin'):
            path = os.path.join(root, f)
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
            depth_maps.append(array)
    return depth_maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A wrapper to parse COLMAP outputs')
    parser.add_argument('operation', choices=['read_matrices', 'depth_information', 'reproject'])
    parser.add_argument('--colmap-dir', help='path to COLMAP folder', default=None)
    parser.add_argument('--input-format', choices=['.bin', '.txt'],
                        help='input model format', default='.bin')
    parser.add_argument('--matrices-file', help='path to saved .npy file', default=None)
    args = parser.parse_args()

    if args.operation == 'read_matrices':
        get_camera_matrices(args.colmap_dir, args.input_format, args.matrices_file)
    elif args.operation == 'depth_information':
        assert args.colmap_dir is not None
        depth = read_depth(args.colmap_dir)
        print(depth.min(), depth.max(), depth.mean(), depth.shape)
        io.imwrite('depth.exr', depth.astype(np.float16))
    elif args.operation == 'reproject':
        view1 = os.path.join(args.colmap_dir, 'images/frame_00522.png')
        view1 = io.imread(view1)
        # view2 = os.path.join(args.colmap_dir, 'images/frame_00522.png')
        K, extrinsics = get_camera_matrices(args.colmap_dir, args.input_format, args.matrices_file)
        cam2world = np.linalg.pinv(K@extrinsics[522])
        depth = np.zeros_like(view1, dtype=np.float32)
        h, w, _ = view1.shape
        for i in range(h):
            for j in range(w):
                P = cam2world @ np.array([i,j,1])
                depth[i,j] = P[2] / P[3]
        depth -= depth.min()
        print(depth.min(), depth.max(), depth.mean(), depth.shape)
        io.imwrite('my_depth.exr', depth.astype(np.float16))
