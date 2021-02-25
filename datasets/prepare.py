import argparse, os
import numpy as np
import imageio as io, cv2
# import colour_demosaicing as cd
from tqdm import tqdm

def get_args():
	ap = argparse.ArgumentParser(description='Extract HR and LR video frames from an input video')
	ap.add_argument('video', help='filename of input video')
	ap.add_argument('-d', '--downsampling', default=1, type=float,
					help='downsample video frames by a factor to ensure GT is noise free')
	ap.add_argument('-p','--operator', choices=['binning', 'linear', 'cubic', 'lanczos4', 'nearest', 'area'],
					default='binning', help='Downsampling operator to use')
	ap.add_argument('-r', '--reduction', choices=['mean', 'sum'], default='mean',
					help='select reduction operation (only for binning)')
	ap.add_argument('-s', '--scale', type=int, default=4)
	ap.add_argument('-b', '--bayer', action='store_true', help='set if input video contains bayer frames')

	args = ap.parse_args()
	dataset = os.path.join(os.path.dirname(args.video), '..')
	scene = os.path.splitext(os.path.basename(args.video))[0]
	scene = f'{scene}_{args.operator}_{args.scale}x'
	for r in ('hr', 'lr'):
		r_dir = os.path.join(dataset, f'{r}_frames', scene)
		if not os.path.isdir(r_dir):
			os.mkdir(r_dir)

	if args.bayer:
		# TODO: implement once we have a bayer video
		raise NotImplementedError

	return args, (dataset, scene)

def extract_bayer(frame, scale=1):
	"""
	If input video contains RGB frames, extract bayer data by sampling

	:reader: video frame
	:return: sampled bayer image and resized RGB frame of same hxw
	"""

	frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LANCZOS4)

	# Assuming RGGB bayer pattern
	bayer = np.empty(frame.shape[:2])
	bayer[::2,::2] = frame[::2,::2,0]			# R
	bayer[::2,1::2] = frame[::2,1::2,1]			# G1
	bayer[1::2,::2] = frame[1::2,::2,1]			# G2
	bayer[1::2,1::2] = frame[1::2,1::2,2]		# B

	return bayer, frame

def binning(img, reduction, scale):
	"""
	Simulate bayer binning according to this figure:
	https://en.ids-imaging.com/techtipp-details/techtip-binning-subsampling-or-scaler.html

	:img: input high-res image
	:reduction: the reduction operation (mean or sum)
	:scale: downsampling factor
	:return: downsampled image by given factor
	"""

	# TODO: batch binning?
	bayer = len(img.shape) == 2
	if bayer:
		# Just call binning on each channel
		# Assuming RGGB bayer pattern
		h, w = img.shape
		binned = np.empty((h//scale, w//scale))
		binned[::2,::2] = binning(img[::2,::2,None], reduction, scale).squeeze()
		binned[::2,1::2] = binning(img[::2,1::2,None], reduction, scale).squeeze()
		binned[1::2,::2] = binning(img[1::2,::2,None], reduction, scale).squeeze()
		binned[1::2,1::2] = binning(img[1::2,1::2,None], reduction, scale).squeeze()
	else:
		h, w, c = img.shape
		reds = {'mean': np.mean, 'sum': np.sum}
		binned = img.reshape(h//scale, scale, w//scale, scale, c)
		binned = reds[reduction](reds[reduction](binned, 1), -2)

	return binned


if __name__ == '__main__':
	args, (dataset, scene) = get_args()

	reader = io.get_reader(args.video)
	for i, frame in tqdm(enumerate(reader)):
		if frame.dtype == np.uint8:
			frame = frame / 255
		elif frame.dtype == np.uint16:
			frame = frame / (2**16 - 1)
		else:
			raise NotImplementedError

		if not args.bayer:
			bayer, hr = extract_bayer(frame, args.downsampling)

		# Write the HR frame
		filename = os.path.join(dataset, 'hr_frames', scene, f'frame_{i+1:05d}.png')
		io.imwrite(filename, (hr * 255).astype(np.uint8))

		# TODO: automatic cropping to nearest multiple of "scale"
		h, w = bayer.shape
		scale = args.scale
		assert h % (scale*2) == 0 and w % (scale*2) == 0, 'Pick lower scaling value'
		if args.operator == 'binning':
			lr = binning(bayer, args.reduction, scale)
		else:
			operator = getattr(cv2, f'INTER_{args.operator.upper()}')
			lr = np.empty((h//scale, w//scale))
			lr[::2,::2] = cv2.resize(bayer[::2,::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=operator)
			lr[::2,1::2] = cv2.resize(bayer[::2,1::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=operator)
			lr[1::2,::2] = cv2.resize(bayer[1::2,::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=operator)
			lr[1::2,1::2] = cv2.resize(bayer[1::2,1::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=operator)

		# Write the LR frame
		filename = os.path.join(dataset, 'lr_frames', scene, f'frame_{i+1:05d}.png')
		io.imwrite(filename, (lr * 255).astype(np.uint8))
