import argparse, os, subprocess as sp
import numpy as np
import imageio as io, cv2
import colour_demosaicing as cd
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
	ap.add_argument('-n', '--noise', type=float, help='Standard deviation of noise to add to HR frames')

	args = ap.parse_args()
	dataset = os.path.join(os.path.dirname(args.video), '..')
	scene = os.path.splitext(os.path.basename(args.video))[0]
	scene = f'{scene}_{args.operator}_{args.scale}x'
	for f in ('hr_frames', 'lr_frames', 'lr_frames_demosaiced', 'hr_frames_noisy'):
		r_dir = os.path.join(dataset, f, scene)
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

	:img: input high-res bayer image
	:reduction: the reduction operation (mean or sum)
	:scale: downsampling factor
	:return: Unpacked 4-channel image (RGGB) downsampled by given factor
	"""

	# TODO: batch binning?
	bayer = len(img.shape) == 2
	if bayer:
		# Just call binning on each channel
		# Assuming RGGB bayer pattern
		h, w = img.shape
		binned = np.empty((h//scale//2, w//scale//2, 4))
		binned[...,0] = binning(img[::2,::2,None], reduction, scale).squeeze()
		binned[...,1] = binning(img[::2,1::2,None], reduction, scale).squeeze()
		binned[...,2] = binning(img[1::2,::2,None], reduction, scale).squeeze()
		binned[...,3] = binning(img[1::2,1::2,None], reduction, scale).squeeze()
	else:
		h, w, c = img.shape
		reds = {'mean': np.mean, 'sum': np.sum}
		binned = img.reshape(h//scale, scale, w//scale, scale, c)
		binned = reds[reduction](reds[reduction](binned, 1), -2)

	return binned

def cv_resize(img, flag, scale):
	"""
	Wrapped function to unpack the image and call cv2.resize

	:img: input high-res bayer image
	:flag: cv2 flag indicating interpolation mode
	:scale: downsampling factor
	:return: 4-channel image (RGGB) downsampled by given factor
	"""

	h, w = img.shape[:2]
	resized = np.empty((h//scale//2, w//scale//2))
	resized[...,0] = cv2.resize(bayer[::2,::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=flag)
	resized[...,1] = cv2.resize(bayer[::2,1::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=flag)
	resized[...,2] = cv2.resize(bayer[1::2,::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=flag)
	resized[...,3] = cv2.resize(bayer[1::2,1::2], (0, 0), fx=1/scale, fy=1/scale, interpolation=flag)

	return resized

def pack_demosaic(img):
	"""
	Pack the data into bayer format and perform demosaicing

	:img: 4-channel image (RGGB) downsampled by given factor
	:return: demosaiced RGB image
	"""

	h, w, _ = img.shape
	bayer = np.empty((h*2, w*2))
	bayer[::2,::2] = img[...,0]
	bayer[::2,1::2] = img[...,1]
	bayer[1::2,::2] = img[...,2]
	bayer[1::2,1::2] = img[...,3]

	demosaiced = cd.demosaicing_CFA_Bayer_bilinear(bayer)
	return demosaiced

if __name__ == '__main__':
	args, (dataset, scene) = get_args()

	print(f'Reading video: {args.video}')
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
		hr = (np.clip(hr, 0, 1) * 255).astype(np.uint8)
		filename = os.path.join(dataset, 'hr_frames', scene, f'frame_{i+1:05d}.png')
		io.imwrite(filename, hr)

		if args.noise:
			# Write noisy HR frame
			noisy = hr + np.random.normal(0, args.noise, hr.shape)
			noisy = np.clip(noisy, 0, 255).astype(np.uint8)
			filename = os.path.join(dataset, 'hr_frames_noisy', scene, f'frame_{i+1:05d}.png')
			io.imwrite(filename, noisy)

		# TODO: automatic cropping to nearest multiple of "scale"
		h, w = bayer.shape
		scale = args.scale
		assert h % (scale*2) == 0 and w % (scale*2) == 0, 'Pick lower scaling value'
		if args.operator == 'binning':
			lr = binning(bayer, args.reduction, scale)
		else:
			operator = getattr(cv2, f'INTER_{args.operator.upper()}')
			lr = cv_resize(bayer, operator, scale)
		lr_rgb = pack_demosaic(lr)

		# Write the LR frame
		lr = (np.clip(lr, 0, 1) * 255).astype(np.uint8)
		filename = os.path.join(dataset, 'lr_frames', scene, f'frame_{i+1:05d}.png')
		io.imwrite(filename, lr)
		lr_rgb = (np.clip(lr_rgb, 0, 1) * 255).astype(np.uint8)
		filename = os.path.join(dataset, 'lr_frames_demosaiced', scene, f'frame_{i+1:05d}.png')
		io.imwrite(filename, lr_rgb)

	# Save HR and LR videos too
	fps = '30'
	crf = '18'
	dump = open(os.devnull, 'w')	# Disable this for debugging

	ffmpeg_cmd = ['ffmpeg', '-framerate', fps, '-i',
				  os.path.join(dataset, 'hr_frames', scene, f'frame_%5d.png'),
				  '-c:v', 'libx264', '-preset', 'veryslow', '-crf', crf, '-y',
				  os.path.join(dataset, 'hr_frames', 'videos', f'{scene}.avi')]
	sp.check_output(ffmpeg_cmd, stdin=sp.PIPE, stderr=dump)

	ffmpeg_cmd = ['ffmpeg', '-framerate', fps, '-i',
				  os.path.join(dataset, 'lr_frames_demosaiced', scene, f'frame_%5d.png'),
				  '-c:v', 'libx264', '-preset', 'veryslow', '-crf', crf, '-y',
				  os.path.join(dataset, 'lr_frames_demosaiced', 'videos', f'{scene}.avi')]
	sp.check_output(ffmpeg_cmd, stdin=sp.PIPE, stderr=dump)
