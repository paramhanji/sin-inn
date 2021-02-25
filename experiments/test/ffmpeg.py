import argparse, os.path as osp, re
import subprocess as sp

def get_args():
	ap = argparse.ArgumentParser(description='Create a video collage of 4 configurations.' \
		'Input scenes must contain a folder "videos" with 3 videos: gt.avi, in.avi and out.avi')
	ap.add_argument('folders', nargs='+',
					help='3 inputs: add GT; 2 inputs: add LR,GT; 1 input:add EDVR,LR,GT')
	ap.add_argument('-s', '--scene', default='IMG_0028')
	ap.add_argument('-f', '--fps', type=int, help='set output framerate')
	ap.add_argument('-c', '--crf', type=int, default=18,
					help='h264 crf controlling quailty: 0 is lossless but results in large files')
	ap.add_argument('-o', '--outfile', default='combined.avi',
					help='filename for output video')
	ap.add_argument('-i', '--images', action='store_true',
					help='Boolean flag indicating that scenes contain images instead of ' \
					'videos. Each scene must contain ground truth frames (gt_0000.png, ' \
					'gt_0001.png, ...), input frames (in_0000.png, in_0001.png, ...) and ' \
					'output frames (out_0000.png, out_0001.png) in the same folder.')

	return ap.parse_args()

def frames_to_videos(folder, opt):
	if opt.fps:
		fps = str(opt.fps)
	else:
		fps = '30'
	crf = str(opt.crf)

	videos_dir = osp.join(folder, 'videos')
	if not osp.isdir(videos_dir):
		os.mkdir(videos_dir)

	# Assume folder contains ground truth frames (gt_0000.png, gt_0001.png, ...),
	# input frames (in_0000.png, in_0001.png, ...) and output frames (out_0000.png, out_0001.png)
	dump = open(os.devnull, 'w')	# Disable this for debugging
	for prefix in ('gt', 'in', 'out'):
		ffmpeg_cmd = ['ffmpeg', '-framerate', fps, '-i',
					  osp.join(folder, f'{prefix}_%04d.png'),
					  '-c:v', 'libx264', '-preset', 'veryslow', 'crf', crf, '-y',
					  osp.join(videos_dir, f'{prefix}_.avi')]
		sp.check_output(ffmpeg_cmd, stdin=sp.PIPE, stderr=dump)

def get_filler_videos(n, folder):
	inputs, display = [], []
	if n == 3:
		inputs.append('edvr.avi')
		display.append('EDVR')
	if n >= 2:
		inputs.append(osp.join(osp.dirname(folder), 'in.avi'))
		display.append('Input')
	if n >= 1:
		inputs.append(osp.join(osp.dirname(folder), 'gt.avi'))
		display.append('GT')
	return inputs, display


if __name__ == '__main__':
	args = get_args()

	# If scene contains frames, first combine to form videos
	if args.images:
		for f in args.folder:
			frames_to_videos(args.folder, args)

	# Get input videos and generate display names
	inputs, display = [], []
	for f in args.folders:
		inputs.append(osp.join(f, 'videos', 'out.avi'))
		f = osp.basename(f.rstrip('/'))
		f = f.replace(f'{args.scene}_', '')
		f = re.sub('_epoch_\d{5}$', '', f)
		display.append(f)
	filler_in, filler_dis = get_filler_videos(4 - len(inputs), inputs[0])
	inputs += filler_in
	display += filler_dis
	assert len(inputs) == 4 and len(display) == 4

	if args.fps:
		opt_out = [f'-vsync 0 -r {args.fps}', args.outfile]
	else:
		opt_out = [args.outfile]
	opt_txt = '; '.join([f'[{i}]drawtext=text={text}:fontsize=20:x=10:y=10:fontcolor=white[v{i}]'
						 for i,text in enumerate(display)] \
			  + ['[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]'])

	ffmpeg_cmd = ['ffmpeg'] + sum([['-i', f] for f in inputs], []) + ['-filter_complex', opt_txt] \
				 + ['-map', '[v]', '-c:v', 'libx264', '-preset', 'veryslow', '-crf', str(args.crf), '-y'] \
				 + opt_out

	# print(ffmpeg_cmd)
	sp.check_output(ffmpeg_cmd)