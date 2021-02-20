import argparse, os.path as osp, re
import subprocess as sp

def get_args():
	ap = argparse.ArgumentParser(description='Create a video collage of 4 configurations')
	ap.add_argument('folders', nargs='+',
					help='3 inputs: add GT; 2 inputs: add LR,GT; 1 input:add EDVR,LR,GT')
	ap.add_argument('-s', '--scene', default='IMG_0028')
	ap.add_argument('-f', '--fps', type=int, help='set output framerate')
	ap.add_argument('-c', '--crf', type=int, default=18,
					help='h264 crf controlling quailty: 0 is lossless')
	ap.add_argument('-o', '--outfile', default='combined.avi', help='filename for output video')

	return ap.parse_args()

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

	# Get input videos and generate display names
	inputs, display = [], []
	for f in args.folders:
		inputs.append(osp.join(f, 'videos', 'out.avi'))
		f = osp.basename(f.rstrip('/'))
		print(f, end=', ')
		f = f.replace(f'{args.scene}_', '')
		print(f, end=', ')
		f = re.sub('_epoch_\d{5}$', '', f)
		print(f)
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