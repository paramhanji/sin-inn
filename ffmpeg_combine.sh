SCENE=experiments/test/GOPR9653_10_epoch_03000/videos

ffmpeg -i $SCENE/out.avi -i $SCENE/edvr.avi -i $SCENE/gt.avi -i $SCENE/in.avi -filter_complex \
"[0]drawtext=text='INN':fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2[v0]; \
 [1]drawtext=text='EDVR':fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2[v1]; \
 [2]drawtext=text='GT':fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2[v2]; \
 [3]drawtext=text='INPUT':fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2[v3]; \
 [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" \
-map "[v]" -c:v libx264 -preset ultrafast -y $SCENE/combined.avi