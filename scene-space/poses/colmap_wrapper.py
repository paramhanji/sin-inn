import os
import subprocess


def run_colmap(basedir, match_type):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    colmap_exe = '/home/pmh64/sw/usr/local/bin/colmap'

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
    print('Sparse reconstruction done')

    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )

