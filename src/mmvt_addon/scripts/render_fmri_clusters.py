import sys
import os
import os.path as op


try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    su.call_script(__file__, args)


def read_args(argv=None):
    parser = su.add_default_args()
    # Add more args here
    return su.parse_args(parser, argv)


def run_script(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    # Call mmvt functions
    mmvt.clear_colors()
    mmvt.find_fmri_files_min_max()
    inflated = False
    inflated_ratio = 0.5
    background_color= 'black'
    quality = 20
    for clusters_file_name in mmvt.get_clusters_file_names():
        mmvt.set_fmri_clusters_file_name(clusters_file_name)
        mmvt.plot_all_blobs()
        image_name = ['lateral_lh', 'lateral_rh', 'medial_lh', 'medial_rh']
        camera = [op.join(su.get_mmvt_dir(), args.subject, 'camera', 'camera_{}{}.pkl'.format(
            camera_name, '_inf' if inflated else '')) for camera_name in image_name]
        image_name = ['{}_{}_{}'.format(name, 'inflated_{}'.format(inflated_ratio) if inflated else 'pial',
                                        background_color) for name in image_name]
        # camera = ','.join(camera)
        # image_name = ','.join(image_name)
        mmvt.render_image(quality=quality, render_background=False,
                          camera_fname=camera, hide_subcorticals=True)
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        run_script(subject_fname)
    else:
        wrap_blender_call()
