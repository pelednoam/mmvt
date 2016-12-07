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
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    # Call mmvt functions
    mmvt.clear_colors()
    mmvt.find_fmri_files_min_max()
    mmvt.set_brain_transparency(0)
    mmvt.set_light_layers_depth(0)
    inflated = False
    inflated_ratio = 0.5
    background_color= 'black'
    quality = 60
    clusters_names = [f for f in mmvt.get_clusters_file_names() if 'spm' in f]
    print('clusters_names: {}'.format(clusters_names))
    for clusters_file_name in clusters_names:
        mmvt.set_fmri_clusters_file_name(clusters_file_name)
        mmvt.plot_all_blobs()
        image_name = ['lateral_lh', 'lateral_rh', 'medial_lh', 'medial_rh']
        camera = [op.join(su.get_mmvt_dir(), args.subject, 'camera', 'camera_{}{}.pkl'.format(
            camera_name, '_inf' if inflated else '')) for camera_name in image_name]
        image_name = ['{}_{}_{}_{}'.format(clusters_file_name, name, 'inflated_{}'.format(
            inflated_ratio) if inflated else 'pial', background_color) for name in image_name]
        mmvt.render_image(image_name, quality=quality, render_background=False,
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
