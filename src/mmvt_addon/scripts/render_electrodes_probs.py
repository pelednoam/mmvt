import sys
import os
import os.path as op
import glob

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
    parser.add_argument('-b', '--bipolar', help='bipolar', required=True, type=su.is_true)
    parser.add_argument('-q', '--quality', help='render quality', required=False, default=60, type=int)
    parser.add_argument('--labeling', help="electrodes' labeling file name", required=False)
    parser.add_argument('--output_path', help='output path', required=False, default='electrodes_labeling')
    parser.add_argument('--rel_output_path', help='relative output path', required=False, default=True, type=su.is_true)
    parser.add_argument('--smooth_figure', help='smooth figure', required=False, default=False, type=su.is_true)
    parser.add_argument('--hide_lh', help='hide left hemi', required=False, default=False, type=su.is_true)
    parser.add_argument('--hide_rh', help='hide right hemi', required=False, default=False, type=su.is_true)
    parser.add_argument('--hide_subs', help='hide sub corticals', required=False, default=False, type=su.is_true)
    return su.parse_args(parser, argv)


def render_electrodes_probs(subject_fname):
    args = read_args(su.get_python_argv())
    figures_dir = su.get_figures_dir(args)
    if args.rel_output_path:
        args.output_path = op.join(figures_dir, args.output_path)
    su.make_dir(args.output_path)

    # Set the labeling file
    labeling_fname = '{}_{}_electrodes_cigar_r_*_l_*{}*.pkl'.format(args.subject, args.real_atlas,
        '_bipolar' if args.bipolar else '')
    labels_fol = op.join(su.get_mmvt_dir(), args.subject, 'electrodes')
    labeling_files = glob.glob(op.join(labels_fol, labeling_fname))
    if len(labeling_files) == 0:
        print('No labeling files in {}!'.format(labels_fol))
        return
    if len(labeling_files) > 1:
        print('More than one labeling files in {}, please choose one using the --labeling flag'.format(labels_fol))
        return
    else:
        labeling_file = labeling_files[0]

    mmvt = su.init_mmvt_addon()
    mmvt.show_hide_hemi(args.hide_lh, 'lh')
    mmvt.show_hide_hemi(args.hide_rh, 'rh')
    mmvt.show_hide_sub_corticals(args.hide_subs)
    mmvt.set_render_quality(args.quality)
    mmvt.set_render_output_path(args.output_path)
    mmvt.set_render_smooth_figure(args.smooth_figure)
    if op.isfile(op.join(args.output_path, 'camera.pkl')):
        mmvt.load_camera()
    else:
        # Try to find the camera in the figures folder
        if op.isfile(op.join(figures_dir, 'camera.pkl')):
            mmvt.set_render_output_path(figures_dir)
            mmvt.load_camera()
            mmvt.set_render_output_path(args.output_path)
        else:
            cont = input('No camera file was detected in the output folder, continue?')
            if not su.is_true(cont):
                return

    mmvt.set_electrodes_labeling_file(labeling_file)
    mmvt.show_electrodes()
    mmvt.color_the_relevant_lables(True)
    leads = mmvt.get_leads()
    for lead in leads:
        electrodes = mmvt.get_lead_electrodes(lead)
        for electrode in electrodes:
            print(electrode)
            mmvt.clear_cortex()
            mmvt.set_current_electrode(electrode, lead)
            mmvt.render_image(electrode)

    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        render_electrodes_probs(subject_fname)
    else:
        wrap_blender_call()
