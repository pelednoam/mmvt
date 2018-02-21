import sys
import os
import os.path as op


try:
    from src.mmvt_addon.scripts import scripts_utils as su
    from src.mmvt_addon.scripts import save_views 
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su
    import save_views


def wrap_blender_call(args=None):
    if args is None:
        args = read_args()
    args = pre_blender_call(args)
    if args.save_views:
        if args.img_name_prefix == '' and args.use_fol_for_img_prefix:
            args.img_name_prefix = su.namebase(su.get_parent_fol(args.nii))
        args = save_views.pre_blender_call(args)
    su.call_script(__file__, args, run_in_background=False, err_pipe=sys.stdin)
    if args.save_views:
        save_views.post_blender_call(args)


def read_args(argv=None):
    parser = su.add_default_args()
    parser = add_args(parser)
    parser = save_views.add_args(parser)
    return su.parse_args(parser, argv)


def add_args(parser):
    parser.add_argument('--nii', help='Full file name of the nii file', required=True, type=str)
    parser.add_argument('--threshold', help='Threshold', required=False, default=2, type=float)
    parser.add_argument('--use_abs_threshold', help='Use abs threshold', required=False, default=True, type=su.is_true)
    parser.add_argument('--cb_min_max', help='Colorbar min max values (default: take the min and abs from the data)',
                        required=False, default=None, type=su.float_arr_type)
    parser.add_argument('--cm', help='Colormap (default RdOrYl)', required=False, default='RdOrYl')
    parser.add_argument('--save_views', help='Save views', required=False, default=False, type=su.is_true)
    parser.add_argument('--fmri_file_template', help='For inner usage', required=False, default='', type=str)
    parser.add_argument('--img_name_prefix',
                        help='Images name prefix (if empty, you can use the use_fol_for_img_prefix flag)',
                        required=False, default='')
    parser.add_argument('--use_fol_for_img_prefix', help='Use the nii folder name as the images prefix', required=False,
                        default=False, type=su.is_true)
    return parser


def pre_blender_call(args):
    from src.mmvt_addon import load_results_panel
    from src.preproc import fMRI as fmri
    from src.utils import preproc_utils as pu
    from src.utils import utils

    user_fol = op.join(pu.MMVT_DIR, args.subject)
    nii_fname = args.nii
    if not op.isfile(args.nii):
        args.nii = op.join(user_fol, 'fmri', nii_fname)
    if not op.isfile(args.nii):
        fmri_fol = op.join(utils.get_link_dir(utils.get_links_dir(), 'fMRI'), args.subject)
        args.nii = op.join(fmri_fol, nii_fname)
    if not op.isfile(args.nii):
        raise Exception("Can't find the nii file!")
    fmri_file_template, _, _ = load_results_panel.load_surf_files(args.nii, run_fmri_preproc=False, user_fol=user_fol)
    preproc_args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='load_surf_files',
        fmri_file_template=fmri_file_template,
        ignore_missing=True
    ))
    ret = pu.run_on_subjects(preproc_args, fmri.main)
    if ret:
        load_results_panel.clean_nii_temp_files(fmri_file_template, user_fol)
        args.fmri_file_template = fmri_file_template
    else:
        raise Exception("Couldn't load the surface files!")
    return args


def wrap_mmvt_calls(subject_fname):
    args = read_args(su.get_python_argv())
    if args.debug:
        su.debug()
    mmvt = su.init_mmvt_addon()
    mmvt_calls(mmvt, args, subject_fname)
    if args.save_views:
        save_views.mmvt_calls(mmvt, args, subject_fname)
    su.save_blend_file(subject_fname)
    su.exit_blender()


def mmvt_calls(mmvt, args, subject_fname):
    mmvt.set_threshold(args.threshold)
    mmvt.set_use_abs_threshold(args.use_abs_threshold)
    if args.cb_min_max is not None:
        cb_min, cb_max = args.cb_min_max
        mmvt.set_colorbar_max_min(cb_max, cb_min)
    mmvt.set_colormap(args.cm)
    mmvt.show_hemis() # Otherwise it'll plot only on the not-hidden hemis
    mmvt.plot_fmri_file(args.fmri_file_template)


if __name__ == '__main__':
    import sys
    if op.isfile(sys.argv[0]) and sys.argv[0][-2:] == 'py':
        wrap_blender_call()
    else:
        wrap_mmvt_calls(sys.argv[1])
