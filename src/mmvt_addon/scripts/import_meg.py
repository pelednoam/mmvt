import sys
import os

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su

STAT_AVG, STAT_DIFF = range(2)


def wrap_blender_call():
    args = read_args()
    su.call_script(__file__, args)


def read_args(argv=None):
    parser = su.add_default_args()
    parser.add_argument('--stat', help='conds stat', required=False, default=STAT_DIFF)
    return su.parse_args(parser, argv)


def import_meg(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    mmvt.add_data_to_parent_brain_obj(args.stat)
    mmvt.add_data_to_brain()
    mmvt.set_render_output_path = su.get_figures_dir(args)
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print('Must specify flags!')
    elif sys.argv[2] == '--background':
        subject_fname = sys.argv[1]
        import_meg(subject_fname)
    else:
        wrap_blender_call()
