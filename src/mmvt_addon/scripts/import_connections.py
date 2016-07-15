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
    parser.add_argument('-t', '--type', help='connection type', required=False, default='')
    parser.add_argument('--threshold', help='connection threshold', required=False, default=0, type=int)
    args = su.parse_args(parser, argv)
    rois_file_exist = op.isfile(op.join(su.get_mmvt_dir(), args.subject, 'rois_con.npz'))
    electrodes_file_exist = op.isfile(op.join(su.get_mmvt_dir(), args.subject, 'electrodes_con.npz'))
    if args.type == '':
        if rois_file_exist and electrodes_file_exist:
            raise Exception('More than one connection file exist, please select one using the -t flag')
        elif rois_file_exist:
            args.type = 'rois'
        elif electrodes_file_exist:
            args.type = 'electrodes'
        else:
            raise Exception('No connection file was found!\n'+
                            'ROIs connection file: rois_con.npz\n' +
                            'Electrodes connection file: electrodes_con.npz')
    elif args.type == 'rois':
        if not rois_file_exist:
            raise Exception('You chose ROIs connection, but the file rois_con.npz does not exist!')
    elif args.type == 'electrodes':
        if not electrodes_file_exist:
            raise Exception('You chose electrodes connection, but the file electrodes_con.npz does not exist!')
    return args


def import_connections(subject_fname):
    args = read_args(su.get_python_argv())
    mmvt = su.init_mmvt_addon()
    mmvt.set_connection_type(args.type)
    mmvt.set_connections_threshold(args.threshold)
    mmvt.create_connections()
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        import_connections(subject_fname)
    else:
        wrap_blender_call()
