import sys
import os
import os.path as op
import shutil

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    # Add current folder the imports path
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    args = read_args()
    create_new_subject_file(args)
    su.call_script(__file__, args)


def create_new_subject_file(args):
    # Create a file for the new subject
    new_fname = su.get_subject_fname(args)
    empty_subject_fname = op.join(su.get_mmvt_dir(), 'empty_subject.blend')
    if not op.isfile(empty_subject_fname):
        shutil.copy(op.join(su.get_resources_dir(), 'empty_subject.blend'), empty_subject_fname)
    if op.isfile(new_fname):
        overwrite = input('The file {} already exist, do you want to overwrite? '.format(new_fname))
        if su.is_true(overwrite):
           os.remove(new_fname)
           shutil.copy(op.join(su.get_mmvt_dir(), 'empty_subject.blend'), new_fname)


def read_args(argv=None):
    parser = su.add_default_args()
    return su.parse_args(parser, argv)


def import_meg(subject_fname):
    mmvt = su.init_mmvt_addon()
    mmvt.import_brain()
    su.save_blend_file(subject_fname)
    su.exit_blender()


if __name__ == '__main__':
    import sys
    subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        import_meg(subject_fname)
    else:
        wrap_blender_call()
