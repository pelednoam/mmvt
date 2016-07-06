import sys
import os
import os.path as op

try:
    from src.mmvt_addon.scripts import scripts_utils as su
except:
    sys.path.append(os.path.split(__file__)[0])
    import scripts_utils as su


def wrap_blender_call():
    from src.utils import utils
    from src.utils import args_utils as au
    import argparse
    import shutil

    LINKS_DIR = utils.get_links_dir()
    MMVT_DIR = op.join(LINKS_DIR, 'mmvt')

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='dkt')
    parser.add_argument('--blender_fol', help='blender folder', required=False, default='')
    args = utils.Bag(au.parse_parser(parser))

    # Create a file for the new subject
    new_fname = op.join(MMVT_DIR, '{}_{}.blend'.format(args.subject, args.atlas))
    if op.isfile(new_fname):
        overwrite = input('The file {} already exist, do you want to overwrite? '.format(new_fname))
        if au.is_true(overwrite):
           os.remove(new_fname)
        else:
            return
    shutil.copy(op.join(MMVT_DIR, 'empty_subject.blend'), new_fname)
    su.call_script(__file__, args)


def create_new_subject(new_subject_fname):
    mmvt_addon = su.init_mmvt_addon()
    mmvt_addon.import_brain()
    su.save_blend_file(new_subject_fname)


if __name__ == '__main__':
    import sys
    new_subject_fname = sys.argv[1]
    if sys.argv[2] == '--background':
        create_new_subject(new_subject_fname)
    else:
        wrap_blender_call()
