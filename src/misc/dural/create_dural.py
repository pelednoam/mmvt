import os
import sys
import shutil
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


def create_dural_surface(subject, subjects_dir):
    '''
    Creates the dural surface in the specified subjects_dir. This is done
    using a standalone script derived from the Freesurfer tools which actually
    use the dural surface.

    The caller is responsible for providing a correct subjects_dir, i.e., one
    which is writable. The higher-order logic should detect an unwritable
    directory, and provide a user-sanctioned space to write the new fake
    subjects_dir to.

    Parameters
    ----------
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. If this folder is not writable,
        the program will crash.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable.
    '''
    print('create dural surface step')
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    # scripts_dir = os.path.dirname(__file__)
    # os.environ['SCRIPTS_DIR'] = scripts_dir
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    print('script_dir: %s' % scripts_dir)

    if (os.path.exists(os.path.join(subjects_dir,subject,'surf','lh.dural'))
            and os.path.exists(os.path.join(subjects_dir, subject,'surf',
            'rh.dural'))):
        print('dural surfaces already exist')
        return

    import subprocess

    # curdir = os.getcwd()
    # os.chdir(os.path.join(subjects_dir, subject, 'surf'))

    for hemi in ('lh','rh'):
        make_dural_surface_cmd = [os.path.join(scripts_dir,
            'make_dural_surface.csh'),'-i',os.path.join(subjects_dir, subject, 'surf','%s.pial'%hemi),
            '-p', sys.executable]
        print(make_dural_surface_cmd)
        p=subprocess.call(make_dural_surface_cmd)
        shutil.move('%s.dural'%hemi, os.path.join(subjects_dir, subject, 'surf', '%s.dural'%hemi))
    # os.chdir(curdir)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    create_dural_surface(args.subject, SUBJECTS_DIR)
