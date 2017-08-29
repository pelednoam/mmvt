import os.path as op
from src.utils import utils

MMVT_DIR = op.join(utils.get_links_dir(), 'mmvt')


def get_matlab_cmd(ini_name='default_args.ini'):
    settings = utils.read_config_ini(MMVT_DIR, ini_name)
    return settings['anatomy'].get('matlab_cmd', 'matlab')

def run_matlab():
    matlab_cmd = get_matlab_cmd()
    print('matlab: {}'.format(matlab_cmd))
    utils.run_script(matlab_cmd)

if __name__ == '__main__':
    run_matlab()
