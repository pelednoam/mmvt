from src.utils import utils


def which_matlab():
    ret = utils.run_script('which matlab')
    ret = ret.replace('\n', '')
    return ret

if __name__ == '__main__':
    matlab_cmd = which_matlab()
    print('matlab full cmd: "{}"'.format(matlab_cmd))