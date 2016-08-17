import os.path as op
import csv
import mne
import numpy as np


def add_points_to_freeview(vertices_pos, points_fname):
    with open(points_fname, 'w') as fp:
        writer = csv.writer(fp, delimiter=' ')
        writer.writerows(vertices_pos * 1000)
        writer.writerow(['info'])
        writer.writerow(['numpoints', len(points_fname)])
        writer.writerow(['useRealRAS', '0'])


def save_cursor_position(point, point_fol):
    np.savetxt(op.join(point_fol, 'tmp', 'edit.dat'), point * 1000)


def grow_labels(subject, seeds, extents, surface, subjects_dir):
    names = [str(seed) for seed in seeds]
    labels = mne.grow_labels(subject, seeds, extents, 0, subjects_dir, names=names, surface=surface)
    seeds_pos = []
    for label, seed in zip(labels, seeds):
        seed_pos = label.pos[np.where(label.vertices==seed)[0][0]]
        seeds_pos.append(seed_pos)
        label.pos /= 1000.0
        label.save(op.join(subjects_dir, subject, 'label', '{}.label'.format(label.name)))
    return np.array(seeds_pos)


def save_freeview_cmd(volume_file, points_file, cmd_fol):
    cmd = 'freeview -f {} -c {}:Radius=2:Color=red'.format(volume_file, points_file)
    with open(op.join(cmd_fol, 'run_freeview.sh'), 'w') as sh_file:
        sh_file.write(cmd)
    print(cmd)

def int_arr_type(var): return var

def parse_parser(parser, argv=None):
    if argv is None:
        in_args = vars(parser.parse_args())
    else:
        in_args = vars(parser.parse_args(argv))
    args = {}
    for val in parser._option_string_actions.values():
        if val.type is str:
            args[val.dest] = args[val.dest].replace('_', ' ')
        elif val.type is int_arr_type:
            args[val.dest] = get_args_list(in_args, val.dest, int, val.default)
        elif val.dest in in_args:
            if type(in_args[val.dest]) is str:
                args[val.dest] = in_args[val.dest].replace("'", '')
            else:
                args[val.dest] = in_args[val.dest]
    return args


def get_args_list(args, key, var_type, default_val):
    if args[key] is None or len(args[key]) == 0:
        return default_val
    args[key] = args[key].replace("'", '')
    if ',' in args[key]:
        ret = args[key].split(',')
    elif len(args[key]) == 0:
        ret = []
    else:
        ret = [args[key]]
    if var_type:
        ret = list(map(var_type, ret))
    return ret


class Bag( dict ):
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Finding the vertices')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-v', '--vertices', help='vertices', required=True, type=int_arr_type)
    parser.add_argument('-r', '--surface', help='surface', required=False, default='seghead')
    parser.add_argument('--subjects_dir', help='subjects_dir', required=False, default='')
    args = Bag(parse_parser(parser))
    subjects_dir = args.subjects_dir if args.subjects_dir != '' else os.environ.get('SUBJECTS_DIR', '')
    if subjects_dir == '':
        print('Please set first SUBJECTS_DIR')
    else:
        seeds_pos = grow_labels(args.subject, args.vertices, 2, args.surface, subjects_dir)
        save_cursor_position(seeds_pos[0], op.join(subjects_dir, args.subject))
        points_fname = op.join(subjects_dir, args.subject, 'vertices.dat')
        add_points_to_freeview(seeds_pos, points_fname)
    # save_freeview_cmd(op.join(subjects_dir, subject, 'mri', '{}.mgz'.format(surface)), points_fname,
    #                           op.join(subjects_dir, subject))
    # subject = 'nmr00978'
    # subjects_dir = '/cluster/neuromind/dwakeman/tsc_pilot/subjects/'
    # extents = 2 # in mm
    # seeds = [111504]
    # surface = 'seghead'
    print('finish!')
