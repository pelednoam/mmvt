import sys
from src.mmvt_addon.scripts import run_mmvt
from src.mmvt_addon.scripts import scripts_utils as su


def main(argv=None):
    if len(argv) == 1:
        argv = ['-s', 'colin27', '-a', 'laus125']
    parser = su.add_default_args()
    args = su.parse_args(parser, argv)
    run_mmvt.run(args.subject, args.atlas)


if __name__ == '__main__':
    main(sys.argv)