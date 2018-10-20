import os.path as op
from src.utils import utils

MMVT_DIR = utils.get_link_dir(utils.get_links_dir(), 'mmvt')


def read_onset_xlsx(xlsx_fname):
    subjects = []
    for vals in utils.xlsx_reader(xlsx_fname, skip_rows=1):
        subject, _, elec_name = vals
        subject = subject.replace('\'', '')
        elec_group, elec_name = elec_name.replace('\'', '').split('.')
        elec_name1, elec_name2 = elec_name.split('-')
        elec_name = '{}{}-{}{}'.format(elec_group, elec_name2, elec_group, elec_name1)
        subjects.append("'{}'".format(subject))
        # yield (subject, elec_name)
    print(','.join(subjects))


if __name__ == '__main__':
    xlsx_fname = op.join(MMVT_DIR, 'misc', 'Onset_regions.xlsx')
    if op.isfile(xlsx_fname):
        read_onset_xlsx(xlsx_fname)
    else:
        print('Can\'t find the xlsx file! {}'.format(xlsx_fname))