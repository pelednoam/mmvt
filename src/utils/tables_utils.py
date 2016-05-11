'''
Created on Nov 3, 2014

@author: noampeled
'''

import numpy as np
import traceback

try:
    import tables
    DEF_TABLES = True
except:
    print('no pytables!')
    DEF_TABLES = False

# http://stackoverflow.com/questions/9002433/how-should-python-dictionaries-be-stored-in-pytables
tables_dict = {
    'key': tables.StringCol(itemsize=40),
    'value': tables.Int32Col(),
}


def create_hdf5_file(file_name):
    try:
        return tables.open_file(file_name, mode='w')
    except:
        return tables.openFile(file_name, mode='w')


def open_hdf5_file(file_name, mode='a'):
    try:
        return tables.open_file(file_name, mode=mode)
    except:
        return tables.openFile(file_name, mode=mode)


# dtype = np.dtype('int16') / np.dtype('float64')
def create_hdf5_arr_table(hdf_file, group, array_name,
        dtype=np.dtype('float64'), shape=(), arr=None,
        complib='blosc', complevel=5):
    atom = tables.Atom.from_dtype(dtype)
    if arr is not None:
        shape = arr.shape
#     filters = tables.Filters(complib=complib, complevel=complevel)
    if not is_table_in_group(group, array_name):
        try:
            ds = hdf_file.create_carray(group, array_name, atom, shape)
        except:
            ds = hdf_file.createCArray(group, array_name, atom, shape)
    else:
        ds = group._v_children[array_name]

    if arr is not None:
        ds[:] = arr
    return ds


def create_hdf5_table(hdf5_file, group, table_name, table_desc):
    if not table_name in group._v_children:
        tab = hdf5_file.createTable(group, table_name, table_desc)
    else:
        tab = group._v_children[table_name]
#     tab.cols.key.createIndex()
    return tab


def is_table_in_group(group, array_name):
    try:
        return array_name in group._v_children
    except:
        return False


def add_dic_items_into_table(tab, d):
    tab.append(d.items())


def read_dic_from_table(tab, keyVal):
    vals = [row['value'] for row in tab.where('key == {}'.format(keyVal))]
    if len(vals) > 0:
        return vals[0]
    else:
        return None


def find_or_create_group(h5file, name=''):
    if name == '':
        group = h5file.root
    else:
        try:
            group = h5file.getNode('/{}'.format(name))
        except:
            group = h5file.createGroup('/', name)
    return group


def find_table(h5file, table_name, group_name=''):
    try:
        path = '/{}'.format(table_name) if group_name == '' else \
               '/{}/{}'.format(group_name, table_name)
        table = h5file.get_node(path)
    except:
        table = None
    return table


def find_tables(h5file, table_names, group_name=''):
    tables = []
    for table_name in table_names:
        tables.append(find_table(h5file, table_name, group_name))
    return tables


def find_table_in_group(group, table_name):
    if is_table_in_group(group, table_name):
        return group._v_children[table_name]
    else:
        return None


def read_tables_into_dict(h5file, group=None):
    if group is None:
        group = h5file.root
    tables_dict = {}
    for tab_name, tab in group._v_children.items():
        if tab._c_classid == 'GROUP':
            continue
        elif tab._c_classid == 'UNIMPLEMENTED':
            print('The table "{}" type is unimplemented!'.format(tab_name))
        else:
            try:
                tables_dict[tab_name] = tab[:]
            except:
                print('Error reading the table {}!'.format(tab_name))
                print(traceback.format_exc())
    return tables_dict