import bpy


def run(mmvt):
    elc_name = 'zzz'
    x, y, z = mmvt.where_am_i.get_tkreg_ras() * 0.1
    mmvt.data.create_electrode(x, y, z, elc_name)
    

