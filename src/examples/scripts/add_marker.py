import bpy


def run(mmvt):
    mmvt.set_current_time(500)
    bpy.ops.marker.add()
    bpy.ops.marker.rename(name='Stimulus onset')