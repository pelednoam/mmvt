import mathutils
import bpy


def run(mmvt):
    mu = mmvt.utils
    vert_ind, mesh = mmvt.appearance.get_closest_vertex_and_mesh_to_cursor()
    vert = bpy.data.objects[mesh].data.vertices[vert_ind]
    vert_normal = vert.normal
    mu.rotate_view3d(vert_normal.to_track_quat('Z', 'Y'))