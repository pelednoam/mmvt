import bpy
import bmesh


def run(mmvt):
    for obj in bpy.data.objects:
        obj.select = False
    bm = bmesh.new()
    for obj in bpy.data.objects['Subcortical_structures'].children:
        mesh = obj.data
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.clear()
        mesh.update()
    bm.free()
