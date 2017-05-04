import bpy
import os

# get the path where the blend file is located
basedir = bpy.path.abspath('//')
fol = os.path.join(basedir, 'objects')
if not os.path.isdir(fol):
    os.makedirs(fol)


# deselect all objects
bpy.ops.object.select_all(action='DESELECT')

# loop through all the objects in the scene
scene = bpy.context.scene
for ob in [bpy.data.objects['rh'], bpy.data.objects['lh']]: # scene.objects:
    ob.hide_select = False
    # make the current object active and select it
    scene.objects.active = ob
    ob.select = True

    # make sure that we only export meshes
    if ob.type == 'MESH':
        # export the currently selected object to its own file based on its name
        bpy.ops.export_scene.obj(
                filepath=os.path.join(fol, ob.name + '.obj'),
                use_selection=True,
                )
    # deselect the object and move on to another if any more are left
    ob.select = False