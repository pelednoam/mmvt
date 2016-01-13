bl_info = {
    'name': 'MMVT loader',
    'author': 'Ohad Felsenstein & Noam Peled',
    'version': (1, 2),
    'blender': (2, 7, 2),
    'location': 'Press [Space], search for "mmvt_addon"',
    'category': 'Development',
}

import bpy
import sys
import os
import importlib as imp


class run_mmvt_addon(bpy.types.Operator):
    bl_idname = 'mmvt_addon.run_addon'
    bl_label = 'Run MMVT addon'
    bl_description = 'Runs the mmvt_addon addon'

    def execute(self, context):
        root = bpy.path.abspath('//')
        print(root)
        sys.path.append(os.path.join(root, 'mmvt_addon'))
        import MMVT_Addon
        # If you change the code and rerun the addon, you need to reload MMVT_Addon
        imp.reload(MMVT_Addon)
        MMVT_Addon.main()

        return {'FINISHED'}

def register():
    bpy.utils.register_class(run_mmvt_addon)


def unregister():
    bpy.utils.unregister_class(run_mmvt_addon)


if __name__ == '__main__':
    register()