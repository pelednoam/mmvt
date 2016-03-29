bl_info = {
    'name': 'MMVT loader',
    'author': 'Ohad Felsenstein & Noam Peled',
    'version': (1, 2),
    'blender': (2, 7, 2),
    'location': 'Press [Space], search for "mmvt_addon"',
    'category': 'Development',
}

import bpy
from bpy.types import AddonPreferences
import sys
import os
import importlib as imp


# https://github.com/sybrenstuvel/random-blender-addons/blob/master/remote_debugger.py
class MMVTLoaderAddonPreferences(AddonPreferences):
    # this must match the addon name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    mmvt_folder = StringProperty(
        name='Path of the mmvt folder',
        description='',
        subtype='DIR_PATH',
        default=bpy.path.abspath('//')
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'mmvt_folder')
        layout.label(text='Path of the mmvt folder')


class MMVTLoaderAddon(bpy.types.Operator):
    bl_idname = 'mmvt_addon.run_addon'
    bl_label = 'Run MMVT addon'
    bl_description = 'Runs the mmvt_addon addon'

    def execute(self, context):

        user_preferences = context.user_preferences
        addon_prefs = user_preferences.addons[__name__].preferences
        root = os.path.abspath(addon_prefs.mmvt_folder)
        print('root: {}'.format(root))
        # root = bpy.path.abspath('//')
        mmvt_root = os.path.join(root, 'mmvt_addon')
        print('mmvt_root: {}'.format(mmvt_root))
        sys.path.append(mmvt_root)
        import MMVT_Addon
        # If you change the code and rerun the addon, you need to reload MMVT_Addon
        imp.reload(MMVT_Addon)
        MMVT_Addon.main()

        return {'FINISHED'}


def register():
    bpy.utils.register_class(MMVTLoaderAddon)
    bpy.utils.register_class(MMVTLoaderAddonPreferences)


def unregister():
    bpy.utils.unregister_class(MMVTLoaderAddon)
    bpy.utils.unregister_class(MMVTLoaderAddonPreferences)


if __name__ == '__main__':
    register()