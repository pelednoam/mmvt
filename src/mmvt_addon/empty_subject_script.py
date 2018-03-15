import bpy
import sys
import importlib as imp

# https://blenderartists.org/forum/archive/index.php/t-365581.html
module = 'mmvt_loader'
bpy.ops.wm.addon_enable(module=module)
user_preferences = bpy.context.user_preferences
addon_prefs = user_preferences.addons[module].preferences
mmvt_root = bpy.path.abspath(addon_prefs.mmvt_folder)
print('mmvt_root: {}'.format(mmvt_root))
sys.path.append(mmvt_root)
import mmvt_addon
# In case you want to debug, unmark the following two lines
# from scripts import scripts_utils as su
# su.debug()
imp.reload(mmvt_addon)
mmvt_addon.main(addon_prefs)
