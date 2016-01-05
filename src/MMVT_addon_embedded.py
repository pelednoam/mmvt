bl_info = {
    "name": "Multi-modal visualization tool",
    "author": "Ohad Felsenstein & Noam Peled",
    "version": (1, 2),
    "blender": (2, 7, 2),
    "api": 33333,
    "location": "View3D > Add > Mesh > Say3D",
    "description": "Multi-modal visualization tool",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Add Mesh"}

import bpy
import sys
import os

# Try to import external
root = bpy.path.abspath('//')
print(root)
sys.path.append(os.path.join(root, 'mmvt_code'))
import MMVT_Addon
MMVT_Addon.main()

# import pydevd
# http://www.blender.org/api/blender_python_api_2_66_release/bpy.props.html
# pydevd.settrace()
