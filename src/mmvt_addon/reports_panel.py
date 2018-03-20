import bpy
import os.path as op
import glob
import time
import traceback
import re
import mmvt_utils as mu


def _addon():
    return ReportsPanel.addon


def update_something():
    fol = op.join(mu.get_parent_fol(mu.get_user_fol()), 'reports')
    report_fname = op.join(fol, '{}.html'.format(bpy.context.scene.reports_files.replace(' ', '_')))
    with open(report_fname, 'r') as f:
        report_text = f.read()
        fields = set([report_text[m.start() +  1: m.end() - 1] for m in re.finditer('~[0-9A-Za-z ]+~', report_text)])
    reports_items = [(c.replace(' ', '_'), c, '', ind) for ind, c in enumerate(fields)]
    bpy.types.Scene.reports_fields = bpy.props.EnumProperty(
        items=reports_items, description="reports fields",)
    bpy.context.scene.reports_files = reports_items[0]


def do_somthing():
    pass



def reports_files_update(self, context):
    # if ReportsPanel.init:
    update_something()


def reports_draw(self, context):
    layout = self.layout
    layout.prop(context.scene, "reports_files", text="")
    row = layout.row(align=0)
    row.prop(context.scene, "reports_fields", text="")
    row.prop(context.scene, "reports_field_value", text="")
    layout.operator(ReportsButton.bl_idname, text="Create report", icon='STYLUS_PRESSURE')


class ReportsButton(bpy.types.Operator):
    bl_idname = "mmvt.reports_button"
    bl_label = "Reports botton"
    bl_options = {"UNDO"}

    def invoke(self, context, event=None):
        do_somthing()
        return {'PASS_THROUGH'}


bpy.types.Scene.reports_files = bpy.props.EnumProperty(items=[], description="reports files")
bpy.types.Scene.reports_fields = bpy.props.EnumProperty(items=[], description="reports fields")
bpy.types.Scene.reports_field_value = bpy.props.StringProperty()


class ReportsPanel(bpy.types.Panel):
    bl_space_type = "GRAPH_EDITOR"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "mmvt"
    bl_label = "Reports"
    addon = None
    init = False

    def draw(self, context):
        if ReportsPanel.init:
            reports_draw(self, context)


def init(addon):
    ReportsPanel.addon = addon
    user_fol = mu.get_user_fol()
    reports_files = glob.glob(op.join(mu.get_parent_fol(user_fol), 'reports', '*.html'))
    if len(reports_files) == 0:
        return None
    files_names = [mu.namebase(fname).replace('_', ' ') for fname in reports_files]
    reports_items = [(c, c, '', ind) for ind, c in enumerate(files_names)]
    bpy.types.Scene.reports_files = bpy.props.EnumProperty(
        items=reports_items, description="reports files",update=reports_files_update)
    bpy.context.scene.reports_files = files_names[0]
    register()
    ReportsPanel.init = True


def register():
    try:
        unregister()
        bpy.utils.register_class(ReportsPanel)
        bpy.utils.register_class(ReportsButton)
    except:
        print("Can't register Reports Panel!")


def unregister():
    try:
        bpy.utils.unregister_class(ReportsPanel)
        bpy.utils.unregister_class(ReportsButton)
    except:
        pass
