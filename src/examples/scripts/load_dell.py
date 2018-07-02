import os.path as op
import sys
import importlib


def run(mmvt):
    mmvt_code_fol = mmvt.utils.get_mmvt_code_root()
    addon_code_fol = op.join(mmvt_code_fol, 'src', 'mmvt_addon')
    if addon_code_fol not in sys.path:
	    sys.path.append(addon_code_fol)
    import dell_panel
    importlib.reload(dell_panel)
    dell_panel.init(mmvt)
    mmvt.dell = dell_panel

