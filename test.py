# launch_napari.py
from napari import Viewer, run

import os
import tifffile

foldpath = '/Users/tampham/switchdrive/Private/Zeiss/data/simulated/'

psf = tifffile.imread(os.path.join(foldpath,'psf_sample_calib_nv_32_coi_2.ome.tif'))
data = tifffile.imread(os.path.join(foldpath,'g_sample_calib_nv_32_coi_2.ome.tif'))

viewer = Viewer()
viewer.window.resize(500,500)

viewer.add_image(psf, name='PSF')
viewer.add_image(data, name='Data')

dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-pyxu-deconv", "Deconvolution"
)
# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()
run()