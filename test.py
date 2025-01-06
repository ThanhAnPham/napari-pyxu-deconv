# launch_napari.py
from napari import Viewer, run

import os
import tifffile

data_oi = 0
if data_oi==0:
    foldpath = '/data/tampham/simulated/'#'/Users/tampham/switchdrive/Private/Zeiss/data/simulated/'
    psf = tifffile.imread(os.path.join(foldpath,'psf_sample_calib_nv_32_coi_2_n_0.001_2.5_bg_0.0001.ome.tif'))
    data = tifffile.imread(os.path.join(foldpath,'g_sample_calib_nv_32_coi_2_n_0.001_2.5_bg_0.0001.ome.tif'))
    pxsz = [1,100,35.7,35.7]
elif data_oi==1:
    foldpath = '/data/tampham/real_donut/'#'/Users/tampham/switchdrive/Private/Zeiss/data/simulated/'
    psf = tifffile.imread(os.path.join(foldpath,'psf.tif'))
    data = tifffile.imread(os.path.join(foldpath,'data.tif'))
    pxsz = [1,0.0794,0.0794]

viewer = Viewer()
viewer.window.resize(500,500)

viewer.add_image(psf, name='PSF', scale=pxsz)
viewer.add_image(data, name='Data', scale=pxsz)

dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "napari-pyxu-deconv", "Deconvolution"
)
# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()
run()