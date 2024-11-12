"""
This module contains four napari widgets declared in
different ways:


- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

#from magicgui import magic_factory
from magicgui import widgets
from magicgui.widgets import Container, create_widget
#from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import os
import json
import torch
import pyxudeconv as pd
import numpy as np
#if TYPE_CHECKING:
import napari
#debug
from napari.utils.notifications import show_info

NGPU = torch.cuda.device_count()


# if we want even more control over our widget, we can use
# magicgui `Container`
class Deconvolution(Container):
    """Deconvolution class for Napari

    Args:
        Container (_type_): 
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._old_method = ''
        self.saved_values_dynamic = {
            "RL": {},
            "GARL": {},
            "Tikhonov": {},
        }
        self.values_from_param_file = {}
        self._viewer = viewer  #will add the deconvolved version

        opts_file_edit = {
            "mode":
            "r",
            "filter":
            "*.json",
            "tooltip":
            'Load values from the parameter file. Override existing values.\nIf you would like to re-apply the parameter file, you need to change the text and choose again the same parameter file.',
        }

        self._param_layer = create_widget(
            name='param',
            label="Parameter file",
            annotation="str",
            value=None,
            widget_type="FileEdit",
            options=opts_file_edit,
        )
        self._param_layer.changed.connect(self._on_param_file_change)

        self.static_container = Container()
        self.dynamic_container = Container()
        self._set_widgets()

    def _set_widgets(self):
        #Common widgets

        self._image_layer_meas = create_widget(
            name='datapath',
            label="Measurements",
            annotation="napari.layers.Image",
            value=self.values_from_param_file.get('datapath', None),
        )
        self._image_layer_psf = create_widget(
            name='psfpath',
            label="Point-spread function",
            annotation="napari.layers.Image",
            value=self.values_from_param_file.get('psfpath', None),
        )

        listGPU = list(range(-1, NGPU))
        gpu_oi = self.values_from_param_file.get('gpu', 0 if NGPU > 0 else -1)
        if gpu_oi not in listGPU:
            gpu_oi = listGPU[-1]
            show_info('Selected GPU in parameter file is not available')
        self._gpu_layer = widgets.ComboBox(
            name='gpu',
            label="GPU",
            choices=listGPU,
            value=gpu_oi,
        )

        if self._image_layer_meas.value is None:
            maxC = 1
        else:
            maxC = self._image_layer_meas.value.shape[0]

        self._coi_layer = widgets.SpinBox(
            name='coi',
            label="Reconstructed channel",
            min=0,
            value=self.values_from_param_file.get('coi', 0),
            step=1,
            #tooltip="",
        )

        self._bg_layer = widgets.FloatSpinBox(
            name='bg',
            label=
            "Background (minimum value).\nIf smaller than 0, automatically chosen.",
            value=self.values_from_param_file.get('bg', 0),
            min=-1,
            step=1,
        )
        self._nepoch_layer = widgets.SpinBox(
            name='Nepoch',
            label="Number of iterations",
            value=self.values_from_param_file.get('Nepoch', 25),
            min=1,
            step=1,
        )
        self._disp_layer = widgets.SpinBox(
            name='disp',
            label="Display frequency",
            value=self.values_from_param_file.get('disp', 10),
            min=0,
            step=1,
        )

        self._method_layer = widgets.ComboBox(
            name='method',
            label="Deconvolution method",
            choices=["RL", "GARL", "Tikhonov"],
            value=self.values_from_param_file.get('method', "RL"),
        )
        self._method_layer.changed.connect(self._on_method_change)

        #Advanced layer

        self._advanced_layer = widgets.CheckBox(
            name='advanced',
            value=False,
            text='Advanced options',
        )
        self._advanced_layer.changed.connect(self._on_advanced_change)

        #Buffer width
        self._bufferwidthx_layer = widgets.SpinBox(
            name='bufferwidthx',
            label="X",  #"Buffer width along x",
            value=self.values_from_param_file.get('bufferwidthx', 3),
            min=0,
            step=1,
        )

        self._bufferwidthy_layer = widgets.SpinBox(
            name='bufferwidthy',
            label="Y",  #"Buffer width along y",
            value=self.values_from_param_file.get('bufferwidthy', 3),
            min=0,
            step=1,
        )

        self._bufferwidthz_layer = widgets.SpinBox(
            name='bufferwidthz',
            label="Z",
            value=self.values_from_param_file.get('bufferwidthz', 1),
            min=0,
            step=1,
        )

        self._bufferwidth_layer = Container(
            name='bufferwidth',
            layout='horizontal',
            widgets=[
                self._bufferwidthx_layer,
                self._bufferwidthy_layer,
                self._bufferwidthz_layer,
            ],
            label='Buffer width',
            visible=False,
        )
        # ROI
        self._roix_layer = widgets.SpinBox(
            name='roix',
            label="X",
            value=self.values_from_param_file.get('roix', 0),
            min=-1,
            step=1,
        )
        self._roiy_layer = widgets.SpinBox(
            name='roiy',
            label="Y",
            value=self.values_from_param_file.get('roiy', 0),
            min=-1,
            step=1,
        )
        self._roiw_layer = widgets.SpinBox(
            name='roiw',
            label="W",
            value=self.values_from_param_file.get('roiw', -1),
            min=-1,
            step=1,
        )
        self._roih_layer = widgets.SpinBox(
            name='roih',
            label="H",
            value=self.values_from_param_file.get('roih', -1),
            min=-1,
            step=1,
        )
        self._roi_layer = Container(
            name='roi',
            layout='horizontal',
            widgets=[
                self._roix_layer,
                self._roiy_layer,
                self._roiw_layer,
                self._roih_layer,
            ],
            label='Region of interest',
            tooltip=
            'Lateral ROI (x,y,w,h) with (x,y) the top-left coordinates of the ROI and with (w,h) the width and height, respectively.\nNote that the Z direction is always fully considered.\nWhole field of view can be selected by setting (x,y,w,h) to (-1,-1,-1,-1).',
            visible=False,
            labels=False,
        )

        # ROI PSF
        self._psfroix_layer = widgets.SpinBox(
            name='psfroix',
            label="X",
            value=self.values_from_param_file.get('psfroix', -1),
            min=-1,
            step=1,
        )
        self._psfroiy_layer = widgets.SpinBox(
            name='psfroiy',
            label="Y",
            value=self.values_from_param_file.get('psfroiy', -1),
            min=-1,
            step=1,
        )
        self._psfroiw_layer = widgets.SpinBox(
            name='psfroiw',
            label="W",
            value=self.values_from_param_file.get('psfroiw', 64),
            min=-1,
            step=1,
        )
        self._psfroih_layer = widgets.SpinBox(
            name='psfroih',
            label="H",
            value=self.values_from_param_file.get('psfroih', 64),
            min=-1,
            step=1,
        )
        self._psfroi_layer = Container(
            name='psfroi',
            layout='horizontal',
            widgets=[
                self._psfroix_layer,
                self._psfroiy_layer,
                self._psfroiw_layer,
                self._psfroih_layer,
            ],
            label='PSF Region of interest',
            tooltip=
            'Lateral PSF ROI (x,y,w,h) with (x,y) the top-left coordinates of the ROI and with (w,h) the width and height, respectively.\nNote that the Z direction is always fully considered.\nCenter of the stack can be selected by setting (x,y) to (-1,-1), and maximal width and height similarly.',
            visible=False,
            labels=False,
        )

        self._run_layer = widgets.PushButton(
            name='run',
            label='Start deconvolution',
        )
        self._run_layer.clicked.connect(self._on_run)
        # append into/extend the container with your widgets
        self.static_container.extend([
            self._param_layer,
            self._image_layer_meas,
            self._image_layer_psf,
            self._coi_layer,
            self._gpu_layer,
            self._bg_layer,
            self._nepoch_layer,
            self._disp_layer,
            self._advanced_layer,
            self._bufferwidth_layer,
            self._roi_layer,
            self._psfroi_layer,
            self._method_layer,
            self._run_layer,
        ])
        self.clear()
        self.extend(self.static_container)
        self.update_dynamic_layout(self._method_layer.value)

    def _on_advanced_change(self):
        """
        Callback function to handle advanved options change and update the parameters accordingly.
        """
        if self._advanced_layer.value:
            self._bufferwidth_layer.visible = True
            self._roi_layer.visible = True
            self._psfroi_layer.visible = True
        else:
            self._bufferwidth_layer.visible = False
            self._roi_layer.visible = False
            self._psfroi_layer.visible = False

    def _on_run(self):
        """
        Callback function to run deconvolution
        """
        param = pd.get_param()
        cparam = self.as_dict
        for key, val in cparam.enumerate():
            param[key] = val
        param['create_fname'] = lambda x, y, z: self.create_fname(
            x, y, self._image_layer_meas.name, z)
        param['save_results'] = self.save_results
        pd.deconvolve(param)
        show_info(f'Deconvolution with {self._method_layer.value} done!')

    def save_results(self, vol, fname, pxsz, unit):
        """Add results to the Napari Viewer

        Args:
            vol (numpy.ndarray or cupy.ndarray): volume
            fname (str): File name
            pxsz (tuple of float): pixel size (tuple of 3)
            unit (str): unit of pixel size
        """
        self._viewer.add_image(
            vol,
            name=fname,
            scale=pxsz,
            units=unit,
        )

    def create_fname(
        self,
        meth,
        paramstr,
        fid,
        metric=-np.inf,
    ):
        """Create a filename

        Args:
            meth (str): Method name
            paramstr (str): Method Parameters
            fid (str): File ID (e.g., image layer name)
            metric (float, optional): Metric value if phantom exists. Defaults to -np.inf.

        Returns:
            _type_: _description_
        """
        if np.isinf(metric) and metric < 0:
            return f'{meth}_{fid}_{paramstr}'
        else:
            return f'{meth}_{fid}_{paramstr}_{metric:.4e}'

    def _on_param_file_change(self):
        """
        Callback function to handle parameter file change and update the parameters accordingly.
        """
        param_file = str(self._param_layer.value)
        if os.path.exists(param_file):
            if 'json' in param_file[param_file.rfind('.'):]:
                with open(param_file, 'r', encoding="utf-8") as f:
                    self.values_from_param_file = json.load(f)

                if 'bufferwidth' in self.values_from_param_file.keys():
                    self.values_from_param_file[
                        'bufferwidthx'] = self.values_from_param_file[
                            'bufferwidth'][0]
                    self.values_from_param_file[
                        'bufferwidthy'] = self.values_from_param_file[
                            'bufferwidth'][1]
                    self.values_from_param_file[
                        'bufferwidthz'] = self.values_from_param_file[
                            'bufferwidth'][2]
                if 'psf_roi' in self.values_from_param_file.keys():
                    self.values_from_param_file[
                        'psfroix'] = self.values_from_param_file['psf_roi'][0]
                    self.values_from_param_file[
                        'psfroiy'] = self.values_from_param_file['psf_roi'][1]
                    self.values_from_param_file[
                        'psfroiw'] = self.values_from_param_file['psf_roi'][2]
                    self.values_from_param_file[
                        'psfroih'] = self.values_from_param_file['psf_roi'][3]
                if 'roi' in self.values_from_param_file.keys():
                    self.values_from_param_file[
                        'roix'] = self.values_from_param_file['roi'][0]
                    self.values_from_param_file[
                        'roiy'] = self.values_from_param_file['roi'][1]
                    self.values_from_param_file[
                        'roiw'] = self.values_from_param_file['roi'][2]
                    self.values_from_param_file[
                        'roih'] = self.values_from_param_file['roi'][3]
                self.static_container.clear()
                self._set_widgets()
            else:
                show_info('Invalid parameter file. Expecting a JSON file', )
        else:
            show_info(f'Selected parameter file does not exist: {param_file}')

    def _on_method_change(self):
        """
        Callback function to handle choice changes and update the layout accordingly.
        """
        self.update_dynamic_layout(self._method_layer.value)

    def update_dynamic_layout(self, method: str):
        """
        Updates the dynamic container with widgets based on the selected method
        """
        for widget in self.dynamic_container:
            self.saved_values_dynamic[self._old_method][
                widget.name] = widget.value
            if widget in self:
                self.remove(widget)  #remove from the layout

        # Clear the dynamic container's current widgets
        self.dynamic_container.clear()
        text_widget = widgets.Label(  #value='',
            value=f"Parameter(s) for {method}")

        # Populate the container with different widgets based on the choice
        if method == "GARL":
            #TODO: Add a possibility to do ranged values?
            #TODO: Will save the parameters in JSON file somewhere so that pyxudeconv can load it.
            #TODO: Add a possibility to load a JSON file?

            opts_file_edit = {
                "mode":
                "d",
                "tooltip":
                'Filepath to the folder containing the trained model.\nSee documentation.',
            }
            model_widget = create_widget(
                name='model',
                label="Trained model filepath",
                annotation="str",
                value=self.values_from_param_file.get(
                    'model',
                    '/Users/tampham/switchdrive/Private/Zeiss/trained_models/3Dtubes/'
                ),
                widget_type="FileEdit",
                options=opts_file_edit,
            )

            epochoi_widget = widgets.SpinBox(
                name="epochoi",
                value=self.saved_values_dynamic[method].get("epochoi", 40180),
                min=0,
                step=1000,
                label='Epoch of interest for the trained model',
            )
            reg_widget = widgets.FloatSpinBox(
                name="lmbd",
                value=self.saved_values_dynamic[method].get("lmbd", 1.),
                min=0,
                label='Regularization parameter',
            )

            sigma_widget = widgets.FloatSpinBox(
                name='sigma',
                value=self.saved_values_dynamic[method].get("sigma", 1.),
                min=0,
                label="Sigma (noise level)",
            )
            self.dynamic_container.extend([
                text_widget,
                model_widget,
                epochoi_widget,
                reg_widget,
                sigma_widget,
            ])
        elif method == "Tikhonov":
            reg_widget = widgets.FloatSpinBox(
                name='lmbd',
                value=self.saved_values_dynamic[method].get("lmbd", 1.),
                min=0,
                label="Regularization parameter",
            )
            self.dynamic_container.extend([text_widget, reg_widget])
        self.extend(self.dynamic_container)
        self._old_method = method


'''
    def _threshold_im(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        image = img_as_float(image_layer.data)
        name = image_layer.name + "_thresholded"
        threshold = self._threshold_slider.value
        if self._invert_checkbox.value:
            thresholded = image < threshold
        else:
            thresholded = image > threshold
        if name in self._viewer.layers:
            self._viewer.layers[name].data = thresholded
        else:
            self._viewer.add_labels(thresholded, name=name)

'''
