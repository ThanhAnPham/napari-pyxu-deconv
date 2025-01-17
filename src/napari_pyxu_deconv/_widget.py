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
import pyxudeconv as pd
import numpy as np
import pathlib
if TYPE_CHECKING:
    import napari
#debug
from napari.utils.notifications import show_info
from argparse import Namespace

import gc
import torch
import cupy as cp

NGPU = cp.cuda.runtime.getDeviceCount()

#NGPU = torch.cuda.device_count()


# if we want even more control over our widget, we can use
# magicgui `Container`
class Deconvolution(Container):
    """Deconvolution class for Napari

    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._old_method = ''
        self.saved_values_dynamic = {
            "RL": {},
            "GARL": {},
            "Tikhonov": {},
            "GLS": {},
            "GKL": {},
            "RLTV": {},
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
        self._run_layer = widgets.PushButton(
            name='run',
            label='Start deconvolution',
        )
        self._run_layer.clicked.connect(self._on_run)

        self.static_container = Container()
        self.dynamic_container = Container()
        self._maxC = 0
        self._set_widgets()
        self.max_width = 500  #not nice to hard-code

    def _set_widgets(self):
        #Common widgets
        if NGPU > 0:
            default_method = "GARL"
        else:
            default_method = "RL"
        if default_method == "RL":
            default_nepoch = 30
        else:
            default_nepoch = 100
            
        self._image_layer_meas = create_widget(
            name='datapath',
            label="Measurements",
            annotation="napari.layers.Image",
            value=self.values_from_param_file.get('datapath', None),
        )
        self._image_layer_meas.changed.connect(self._on_meas_change)

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

        #Airyscan layer
        default_order = "CZYX"
        if self._image_layer_meas.value is None:
            airyscan = True
        else:
            if np.ndim(self._image_layer_meas.value) >= 4:
                airyscan = True
                default_order = "NZCYX"
            else:
                airyscan = False

        self._airyscan_layer = widgets.CheckBox(
            name='airyscan',
            value=airyscan,
            text='Is multi-viewed data (e.g., Airyscan)',
            tooltip='Ignored if data is 3D',
        )

        self._airyscan_layer.changed.connect(self._on_airyscan_change)

        self._dim_order_layer = widgets.ComboBox(
            name='dimorder',
            label="Dimensions order",
            choices=["NZCYX", "NCZYX", "ZYX", "CZYX", "NZYX"],
            value=self.values_from_param_file.get('dimorder', default_order),
            visible=True,
            tooltip="Can be automatically determined in some cases.")

        self._dim_order_layer.changed.connect(self._on_metadata_change)

        self.update_max_channels()

        self._coi_layer = widgets.SpinBox(
            name='coi',
            label="Reconstructed channel",
            min=0,
            value=self.values_from_param_file.get('coi', 0),
            max=self._maxC,
            step=1,
            tooltip="Leave at 0 if there is no channel in the data",
        )

        self._bg_layer = widgets.FloatSpinBox(
            name='bg',
            label=
            "Background (minimum value).\nIf smaller than 0, automatically chosen.",
            value=self.values_from_param_file.get('bg', -1),
            min=-1,
            step=1,
        )

        self._nepoch_layer = widgets.SpinBox(
            name='Nepoch',
            label="Number of iterations",
            value=self.values_from_param_file.get('Nepoch', default_nepoch),
            min=1,
            step=1,
        )
        self._disp_layer = widgets.SpinBox(
            name='disp',
            label="Display frequency",
            value=self.values_from_param_file.get('disp', 0),
            min=0,
            step=1,
        )

        self._method_layer = widgets.ComboBox(
            name='methods',
            label="Deconvolution method",
            choices=["RL", "GARL", "Tikhonov", "GLS", "GKL", "RLTV"],
            value=self.values_from_param_file.get('methods', default_method),
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
            value=self.values_from_param_file.get('bufferwidthx', 15),
            min=0,
            step=1,
        )

        self._bufferwidthy_layer = widgets.SpinBox(
            name='bufferwidthy',
            label="Y",  #"Buffer width along y",
            value=self.values_from_param_file.get('bufferwidthy', 15),
            min=0,
            step=1,
        )

        self._bufferwidthz_layer = widgets.SpinBox(
            name='bufferwidthz',
            label="Z",
            value=self.values_from_param_file.get('bufferwidthz', 15),
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
            tooltip=
            'The volume is reconstructed with a larger size to allow contribution from outside the measured area.',
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
            'Lateral ROI (x,y,w,h) with (x,y) the top-left coordinates of the ROI and with (w,h) the width and height, respectively.\nCenter of the stack can be selected by setting (x,y) to (-1,-1).\nNote that the Z direction is always fully considered.\nWhole field of view can be selected by setting (x,y,w,h) to (-1,-1,-1,-1).',
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
            name='psf_sz',
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

        # append into/extend the container with your widgets
        self.static_container.extend([
            self._param_layer,
            self._image_layer_meas,
            self._image_layer_psf,
            self._airyscan_layer,
            self._dim_order_layer,
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
        ])
        self.clear()
        self.extend(self.static_container)
        self.update_dynamic_layout(self._method_layer.value)

    def _on_airyscan_change(self):
        if self._airyscan_layer.value:
            if self._image_layer_meas.value is not None:
                ndim_meas = np.ndim(self._image_layer_meas.value)
                if ndim_meas == 4:
                    self._dim_order_layer.value = "NZYX"
                #elif ndim_meas == 5:
                #self._dim_order_layer.value = "NZCYX"
                elif ndim_meas < 4:
                    show_info("Not enough dimensions in measurements")
                    self._airyscan_layer.value = False
        else:
            if self._image_layer_meas.value is not None:
                ndim_meas = np.ndim(self._image_layer_meas.value)
                if ndim_meas == 4:
                    self._dim_order_layer.value = "CZYX"
                elif ndim_meas == 3:
                    self._dim_order_layer.value = "ZYX"

        self._on_metadata_change()

    def _on_meas_change(self):
        if self._image_layer_meas.value is not None:
            ndim_meas = np.ndim(self._image_layer_meas.value)
            if self._airyscan_layer.value:
                if ndim_meas == 4:
                    self._dim_order_layer.value = "NZYX"
            else:
                if ndim_meas == 3:
                    self._dim_order_layer.value = "ZYX"
                elif ndim_meas == 4:
                    self._dim_order_layer.value = "CZYX"
        self._on_metadata_change()

    def _on_metadata_change(self):
        self.update_max_channels()
        self._coi_layer.max = self._maxC

    def update_max_channels(self):
        if self._image_layer_meas.value is None:
            self._maxC = 0
        else:
            if np.ndim(self._image_layer_meas.value.data) == 3 or (
                    np.ndim(self._image_layer_meas.value.data) == 4
                    and self._airyscan_layer.value):
                self._maxC = 0
            else:
                self._maxC = self._image_layer_meas.value.data.shape[
                    self._dim_order_layer.value.find("C")] - 1

    def _on_advanced_change(self):
        """
        Callback function to handle advanced options change and update the parameters accordingly.
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
        param = vars(param)
        for cwidget in self.static_container:
            if isinstance(cwidget, Container) and len(cwidget) > 1:
                param[cwidget.name] = tuple([cw.value for cw in cwidget])
                if 'bufferwidth' == cwidget.name:
                    param[cwidget.name] = param[cwidget.name][::-1]
            elif 'path' in cwidget.name:
                if cwidget.value is None:
                    show_info('Please specify the PSF and the measurements')
                    return 0
                else:
                    param[cwidget.name] = img_as_float(cwidget.value.data)
            else:
                param[cwidget.name] = cwidget.value
        if param['datapath'].ndim == 3:
            param['nviews'] = 1
        param['create_fname'] = lambda x, y, z: self.create_fname(
            x, y, self._image_layer_meas.value.name, z)
        param['save_results'] = self.save_results
        param['fres'] = ''
        param['saveMeas'] = False
        param['methods'] = [param['methods']]
        param['saveIter'] = (param['disp'] if param['disp'] > 0 else 1e8, )
        param['pxsz'] = self._image_layer_meas.value.scale[-3:]
        param['unit'] = str(self._image_layer_meas.value.units[0])
        param['normalize_meas'] = True
        if param['bg'] == 0:
            param['bg'] = 1e-9
        if np.ndim(param['datapath']) == 3:
            self._dim_order_layer.value = "ZYX"
        elif np.ndim(param['datapath']) == 4 and self._airyscan_layer.value:
            self._dim_order_layer.value = "NZYX"

        param['datapath'] = self.select_roi(param['datapath'], param['roi'],
                                            param['coi'],
                                            self._dim_order_layer.value)
        param['psfpath'] = self.select_roi(param['psfpath'], param['psf_sz'],
                                           param['coi'],
                                           self._dim_order_layer.value)

        #add dynamic layers
        config_meth = 'config_' + param['methods'][0]
        param[config_meth] = dict()
        for cwidget in self.dynamic_container:
            cval = cwidget.value
            if not isinstance(cwidget,
                              widgets.Label) and cwidget.name != 'run':
                if isinstance(cval, (float, int, str)):
                    param[config_meth][cwidget.name] = (cval, )
                elif isinstance(cval, pathlib.PurePath):
                    if str(cval).lower() == 'default model':
                        param[config_meth][cwidget.name] = ('', )
                    elif not cval.exists():
                        show_info(f'Folder {cval} does not exist')
                        return 0
                    else:
                        param[config_meth][cwidget.name] = (str(cval), )
                else:
                    param[config_meth][cwidget.name] = cval

        param = Namespace(**param)

        show_info(f'Starting Deconvolution with {self._method_layer.value}...')
        ims = pd.deconvolve(param)
        del ims
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        torch.cuda.empty_cache()
        show_info(f'Deconvolution with {self._method_layer.value} done!')

    def select_roi(self, data, roi, coi, dim_order):
        """Select region of interest

        Args:
            data (numpy.ndarray): region of interest is selected from data (3,4,5D array)
            roi (4-tuple of int): region of interest (x0,y0,w,h) with (x0,y0) top-left coordinate and (w,h) the width and height of the ROI, respectively.
                                  If x0,y0==-1, set in such a way that the ROI is centered. If w,h=-1, set to maximize the field of view.
            coi (int or tuple of int): channel of interest (-1 if no channel)
        """

        #reorder the dimensions to "CNZYX" (singleton dimensions are created)
        dim_to_expand = None
        if dim_order == "NZCYX":
            dim_perm = (2, 0, 1, 3, 4)
        elif dim_order == "NCZYX":
            dim_perm = (1, 0, 2, 3, 4)
        elif dim_order == "ZYX":
            dim_perm = (0, 1, 2)
            dim_to_expand = (0, 1)
        elif dim_order == "CZYX":
            dim_perm = (0, 1, 2, 3)
            dim_to_expand = (1)
        elif dim_order == "NZYX":
            dim_perm = (0, 1, 2, 3)
            dim_to_expand = (0)

        data = np.permute_dims(data, dim_perm)
        if dim_to_expand != None:
            data = np.expand_dims(data, dim_to_expand)

        coi = np.array(coi)
        hoi = np.array(np.arange(0, np.shape(data)[1]))
        roi = np.array(roi)

        if np.any(roi[2:] == None) or np.any([cr <= 0 for cr in roi[2:]]):
            roi = np.array((0, 0, *data.shape[-2:]))
        elif np.any(roi[:2] == None) or np.any([cr < 0 for cr in roi[:2]]):
            # top-left coordinates taken in such way that ROI is centered
            roi[0] = np.maximum(data.shape[-2] // 2 - roi[2] // 2, 0)
            roi[1] = np.maximum(data.shape[-1] // 2 - roi[3] // 2, 0)
        #make sure that ROI doesn't go out of bounds
        roi[-2:] = np.minimum(roi[:2] + roi[-2:] - 1,
                              np.array(data.shape[-2:]) - 1) - roi[:2] + 1
        roi = tuple(map(int, roi))
        out = data[
            coi.reshape((-1, 1, 1, 1, 1)),
            hoi.reshape((1, -1, 1, 1, 1)),
            :,
            roi[0]:roi[0] + roi[2],
            roi[1]:roi[1] + roi[3],
        ].squeeze()

        return out

    def save_results(self, vol, fname, pxsz, unit):
        """Add results to the Napari Viewer

        Args:
            vol (numpy.ndarray or cupy.ndarray): volume
            fname (str): File name
            pxsz (tuple of float): pixel size (tuple of 3)
            unit (str): unit of pixel size
        """
        self._viewer.add_image(
            np.asarray(vol),
            name=fname,
            scale=pxsz,  #(1, pxsz[1] / pxsz[0], pxsz[2] / pxsz[0]),
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
        text_widget = widgets.Label(value=f"Parameter(s) for {method}")

        # Populate the container with different widgets based on the choice
        if method == "RL":
            accel_widget = widgets.CheckBox(
                name='acceleration',
                value=True,
                text='Accelerated algorithm',
                tooltip=
                "Results may differ a bit from the non-accelerated version. The expected acceleration is roughly 2-3 times faster.",
            )
            self.dynamic_container.extend([accel_widget])
        if method == "RLTV":
            reg_widget = widgets.FloatSpinBox(
                name="tau",
                value=self.saved_values_dynamic[method].get("tau", 5e-1),
                min=0,
                label='Regularization parameter',
                tooltip="Higher value means stronger regularization",
                step=0.0001,
            )
            accel_widget = widgets.CheckBox(
                name='acceleration',
                value=True,
                text='Accelerated algorithm',
                tooltip=
                "Results may differ a bit from the non-accelerated version. The expected acceleration is roughly 2-3 times faster.",
            )
            self.dynamic_container.extend([reg_widget, accel_widget])
        if method == "GARL" or method == "GLS" or method == "GKL":
            #TODO: Add a possibility to do ranged values?
            #TODO: save the parameters in JSON file somewhere so that pyxudeconv can load it.
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
                value=self.values_from_param_file.get('model',
                                                      'default model'),
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
                value=self.saved_values_dynamic[method].get("lmbd", 2.5e-1),
                min=0,
                label='Regularization parameter',
                tooltip="Higher value means stronger regularization",
                step=0.0001,
            )

            sigma_widget = widgets.FloatSpinBox(
                name='sigma',
                value=self.saved_values_dynamic[method].get("sigma", 5.),
                min=0,
                label="Sigma",
                tooltip=
                "Related to noise level. Higher value means stronger regularization.",
                step=0.01,
            )
            accel_widget = widgets.CheckBox(
                name='acceleration',
                value=True,
                text='Accelerated algorithm',
                tooltip=
                "Results may differ a bit from the non-accelerated version. The expected acceleration is roughly 2-3 times faster.",
            )
            self.dynamic_container.extend([
                text_widget,
                model_widget,
                epochoi_widget,
                reg_widget,
                sigma_widget,
                accel_widget,
            ])
        elif method == "Tikhonov":
            reg_widget = widgets.FloatSpinBox(
                name='tau',
                value=self.saved_values_dynamic[method].get("tau", 1.),
                min=0,
                label="Regularization parameter",
                step=0.0001,
            )
            self.dynamic_container.extend([text_widget, reg_widget])

        self.dynamic_container.extend([self._run_layer])
        self.extend(self.dynamic_container)
        self._old_method = method
