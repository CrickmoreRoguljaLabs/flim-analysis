from typing import Union, List
from pathlib import Path
from datetime import datetime

import pandas as pd
import napari
import numpy as np

from crickflim.sct.napari.flim_widget import FlimWidget
from crickflim.FLIMageFileReader import FileReader

class NapariWrapper():

    def __init__(self, path : Union[str, Path] = None):
        self.viewer = napari.Viewer()
        self.reader = FileReader()
        if path is None:
            return
        self.open(path)
    
    def open(self, path : Union[str, Path]):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")
        self.path = path
        self.reader.read_imageFile(path)
        self.lifetime_data = np.array(self.reader.image)
        self.viewer.add_image(
            self.intensity,
            name=[f"Channel {i+1}" for i in range(self.lifetime_data.shape[2])],
            colormap="gray",
            blending="additive",
            channel_axis = 2 if self.lifetime_data.ndim == 6 else None,
        )

        self.viewer.add_shapes(name = 'ROIs', face_color = '#FFFFFF00')

        self.flim_widget = FlimWidget(self.reader, self.lifetime_data)
        self.viewer.window.add_dock_widget(
            self.flim_widget.params_container.container,
            name="FLIM fit params",
        )

        self.viewer.window.add_dock_widget(
            self.flim_widget.canvas,
            name="FLIM fit",
        )

        self.flim_widget.analyze_rois_button.clicked.connect(self.analyze_rois)

    @property
    def timepoints(self)->np.ndarray:
        """ Seconds from START time """
        as_datetimes = [
            datetime.strptime(
                datestr,
                "%Y-%m-%dT%H:%M:%S.%f"
            )
            for datestr in self.reader.acqTime
        ]
        return np.array([
            (dt - as_datetimes[0]).total_seconds()
            for dt in as_datetimes
        ])

    @property
    def masks(self)->np.ndarray:
        return self.viewer.layers['ROIs'].to_masks(
            mask_shape=self.lifetime_data.shape[-3:-1]
        )

    def analyze_rois(self, use_fractional_fit : bool = False)->np.ndarray:
        """
        Takes the emgs in the flim_widget attribute and uses them to
        compute values on the image.
        """

        masks = self.masks

        data_by_roi = np.array(
            [np.sum(self.lifetime_data[..., mask,:],axis=-2) for mask in masks]
        ) # roi list of t by z by channel by arrival time

        channelwise_empirical_taus = np.array([
            np.array([
                emg.calculate_empirical_tau(
                    roi[..., channel_idx, :],
                )
                for roi in data_by_roi
            ])*self.flim_widget.bin_size
            for channel_idx, emg in enumerate(self.flim_widget.emgs)
        ]) # n_channels by n_rois by n_t by n_z in nanoseconds

        channelwise_weighted_tau_sum = None
        if use_fractional_fit:
            # Hate to see it.
            channelwise_weighted_tau_sum = np.array(
                [
                emg.fit_fraction_for_output(
                    roi[t, z, channel_idx, :],
                )
                for roi in data_by_roi
                for t in range(roi.shape[0])
                for z in range(roi.shape[1])
                for channel_idx, emg in enumerate(self.flim_widget.emgs)
                ]
            )

        time = self.timepoints
        savestem = self.flim_widget.savepath / self.path.stem

        # Now save the outputs
        # First store the arrays, just in case there's an error
        # this makes it recoverable
        np.savez(savestem.with_name(savestem.name + "_rois.npz"), masks)
        np.save(savestem.with_name(savestem.name + "_time.npy"), time)
        np.save(savestem.with_name(savestem.name + "_channelwise_empirical_taus.npy"), channelwise_empirical_taus)
        if use_fractional_fit:
            np.save(savestem + "_channelwise_weighted_tau_sum.npy", channelwise_weighted_tau_sum)
        
        # Save a .xlsx file with timestamps and the tau values
        # for each ROI

        # First, make a dataframe with the timepoints
        # and the tau values for each ROI
        df = pd.DataFrame({'Experiment time (sec)' : self.timepoints})
        df.set_index('Experiment time (sec)', inplace=True)

        for channel_idx, channel in enumerate(channelwise_empirical_taus):
            for roi_idx, roi in enumerate(channel):
                for plane in range(roi.shape[-1]):
                        df[
                            f"""
                            Empirical tau\n
                            Channel {channel_idx+1}\n
                            ROI {roi_idx+1}\n
                            Plane {plane+1}(ns):"""
                        ] = roi[:, plane]

        df.to_excel(savestem.with_name(savestem.name + "_empirical_taus.xlsx"),
                    engine = 'xlsxwriter',)

        return channelwise_empirical_taus
    
    def load(path : Path):
        raise NotImplementedError("Sorry nerds")

    @property
    def intensity(self):
        return self.lifetime_data.sum(axis=-1)
    



