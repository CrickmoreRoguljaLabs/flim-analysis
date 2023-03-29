import inspect
import pathlib

from napari.utils.events import Event, EmitterGroup
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvas
)

from magicgui import magicgui
from magicgui.widgets import Button, Container, ComboBox, FloatSlider, Widget, FileEdit

from crickflim.FLIMageFileReader import FileReader
from crickflim.sct.exponentials.multi_exponential import MultipleExponentialFits
import crickflim.sct.exponentials.loss_functions as loss_functions
from crickflim.sct.utils import pico_to_nano

# Some defaults for matplotlib
matplotlib.use('Qt5Agg')
fdict = {
    'font.family':  'Arial',
    'font.size':    9.0,
    'text.color' : '#FFFFFF',
    'figure.dpi' : 300,
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),
    "xtick.color" : "#FFFFFF",
    "ytick.color" : "#FFFFFF",
    "axes.edgecolor" : "#FFFFFF",
    "axes.labelcolor" : "#FFFFFF",
    "legend.frameon" : False,
    "legend.handlelength" : 0,
    "legend.handletextpad" : 0,
}

# Kinda hacky, should use in figure creation alone
plt.rcParams.update(**fdict)

def get_loss_func_dict():
    return {
        name : obj
        for name, obj in inspect.getmembers(loss_functions)
        if (
            inspect.isclass(obj) and issubclass(obj, loss_functions.LossFunction)
            and (obj is not loss_functions.LossFunction)
        )
    }

class FlimWidget():

    cmap = {
        0: '#2D66A5',
        1: '#C13AA7',
    }

    def __init__(
            self,
            reader : FileReader,
            lifetime_data : np.ndarray,
        ):
        self.fig = plt.figure(figsize=(1, 0.5), dpi=200)
        self.bin_size = pico_to_nano(reader.State.Spc.spcData.resolution[0])
        self.lifetime_data = lifetime_data
        
        self.emgs = [
            MultipleExponentialFits() 
            for _ in range(self.nchannels)
        ]

        self.container = Container(
            layout="vertical",
        )
        self.params_container = ParamsContainer(self)
        self.canvas = FigureCanvas(self.fig)
        self.container.append(self.params_container.container)
        
        # Creating the parameters plot
        self.ax = self.fig.add_axes([0.1, 0.2, 0.8, 0.7])

        self.data_plots = []
        self.fit_lines = []
        self.tau_axis = np.arange(self.lifetime_data.shape[-1]) * self.bin_size
        for c_chan in range(self.nchannels):
            this_chan = self.lifetime_data[:,:,c_chan,...]
            self.data_plots.append(
                self.ax.semilogy(
                    self.tau_axis,
                    this_chan.sum(
                        axis=tuple(range(this_chan.ndim - 1))
                    ),
                    label=f"Channel {c_chan+1}",
                    marker = 'o',
                    markersize=2,
                    linewidth = 0,
                    color = f"{self.__class__.cmap[c_chan]}",
                )
            )

            self.fit(c_chan)

            fit_line, = self.ax.semilogy(
                    self.tau_axis,
                    self.n_photons(channel=c_chan) * self.emgs[c_chan].pdist(self.bin_axis),
                    linewidth = 1, markersize = 0,
                    linestyle = 'dashed',
                    color = '#FFFFFF'
                )
            self.fit_lines.append(fit_line)

        # Never outgrow the data
        curr_min = self.n_photons(channel = 0)
        curr_max = curr_min
        for data_plot in self.data_plots:
            ydata = data_plot[0].get_ydata()
            curr_min = min(curr_min, np.nanmin(ydata[ydata>0]))
            curr_max = max(curr_max, np.nanmax(ydata))

        self.ax.set_ylim(10**(int(np.log10(curr_min))), 10**(int(np.log10(curr_max))))

        # Legend formatting
        l = self.ax.legend()

        for i, text in enumerate(l.get_texts()):
            text.set_color(self.__class__.cmap[i])

        for item in l.legendHandles:
            item.set_visible(False)

        # Axis formatting
        self.ax.set_xlabel("Lifetime (ns)")
        self.ax.set_ylabel("Number of photons")
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        self.params_container.events.connect(
            self.plot_params
        )

    def plot_params(self):
        for i, emg in enumerate(self.emgs):
            self.fit_lines[i].set_ydata(
                self.n_photons(channel=i) * emg.pdist(self.bin_axis)
            )
        self.canvas.draw()

    def n_photons(self, channel : int = 0):
        return self.histogram(channel=channel).sum()

    @property
    def nchannels(self):
        """
        Dangerous.
        """
        return self.lifetime_data.shape[2]
    
    def histogram(self, channel : int = 0):
        this_channel = self.lifetime_data[:,:,channel,...]
        return this_channel.sum(
            axis=tuple(range(this_channel.ndim - 1))
        )
    
    @property
    def bin_axis(self):
        return np.arange(self.lifetime_data.shape[-1])

    @property
    def widget(self):
        return self.container
    
    @property
    def analyze_rois_button(self):
        return self.params_container.analyze_rois_button
    
    @property
    def savepath(self)->pathlib.Path:
        return self.params_container.save_directory_button.value
    
    def fit(self, channel : int):
        hist = self.histogram(channel=channel)
        self.emgs[channel].fit(hist)

class ParamsContainer():

    def __init__(self, flim_widget : FlimWidget):
        
        # Circular! Bad! Lazy!
        self.flim_widget = flim_widget

        self.channel_select = ComboBox(
            name = "channel_select",
            label="Channel",
            choices=[f"Channel {i+1}" for i in range(flim_widget.nchannels)],
            value='Channel 1',
        )

        self.channel_select.changed.connect(self.update_param_widgets)

        loss_func_names = list(get_loss_func_dict().keys())

        self.loss_function = ComboBox(
            name = "loss_function",
            label="Loss function",
            choices=loss_func_names, # names
            value=loss_func_names[0], # default
        )

        self.n_exp = ComboBox(
            name = "num_exps",
            label="Number of exponentials",
            choices=[1, 2],
            value=2,
        )

        self.n_exp.changed.connect(self.update_param_widgets)

        self.params_subcontainer = Container(
            layout="vertical",
            widgets=self.param_widget_list,
        )

        self.buttons_container = Container(
            layout="vertical",
            widgets=[],
        )

        self.fit_button = Button(label = "Fit channel histogram")
        self.fit_button.clicked.connect(self.fit_histogram)
        self.analyze_rois_button = Button(label = "Analyze ROI timeseries")

        self.sub_button_container = Container(
            layout="horizontal",
            widgets=[self.fit_button, self.analyze_rois_button],
        )

        self.save_directory_button = FileEdit('d', label = "Save directory")

        self.buttons_container.append(self.sub_button_container)
        self.buttons_container.append(self.save_directory_button)
    
        self.container = Container(
            layout="vertical",
            widgets=[self.params_subcontainer, self.buttons_container],
        )

        self.container.max_height = 400
        self.container.min_width = 500

        self.update_param_widgets()

        self.events = EmitterGroup(
            self,
        )
        self.events.add(
            changed = Event,
        )

    @property
    def param_widget_list(self):
        if not hasattr(self, "_widget_list"):
            self.generate_param_widget_list()

        return self._widget_list

    def generate_param_widget_list(self)->list[Widget]:
        emg = self.flim_widget.emgs[self.selected_channel_idx]

        taus = [
            FloatSlider(
                name=f"Tau {i+1}",
                value=emg.exps[i].tau*self.flim_widget.bin_size,
                label=f"Tau {i+1} (ns)",
                min = 0.01,
                max = 12.5
            )
            for i in range(self.n_exp.value)
        ]

        fracs = [
            FloatSlider(name=f"Fraction in state {i+1}", value=emg.exps[i].frac, label=f"Fraction {i+1}", min = 0.0, max = 1)
            for i in range(self.n_exp.value - 1)
        ]

        for frac in fracs:
            frac.changed.connect(lambda x: self.update_fracs(frac))

        irf_mean = FloatSlider(name="Irf_mu", value = emg.irf.mean*self.flim_widget.bin_size, label="Offset (ns)", min = 0.01,  max = 12.5)
        irf_std = FloatSlider(name="Irf_sigma", value = emg.irf.sigma*self.flim_widget.bin_size, label="IRF width (ns)", min = 0.01, max = 12.5)
        widget_list = [self.channel_select, self.loss_function, self.n_exp] + taus + fracs + [irf_mean, irf_std]
        for widget in widget_list:
            widget.changed.connect(self.to_par_callback(widget))
        
        self._widget_list = widget_list
        return widget_list

    def update_fracs(self, adjusted_frac : Widget):
        for widget in self.param_widget_list:
            if 'Fraction' in widget.name:
                widget.max = 1 - sum(
                    [
                    w.value for w in self.param_widget_list
                    if 'Fraction' in w.name and w is not widget
                    ]
                )

    @property
    def selected_channel_idx(self)->int:
        return int(self.channel_select.value[-1]) - 1

    def to_par_callback(self, widget):
        return lambda x: self.update_params(widget, x)

    def fit_histogram(self, *arg, **kwargs):
        self.flim_widget.fit(channel=self.selected_channel_idx)
        self.update_param_widgets()
        self.events.changed()

    def update_params(self, source : Widget, event):
        # Update the params in the model
        emg = self.flim_widget.emgs[self.selected_channel_idx]
        emg.loss_func = get_loss_func_dict()[self.loss_function.value]()
        if 'Tau' in source.name:
            tau_idx = int(source.name[-1]) - 1
            emg.exps[tau_idx].tau = source.value/self.flim_widget.bin_size
        if 'Fraction' in source.name:
            frac_idx = int(source.name[-1]) - 1
            emg.exps[frac_idx].frac = source.value
        if source.name == 'Irf_mu':
            emg.irf.mean = source.value/self.flim_widget.bin_size
        if source.name == 'Irf_sigma':
            emg.irf.sigma = source.value/self.flim_widget.bin_size

        self.events.changed()

    def update_param_widgets(self):
        # Update the params container and then
        # refresh the actual widget display
        self.params_subcontainer.clear()
        for widget in self.generate_param_widget_list():
            self.params_subcontainer.append(widget)
        #self.params_subcontainer.refresh()

    @magicgui(fn={'mode': 'd'})
    def save_directory(self, fn = pathlib.Path.home()):
        pass
