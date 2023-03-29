from PyQt5.QtWidgets import QSpinBox, QComboBox, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QMainWindow, QPushButton
import os
from PyQt5.QtGui import QPixmap
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import csv
from crickflim.roi import ROI
from crickflim.exponentials import ExponentiallyModifiedGaussian
from crickflim.flim_files import FlimFile, StackedFlimFile


"""-----------------------------------------------------------------------------------------------------------"""

matplotlib.use('Qt5Agg')

# Global Parameters
IMAGE_RESCALING = 1
POINT_SIZE = 1
LINE_WIDTH = 1
ROI_COLOR = "red"


# ----------------------------------------- Helper Class -----------------------------------------

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def plot(self, x, y, label):
        pass

# ----------------------------------------- FlimAnalysisGUI -----------------------------------------


class FlimAnalysisGUI(QMainWindow):

    def __init__(self, path, save_path, q_application):

        """ path = the location of the flim file for analysis
            save_path = location where the metrics and analysis are saved as a csv
            q_application = the overarching QApplication : used to get screen dimensions from the user """

        super().__init__()

        # the cvs file that we wish to write too
        self.csv = self.create_csv(save_path)

        # Construct a ROI object to manage the ROI (this will interface with the buttons in this class)
        self.ROI = ROI()

        # The exponential model that we will try fitting to the photon data
        self.exp_model = ExponentiallyModifiedGaussian(bin_width=0, exponential=2)

        # Construct a FlimFile object to use to manage the flim file
        self.flim_data = self.create_flim_storage(path, self.csv)

        # The size of the time bins (each time bin is a point in the photon histogram)
        self.bin_width = self.flim_data.file_reader.State.Spc.spcData.resolution[0]

        # Update the bin width of the exponential (the order is weird here since i need to construct both)
        self.exp_model.bin_width = self.bin_width

        # Create a canvas for plotting matplotlib graphs
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # The scale of the y axis for the photon counts
        self.y_scale = "linear"

        # Variable representing how much we want to scale the image
        self.image_rescaling = IMAGE_RESCALING

        # We then initialize the plot itself
        self.update_frame_histogram_graph()

        # The size information about the screen
        self.screen_size = q_application.primaryScreen().size()

        # TODO how to make this into a fixed size (or at least bigger?) self.setFixedSize(QSize(400, 300))
        self.setWindowTitle("Flim Image Analysis: V 1.0")

        # Constructs a vertical layout that contains the interactive buttons and will be embedded
        # into the overall window on the right side
        self.button_layout = QVBoxLayout()
        self.initialize_button_layout()  # Fill out the window buttons

        # Constructs a vertical layout that contains both the images of the individual time frames from the flim file
        # (top), the histogram of photon arrival times + curve fitting for the frame show (middle) and the flourescent
        # lifetime / empirical tau for the entire file (bottom)
        self.frame_graph_layout = QVBoxLayout()
        self.initialize_frame_graph_layout()  # Fill out the image frame and two graphs

        # Final layout that contains the frame/graph layout (left) and the button layout (right)
        self.complete_layout = QHBoxLayout()

        # Add both layouts
        self.complete_layout.addWidget(self.plot_canvas)

        self.complete_layout.addLayout(self.button_layout)
        self.complete_layout.addLayout(self.frame_graph_layout)

        self.graphing = QHBoxLayout()

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.graphing.addWidget(self.plot_canvas)

        self.complete_layout.addLayout(self.graphing)

        container = QWidget()
        container.setLayout(self.complete_layout)
        container.resize(self.screen_size.width(), self.screen_size.height())

        container.show()

        # Set the central widget of the Window.
        self.setCentralWidget(container)

    def create_csv(self, path):
        """ :returns an opened csv file """

        file = open(path, "w")
        return csv.writer(file)

    def create_flim_storage(self, path, csv):
        """ :returns a flim storage (either a flimfile or a stackedflimfile depending on the path type)
            - if the user inputs a folder we read all of the internal flim files, and if the user inputs a individual
                file we process it individually """

        # first we check to see if the path is a valid file or directory
        if os.path.isfile(path) and os.path.splitext(path)[1] == ".flim":

            return FlimFile(path, csv, self.exp_model, self.ROI)

        elif os.path.isdir(path):

            # Here we have a directory so we want to take only the files that end in the flim extension

            # Find only the files that are flim files
            valid_files = list(filter(lambda x: os.path.splitext(x)[1] == ".flim", os.listdir(path)))

            # Sort the files by date modified since this should be the correct order
            valid_files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))

            # We then create a stacked flim file for the valid files
            return StackedFlimFile([os.path.join(path, extension) for extension in valid_files], csv, self.exp_model, self.ROI)

        # In the case that the above two cases were not true we want to quit the program and launch an error message
        raise Exception("The file path given was either not a valid flim file, or a folder with no flim files inside.")
        quit()

    def initialize_button_layout(self):
        """ Called by FlimAnalysisGUI init to construct the button layout """

        self.initialize_frame_indexing_buttons()

        self.initialize_roi_buttons()

        self.initialize_graph_interface()

        self.initialize_averaging_buttons()

        self.initialize_fitting_summary_button()

    def initialize_graph_interface(self):
        """ Creates the buttons that are used to interact with the graph
            there are multiple ways to affect the graph
            1.) by changing the roi directly you change the actual counts for the photon histogram
            2.) by using the buttons (defined below) """

        # We create a label and then each button and link each button to the correct update functions
        self.graph_button_label = QLabel("Graph Interface:")

        # Create a combo box where the user can select the scales for the y axis
        self.y_axis_scale = QComboBox()
        self.y_axis_scale.addItem("linear-linear scale")
        self.y_axis_scale.addItem("linear-log scale")

        self.y_axis_scale.activated.connect(self.change_log_scale)

        # Create a combo box that represents the number of exponentials that will be used in the model
        self.exponential_count_choices = QComboBox()
        self.exponential_count_choices.addItem("Double Exponential")
        self.exponential_count_choices.addItem("Monoeponential")

        self.exponential_count_choices.activated.connect(self.change_exponential)

        self.button_layout.addWidget(self.graph_button_label)
        self.button_layout.addWidget(self.y_axis_scale)
        self.button_layout.addWidget(self.exponential_count_choices)

    def change_log_scale(self, index):
        """ In this case the user is trying to change the scale of the y axis (linear <--> log) """

        # Switch depending on an index that represents the two scales
        if index == 0:
            # update the log scale parameter
            self.y_scale = "linear"
        else:
            self.y_scale = "log"

        # after we have updated the model we want to update the visuals
        self.update_frame_histogram_graph()

    def change_exponential(self, index):
        """ In this case the user is trying to switch the number of terms in the exponential model, so we
            replace the current model with a new instantiation of a exponentially modified gaussian with the correct
            number of terms """

        if index == 0:
            # create a new model with 1 exponential
            self.exp_model = ExponentiallyModifiedGaussian(self.bin_width, exponential=2)
        else:
            # create a new model with 2 exponentials
            self.exp_model = ExponentiallyModifiedGaussian(self.bin_width, exponential=1)

        # after we have updated the model we want to update the visuals
        self.update_frame_histogram_graph()

    def initialize_averaging_buttons(self):
        """ Creates an interface to allow the user to average over their frames to improve their tau estimates """

        # We first create a label to define this section
        averaging_label = QLabel("Averaging Tools: ")

        # small numerical input box with arrows
        averaging_count = QSpinBox()

        # Only allow the frame index to be between [1, max # frames], and print max # frames
        averaging_count.setMinimum(1)
        averaging_count.setMaximum(self.flim_data.frame_count - 1)
        averaging_count.setPrefix("# Frames to Average = ")

        # Update the way the flim data performs the summary printing
        averaging_count.valueChanged.connect(self.update_averaging_count)

        averaging_type = QComboBox()
        averaging_type.addItem("Sliding Window")
        averaging_type.addItem("Disjoint Window")

        averaging_type.activated.connect(self.change_averaging_type)

        self.button_layout.addWidget(averaging_label)
        self.button_layout.addWidget(averaging_count)
        self.button_layout.addWidget(averaging_type)

    def change_averaging_type(self, index):
        """ Updates the averaging type field in the flim_data object to reflect the type of averaging the user wants to
            perform """

        # Switch depending on an index that represents the two types
        if index == 0:
            self.flim_data.averaging_type = "sliding"
        else:
            self.flim_data.averaging_type = "disjoint"

    def update_averaging_count(self, new_size):
        """ Updates the averaging count field within the flim_data file to reflect the new average window size """
        self.flim_data.average_size = new_size

    def initialize_fitting_summary_button(self):
        """ Creates a single button that when pressed prints a matrix of values to be entered directly into prism """

        # We first create a label to define this section
        self.printing_label = QLabel("Data Output:")

        self.print_fitting = QPushButton("Print Taus + Error")
        self.print_fitting.clicked.connect(self.flim_data.print_tau_summary)

        self.button_layout.addWidget(self.printing_label)
        self.button_layout.addWidget(self.print_fitting)

    def initialize_roi_buttons(self):
        """ Called when constructing the GUI (specifically when making the button layout on the right): this
            contains the initialization and construction of the interface for making a roi """

        # Create a button that can be toggled to
        self.construct_roi = QPushButton("Create ROI")
        self.construct_roi.setCheckable(True)
        self.construct_roi.clicked.connect(self.roi_construction)

        # Create a button than when pressed clears the ROI
        self.destroy_roi = QPushButton("Destroy ROI")
        self.destroy_roi.clicked.connect(self.remove_roi_points)

        roi_control = QLabel("ROI Tools:")

        self.button_layout.addWidget(roi_control)
        self.button_layout.addWidget(self.construct_roi)
        self.button_layout.addWidget(self.destroy_roi)

    def remove_roi_points(self):
        """ Updates the roi information to get rid of the points """

        self.ROI.remove_roi_points()

        self.update_displayed_frame(self.flim_data.current_frame)

    def roi_construction(self, checked):
        """ Updates the instantiation of the ROI class to reflect the fact that we are either entering or leaving the
            construction of the ROI phase """

        self.ROI.update_constructing(checked)

        self.update_displayed_frame(self.flim_data.current_frame)

    def initialize_frame_graph_layout(self):
        """ Called by FlimAnalysisGUI init to construct the frame image and graph layouts """

        self.initialize_frame_viewing_label()

    def initialize_frame_viewing_label(self):
        """ Constructs a QLabel that will contain an image of the current frame of the flim file
                this image will be embedded into the window in the top left and image/frame index is changed
                through the frame index buttons """

        # Construct a new label : not sure why images are in labels but this seems to be the convention?
        self.current_frame = QLabel()
        # Start off with the first
        self.current_frame.setPixmap(QPixmap.fromImage(self.flim_data.ith_frame_image(self.flim_data.current_frame)))

        # Allow this image to be rescaled
        #self.current_frame.setScaledContents(True)
        self.current_frame.setFixedWidth(self.flim_data.image_width * self.image_rescaling)
        self.current_frame.setFixedHeight(self.flim_data.image_width * self.image_rescaling)

        # Override the mouse press event function to read the pixel location (we will then check if the roi construction
        # button is toggled as on)
        self.current_frame.mousePressEvent = self.get_individual_roi_point

        self.frame_graph_layout.addWidget(self.current_frame)

    def get_individual_roi_point(self, event):
        """This will be called when the user clicks on the image itself, in this case we want to provide this information
            to the ROI class (which will add this point to the internal roi storage if in the construction phase or
            disregard this data) """

        self.ROI.integrate_new_roi_point(event.pos().x(), event.pos().y())

        self.update_displayed_frame(self.flim_data.current_frame)

    def initialize_frame_indexing_buttons(self):
        """ ---------------  Frame Indexing, rescaling  and changing channels Buttons --------------- """

        # The label for the frame indexing button
        frame_index_label = QLabel("Frame Tools:")

        # small numerical input box with arrows
        frame_index = QSpinBox()

        # Only allow the frame index to be between [1, max # frames], and print max # frames
        frame_index.setMinimum(0)
        frame_index.setMaximum(self.flim_data.frame_count - 1)
        frame_index.setPrefix("Frame index = ")
        frame_index.setSuffix("/" + str(self.flim_data.frame_count - 1))

        # Since we want to change the image being displayed on the button presses, we create
        # a connection from this button
        frame_index.valueChanged.connect(self.update_displayed_frame)

        # integer box with arrows for changing the image scale
        frame_scale = QSpinBox()

        # We only want the rescaling to be between [1, 10?] -> the uper bound is kinda arbitrary
        frame_scale.setMinimum(1)
        frame_scale.setMaximum(10)
        frame_scale.setPrefix("Frame Scaling = ")

        # We want to change the size of the image being displayed => change the parameter everywhere and then
        # regraph the image
        frame_scale.valueChanged.connect(self.update_image_scaling)

        # Create a two option
        channel_button = QComboBox()
        channel_button.addItem("Channel 1 (green)")
        channel_button.addItem("Channel 2 (red)")

        channel_button.activated.connect(self.change_channel_button)

        self.button_layout.addWidget(frame_index_label)
        self.button_layout.addWidget(frame_index)
        self.button_layout.addWidget(frame_scale)
        self.button_layout.addWidget(channel_button)

    def change_channel_button(self, i):
        """ Called when the user wants to change the channel of the flim data we are looking at, so we update the frame
            and regraph """

        # Update the channel id
        self.flim_data.channel = i

        # redraw the frame
        self.update_displayed_frame(self.flim_data.current_frame)

    def update_image_scaling(self, i):
        """ called when the user is trying to change the size of the image that we are viewing, changes the variable in
            both the flim and self and then regraphs the image """

        # Update the representation of image scale
        self.image_rescaling = i
        self.flim_data.image_rescaling = i
        self.ROI.image_rescaling = i

        # Regraph the current frame
        self.update_displayed_frame(self.flim_data.current_frame)

    def update_frame_histogram_graph(self):
        """ Called when we are showing a new frame, in this case we want the histogram to reflect the new graph """

        self.plot_canvas.axes.cla()

        # Get the histogram data
        y = list(self.flim_data.ith_frame_photon_histogram(self.flim_data.current_frame))
        x = [i for i in range(len(y))]

        # Convert the bin/time data to nanoseconds
        x_adj = [x_i / (self.bin_width / 1000.0) for x_i in x]

        # Change the graph depending on the user defined y axis scale
        if self.y_scale == "linear":

            self.plot_canvas.axes.plot(x_adj, y, label="Photon histogram")
            self.exp_model.fit(x, y)
            self.plot_canvas.axes.plot(x_adj, self.exp_model.predict(x), label="Model Fit")

        else:

            self.plot_canvas.axes.plot(x_adj, np.log(y), label="Photon histogram")
            self.exp_model.fit(x, y)
            self.plot_canvas.axes.plot(x_adj, np.log(self.exp_model.predict(x)), label="Model Fit")


        # Set the labels for the graph
        self.plot_canvas.axes.set_xlabel(str("Time (nanoseconds)"))
        self.plot_canvas.axes.set_ylabel(str("Normalized photon count"))

        self.plot_canvas.axes.legend()

        self.plot_canvas.draw()

    def update_displayed_frame(self, i):
        """ Updates self.current_frame (the picture of the flim file in the top left) to whatever frame the user
            requests """

        # We first get a QPixmap of the frame of interest
        frame = QPixmap.fromImage(self.flim_data.ith_frame_image(i))

        # Resize the frame if needed
        self.current_frame.setFixedWidth(self.flim_data.image_width * self.image_rescaling)
        self.current_frame.setFixedHeight(self.flim_data.image_width * self.image_rescaling)

        # Update the drawing of the frame and update the counter for the current frame index
        self.current_frame.setPixmap(frame)
        self.flim_data.current_frame = i

        # We also want to update the graph to be showing the photon histogram for this
        self.update_frame_histogram_graph()


