from crickflim.FLIMageFileReader import FileReader
import numpy as np
from PIL import Image as im
from PIL import ImageDraw
from PIL.ImageQt import ImageQt as im_qt


# Global Parameters
IMAGE_RESCALING = 1
POINT_SIZE = 1
LINE_WIDTH = 1
ROI_COLOR = "red"


# ----------------------------------------- Helper Functions -----------------------------------------


def convert_time(ind_time):
    """ :returns the time in seconds (float) expected form is [2202-06-27T18:04:35.411] """

    # First split the time based on the date <-> time
    _, time = ind_time.split("T")

    # Split the time into hours, minutes and seconds
    hours, minutes, seconds = time.split(":")

    # Convert hours and minutes into seconds
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)


# ----------------------------------------- FlimFile -----------------------------------------


class FlimFile:

    def __init__(self, path, csv, exp_model, roi):

        # The path to the flim file on the computer
        self.file_path = path

        # we first create a file reader object
        self.file_reader = FileReader()

        # Ask this reader to read the first image
        self.file_reader.read_imageFile(self.file_path)

        # Cast the images from the flim reader into an array
        self.images = np.asarray(self.file_reader.image)

        # time metrics for the experiment
        self.times = self.get_data_times()

        # Number of total frames in the experiment
        self.frame_count = self.images.shape[0]

        # Resolution of the flim experiment
        self.image_width = self.images.shape[3]

        # A csv file where we can print too
        self.csv = csv

        # The current frame that is being visualized
        self.current_frame = 0

        # The type of averaging that we will be performing
        self.averaging_type = "sliding"

        # the number of frames that we would like to average
        self.average_size = 1

        # the exponential model
        self.exp_model = exp_model

        # We send in the roi object from the overarching class to work as a mask for the data
        self.roi = roi

        # The scale for the image (changed this such that the user can actually change it themselves in live time)
        self.image_rescaling = IMAGE_RESCALING

        # The channel the experiment was run in (switch for red and green)
        self.channel = 0

    def get_data_times(self):
        """ :returns a list of times (where each time is converted to seconds and subtracts the first frame) where the
            ith entry is the time of the ith frame """

        # Get the time data from the frames
        times = [convert_time(time) for time in self.file_reader.acqTime]

        # Subtract everything by the first time to get a start of experiment time
        adj_times = [time - times[0] for time in times]

        return adj_times

    def get_mask(self):
        """ :returns a numpy array that serves as a mask for the roi (defaults to all 1's if there is no roi) """

        # Check to see if there is a roi to draw
        if self.roi.roi_exists:
            # create a blank image the size of the flim data being visualized
            img = im.new('L', (self.image_width, self.image_width), 0)
            # The points that define the polygon roi
            points = [x for xs in self.roi.roi_points for x in xs]
            # Use imagedraw tools to draw the polygone directly on the image (slightly messy)
            ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
            # Convert this back to a mask/array
            mask = np.array(img)
        else:
            # Blank mask
            mask = np.ones((self.image_width, self.image_width))

        return mask

    def ith_frame_image(self, i):
        """ :returns the ith frame of this flim file as an image:
                To get a 2D representation, photon counts are summed for each x,y location across time bins and then the
                entire image is normalized -> [0, 255]
                in the case that roi is not empty we can also add the roi to the screen """

        # sum across the time bins for the ith image in the sequence
        ind_image = np.sum(self.images[i][0][self.channel], axis=2)

        # Normalize the image
        ind_image = (ind_image - np.amin(ind_image)) / (np.amax(ind_image) - np.amin(ind_image)) * 255

        # Multiply this image by the mask to get rid of the points we are not viewing
        ind_image = np.uint8(ind_image * self.get_mask())

        # Stack this image 3 times to convert it into a rgb image
        ind_image = np.dstack([ind_image, ind_image, ind_image])  # Comment out this line if the roi colors are inverted

        # Save this as a PIL image and return it
        # TODO I cannot use both the pil and tiff libraries, maybe i need to pick a single convention and go with it
        pil_image = im.fromarray(ind_image)

        # Resize the image
        pil_image = pil_image.resize((self.image_width * self.image_rescaling, self.image_width * self.image_rescaling))

        # Add the ROI data
        pil_image = self.draw_roi(pil_image)

        # convert the image to the qt form at the end
        return im_qt(pil_image).copy()

    def ith_frame_photon_histogram(self, i, normalized=True):
        """ :returns a numpy array of length (64 * based on current time bin conventions) where the ith entry
            is the number of photons that arrived at during the ith time bin """

        # Calculate the mask based on the roi to not add photon counts for pixels outside of the roi
        mask = np.asarray([self.get_mask() for _ in range(64)]).transpose((1, 2, 0))

        # Sum across the x and y dimensions of the ith frame to get the photon counts per time bin
        photon_histogram = np.sum(self.images[i][0][self.channel] * mask, axis=(0, 1))

        if normalized:
            # Get the total number of photons
            total_photons = photon_histogram.sum()
            return photon_histogram / total_photons
        else:
            return photon_histogram

    def draw_roi(self, current_image):
        """ :returns a pil image that is constructed by drawing the roi information in white on top of the
            current image """

        # Create the draw object first
        draw_object = ImageDraw.Draw(current_image)

        # First we check if there is a roi to be drawn
        if self.roi.roi_exists or self.roi.constructing_roi:

            # Draw each of the points
            for points in self.roi.roi_points:

                # We first rescale the points (since the roi data is collected in small scale pixel values)
                points = [ind_point * self.image_rescaling for ind_point in points]

                # We need to scale these points down (the x,y points come from the image which is already expanded)
                draw_object.ellipse((points[0], points[1],
                                     points[0] + POINT_SIZE, points[1] + POINT_SIZE),
                                    fill=ROI_COLOR, outline=ROI_COLOR)

            # Draw lines connecting each of the points (we split on the case that we want to join first and last)
            if self.roi.roi_exists:

                points_list = self.roi.roi_points + [self.roi.roi_points[0]]

            else:

                points_list = self.roi.roi_points

            # convert the points list
            adj_points = []

            # Convert the list of ind points into a temporary list of pairs of adjacent points for line drawing
            for point in points_list:
                adj_points.append([point[0] * self.image_rescaling, point[1] * self.image_rescaling])

            # Draw lines between consecutive points
            for i in range(len(adj_points) - 1):

                # Use the draw library to draw a line
                draw_object.line((adj_points[i][0], adj_points[i][1],
                                  adj_points[i + 1][0], adj_points[i + 1][1]),
                                 fill=ROI_COLOR, width=LINE_WIDTH)

        return current_image

    def averaging_frame_count(self):
        """ :returns the number of individual histograms that will need to be fit based on the type of averaging and
            the size of the windows """

        # Split based on the type of averaging
        if self.averaging_type == "sliding":

            return self.frame_count - self.average_size + 1

        elif self.averaging_type == "disjoint":

            return self.frame_count // self.average_size

        else:

            raise Exception("You are attempting to average using an unknown method, try a different averaging option")
            quit()

    def average_photon_dist(self, i):
        """ :returns the average of photon distributions (individual frames are joined across the space and time
            dimensions) based on the averaging protocol """

        # Split based on the type of averaging
        if self.averaging_type == "sliding":

            # Storage for the final set of distributions to be returned
            ind_dists = []

            # The time of the first frame that is in the sliding average
            ind_time = self.times[i]

            # The individual histograms
            for i in range(i, i + self.average_size):

                ind_dists.append(self.ith_frame_photon_histogram(i, normalized=False))

        elif self.averaging_type == "disjoint":

            # Storage for the final set of distributions to be returned
            ind_dists = []

            # The time of the first frame that is in the disjoint average
            ind_time = self.times[i * self.average_size]

            # The individual histograms
            for i in range(i * self.average_size, (i + 1) * self.average_size):
                ind_dists.append(self.ith_frame_photon_histogram(i, normalized=False))

        else:

            raise Exception("You are attempting to average using an unknown method, try a different averaging option")
            quit()

        # We add along the time bins => we want a distribution that is normalized so we can treat it like probability
        combined_dist = sum(ind_dists)

        # We then normalize the dist so our fitting works better
        total_photons = combined_dist.sum()

        return combined_dist / total_photons, ind_time

    def print_tau_summary(self):
        """ Prints out the taus, two functions of tau and also a error metric """

        # Some spaces to ensure that we can see the print out
        for i in range(5):
            print()

        print("time (s), tau_1, tau_2, average_tau, empirical_tau, mse")

        # Add this to the csv file too
        self.csv.writerow(["time (s)", "tau_1", "tau_2", "average_tau", "empirical_tau", "mse"])

        # The number of frames we will be iterating through will be a function of the window size and averaging method
        adjusted_frame_count = self.averaging_frame_count()

        # Iterate through all of the frames
        for i in range(adjusted_frame_count):

            # for each frame we need to grab that corresponding data
            photon_dist, ind_time = self.average_photon_dist(i)

            # Establish the x axis
            x = [i for i in range(len(photon_dist))]

            # Fitting the model, and accumulating the time of this fit and the respective tau metrics
            metrics = [round(ind_time, 3)] + self.exp_model.fit(x, photon_dist)

            # Add this to the csv file and print it out too
            self.csv.writerow(metrics)
            print(metrics)


class StackedFlimFile(FlimFile):

    def __init__(self, list_of_paths, exp_model, roi):
        """ An extension of the Flim File class to deal with batch processing (here we just place all flim files on top
        of each other w.r.t. time """

        # The file locations (this is in order by time created)
        self.file_paths = list_of_paths

        # we first create a file reader object
        self.file_reader = FileReader()

        # The individual frames joined together
        self.images, self.times = self.create_stacked_flim_images()

        # Total number of frames across all of the flim files
        self.frame_count = self.images.shape[0]

        # The resolution of the flim experiment
        self.image_width = self.images.shape[3]

        # The current frame that is being visualized
        self.current_frame = 0

        # The type of averaging that we will be performing
        self.averaging_type = "sliding"

        # the number of frames that we would like to average
        self.average_size = 1

        # the exponential model
        self.exp_model = exp_model

        # We send in the roi object from the overarching class to work as a mask for the data
        self.roi = roi

        # The scale for the image (changed this such that the user can actually change it themselves in live time
        self.image_rescaling = IMAGE_RESCALING

        # The channel the experiment was run in (switch for red and green)
        self.channel = 0

    def create_stacked_flim_images(self):
        """ :returns all of the flim data matrices joined on the first axis, and the corresponding times  """

        # Collect all of the image data for the individual files
        individual_images = []

        # The collection of times from all of the individual files
        individuals_times = []

        # Iterate through all of the paths
        for path in self.file_paths:

            # create a new file reader (not sure if this is necessary but it was giving weird bugs when i didnt
            self.file_reader = FileReader()

            # Ask this reader to read the first path
            self.file_reader.read_imageFile(path)

            # Add this data to the individual frames collection
            individual_images.append(np.asarray(self.file_reader.image))

            # Get the time data from the frames
            individuals_times.append([convert_time(time) for time in self.file_reader.acqTime])

        # Concatenate the flim experiment image into a single array
        joined_images = np.concatenate(individual_images, axis=0)

        # Concatenate the flim experiment time data into a single array
        joined_times = list(np.concatenate(individuals_times))

        # Align all of the times by the initial time
        adj_times = [time - joined_times[0] for time in joined_times]

        return joined_images, adj_times
