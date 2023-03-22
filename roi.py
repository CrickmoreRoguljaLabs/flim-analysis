

# The rescaling value for the image (this is becuase experiments happen at different resolutions so we want the user
# to be ablke to rescale their image so they can make a proper ROI)
IMAGE_RESCALING = 1  # Default starting value


class ROI:

    def __init__(self):

        # Boolean that represents if we are in the middle of constructing the ROI
        # this will be true when the construct_roi button button is toggled as on
        self.constructing_roi = False

        # Boolean representing if the roi has been created already
        self.roi_exists = False

        # A list of ROI coordinate points (here points = x,y coordinates of polygon points)
        self.roi_points = []

        # Rescaling factor for the image s.t. users can view experiments at different resolutions
        self.image_rescaling = IMAGE_RESCALING

    def update_constructing(self, checked):
        """ Updates internal booleans to reflect the fact that we are either switching in or out of the roi
            construction phase
             - in the case that we are switching out of the phase we want to link the last point that was constructed
                to the first one picked """

        # In the case that we are finishing the construction we want to set roi_exists to true
        if self.constructing_roi and not checked:

            # Ensure that the user is at least making a triangle for the ROI, raise exception if not
            if len(self.roi_points) < 3:
                raise Exception("You tried to create an ROI with less than 3 points, delete the roi and use more points")

            else:

                self.roi_exists = True

        # Reflect the new state of the button
        self.constructing_roi = checked

    def integrate_new_roi_point(self, new_x, new_y):
        """ The pixel location of a mouse click on the current image
            if we are in the process of constructing the roi we want to keep this data; otherwise we can ignore it """

        if self.constructing_roi:

            # Rescale the points (we do this so we can switch the coordinates at any time)
            self.roi_points.append([new_x // self.image_rescaling, new_y // self.image_rescaling])

    def remove_roi_points(self):
        """ Resets the roi points and sets the exists field to false """

        self.roi_points = []
        self.roi_exists = False
