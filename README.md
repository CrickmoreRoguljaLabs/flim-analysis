# flim-analysis
Code for calculating fluorescent lifetimes from .flim files for the Crickmore Lab

[Stephen updates](#SCT)

# How to use

- Navigate to "main.py"
- Change "flim_data_path" to the location of the either a folder of .flim files or an individual .flim file
  - Use the folder option in the case where your experiemnt occured over multiple .flim files and you wish to analyze them together
    - Note: this requires that you use the same ROI for all of the files 
  - If you want to analyze only a indivdual .flim file, make sure the extension is .flim
- Change "flim_save_path" to the location where the program should output the .csv file 
  - Note: if you use the same "flim_save_path" it will write over the data from the first run
- Run "main.py" after a few seconds the interface should pop up
- GUI Explained 
  - Normalized phtonon counts vs time graph (left)
    - The graph on the left displays both the photon histogram (blue) from a particular frame from the .flim file, and the current model fit (yellow)
  - Frame Tools: 
    - Frame index = The frame that is being displayed on the right and whose photon distribution is being graphed on the left
    - Frame Scaling = Increase the size of the frame image on the right (depending on experiment resolution the frame might be too small to make an roi)
    - Channel = Change the channel of analysis, make sure you set this to avoid processing noise from the off channel
  - ROI Construction: 
    - First press "Create ROI"
      - This should toggle to blue
    - Create the boundaries of your ROI by clicking on the frame visualizer on the right
      - each click (besides the first) will connect the point where you just clicked with the last point you created, you should see a line connecting these points
    - To connect your final point to the first point press "Create ROI"
      - This should toggle the button back to white/gray
      - This should close the shape and all of the pixels outside of the ROI should go to black (this data is not lost, it is just not used during the processing)
    - If you want to create another ROI, once you have untoggled "Create ROI" press "Destroy ROI"
  - Graph Interface: 
    - linear vs log scale = Change the y axis of the graph on the left to a log or linear scale 
    - double vs mono exponential = Change the model type used in the model fit from either a single or double exponential model 
      - If you use a double exponential you can expect tau_1 and tau_2 in the output file, if you use a single exponential you will only get tau_1
  - Averaging Tools: 
    - Frames to Average = The number of frames that you would like to group together, here grouping means their photon distributions will be combined and fit in one shot
    - Sliding vs Disjoint Window = The type of averaging window used for the analysis 
      - Example: Consider 10 consecutive frames -> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
      - Sliding => [1, 2, 3], [2, 3, 4], [3, 4, 5], ... where each [] is frames combined
      - Disjoint => [1, 2, 3], [4, 5, 6], [7, 8, 9], ... where each [] is frames combined
  - Data Output:
    - Once you have constructed your ROI, selected model type and averaging types press this to calculate the flourescent lifetimes for each of the indivudal frames
      - The output will be of the following form = time (s), tau_1, tau_2, average_tau, empirical_tau, mse
  - Frame Visualization (right)

# Bug Fixes 
List of bugs + fixes that have been found when implemening this code on different computers within the lab

- ROI colors inverted/wrong => comment out line 130 in flim_files.py
- pip install libtiff issues => Build from http://www.libtiff.org


# SCT
<a name=SCT></a>
Install with `python -m pip install .` You can use the old (Marco) version with `crickflim` in the command line or the new with `sctflim`. You can specify
a file path on the command line, or by clicking