import sys

import napari
from PyQt5.QtWidgets import QApplication

from crickflim.flim_gui import FlimAnalysisGUI
from crickflim.sct.napari.napari_gui import NapariWrapper



# ------------------------------------------------------------------------------------------------------

# The location of the flim files to be analyzed
# You can either put the a single flim file (in that case the extension needs to be a .flim)
# Or you can put a folder of flim files (use this if your experiment occurs over multiple flim files)
#flim_data_path = "/Users/marco/Desktop/flim_practice/deets/AKAR/20220717/NP2AKAR/NP2AKARBaseline002.flim"
flim_data_path = "/Users/stephen/Desktop/CrickmoreTest/flim-analysis/NP5XXMCamui100uMDopa010.flim"

# The location you would like to save your flim analysis data to
# The extension should be a .csv
flim_save_path = "/Users/stephen/Desktop/CrickmoreTest/flim-analysis/dummytest.csv"


# ------------------------------------------------------------------------------------------------------

def sct():
    flim_data_path = sys.argv[1]
    n_w = NapariWrapper(flim_data_path)
    napari.run() 

def main():
    # Create the application
    app = QApplication(sys.argv)

    # Initialize the flim gui and display it
    main_window = FlimAnalysisGUI(flim_data_path, flim_save_path, app)
    main_window.show()

    app.exec()

if __name__ == "__main__":
    main()

