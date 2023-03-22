import sys
from PyQt5.QtWidgets import QApplication
from flim_gui import FlimAnalysisGUI

# ------------------------------------------------------------------------------------------------------

# The location of the flim files to be analyzed
# You can either put the a single flim file (in that case the extension needs to be a .flim)
# Or you can put a folder of flim files (use this if your experiment occurs over multiple flim files)
flim_data_path = "/Users/marco/Desktop/flim_practice/deets/AKAR/20220717/NP2AKAR/NP2AKARBaseline002.flim"

# The location you would like to save your flim analysis data to
# The extension should be a .csv
flim_save_path = "/Users/marco/Desktop/flim_save.csv"


# ------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Create the application
    app = QApplication(sys.argv)

    # Initialize the flim gui and display it
    main_window = FlimAnalysisGUI(flim_data_path, flim_save_path, app)
    main_window.show()

    app.exec()
