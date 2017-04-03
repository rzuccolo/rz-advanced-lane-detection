import numpy as np

class Line():
    '''
    Define a class to keep track of lane line detection
    This information will help the main pipeline code to decide if the current polyfit is good or bad
    '''
    def __init__(self):

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]


        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

