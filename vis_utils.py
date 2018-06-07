"""
Utility functions for visualizing reward functions
"""

import numpy as np
import matplotlib.pyplot as plt

def heatmap_2d(reward_matrix, title='', display_text = True, blocking = True, figure_num = 1):
    """
    Draws and then displays a 2d heatmap using matplotlib

    Args:
        reward_matrix (MxN array): Array mapping from position to reward value

    Returns:
        None
    
    """
    if blocking:
        plt.figure(figure_num)
        plt.clf()

    plt.imshow(reward_matrix, interpolation = "nearest")
    plt.title(title)
    plt.colorbar()
    plt.gca().invert_yaxis()

    if display_text:
        for y in range(reward_matrix.shape[0]):
            for x in range(reward_matrix.shape[1]):
                plt.text(x, y, "%.1f" % reward_matrix[y, x],
                        horizontalalignment = "center",
                        verticalalignment = "center"
                )

    if blocking:
        plt.ion()
        plt.show()
        input("Press Enter to continue...")

#test = np.array([0., 0.54802036, 0.729247, 0.49788868, 0.63374794, 0.7746057, 0.88566065,
        #0.6542306, 0.72214484, 0.608043, 0.6168779, 0.65174246, 0.9048787, 0.92718637,
        #0.7961359, 0.5894675, 0.5575371, 0.66720784, 0.66517085, 0.6181151, 0.7783362,
        #0.9470316, 0.9004493, 0.72237384, 0.57893395, 0.71991646, 0.7173019, 0.677485, 
        #0.8575239, 0.69307, 0.66837156, 0.6964057, 0.6707367, 0.79715437, 0.9830394, 1.])

#test = np.reshape(test, (6, 6))
#heatmap_2d(test)
