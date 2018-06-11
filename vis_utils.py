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
