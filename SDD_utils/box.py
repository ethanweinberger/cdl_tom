#Class for representing bounding boxes in provided annotations

class Box(object):
    
    def __init__(self, xmin, ymin, xmax, ymax):
        """
        Class initializer

        Args:
            xmin: Top left x-coordinate of the bounding box
            xmax: Bottom right x-coordinate of the bounding box
            ymin: Top left y-coordinate of the bounding box
            ymax: Bottom right y-coordinate of the bounding box

        Returns:
            Box object
        """

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __repr__(self):
        return "<Box: xmin:%s ymin:%s xmax:%s ymax:%s>" % \
                (self.xmin, self.ymin, self.xmax, self.ymax)
