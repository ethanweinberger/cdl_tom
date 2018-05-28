# Utility functions for handling Stanford Drone Data annotations

from enum import Enum
from box import Box

class Agent_Type(Enum):
    BICYCLIST    = 1
    PEDESTRIAN   = 2
    SKATEBOARDER = 3
    CART         = 4
    CAR          = 5
    BUS          = 6

class Agent:
    
    def __init__(self, lines):
        self.type  = self.get_type_from_lines(lines)
        self.boxes = self.get_boxes_from_lines(lines)


    def get_type_from_lines(self, lines):
        """
        Extracts the agent type from the annotation lines
        for a given agent

        *For internal use only*

        Args:
            Lines from annotations.txt corresponding to a given agent

        Returns:
            Agent_Type instance

        """
        initial_line        = lines[0]
        split_line          = initial_line.split()
        agent_type_string   = split_line[-1]
        agent_type_string   = agent_type_string.strip('"')

        if agent_type_string == "Bicyclist":
            return Agent_Type.BICYCLIST
        elif agent_type_string == "Pedestrian":
            return Agent_Type.PEDESTRIAN
        elif agent_type_string == "Skateboarder":
            return Agent_Type.SKATEBOARDER
        elif agent_type_string == "Cart":
            return Agent_Type.CART
        elif agent_type_string == "Car":
            return Agent_Type.CAR
        elif agent_type_string == "Bus":
            return Agent_Type.BUS
        else:
            raise ValueError("Agent type not one of the six possibilities")

    def get_boxes_from_lines(self, lines):
        """
        Extracts the bounding boxes from each frame
        from the annotation lines for a given agent

        *For internal use only*

        Args:
            lines: Lines from annotations.txt corresponding to 
                   a given  agent

        Returns:
            box_list: List of Box objects
        """
        box_list = []
        for line in lines:
            new_box = self.get_box_from_line(line)
            box_list.append(new_box)
        return box_list

    def get_box_from_line(self, line):
        """
        Extract box coordinates from a given annotation line

        *For internal use only*

        Args:
            line: A line from annotations.txt

        Returns:
            new_box: Box object extracted from line
        """
        split_line = line.split()
        xmin = int(split_line[1])
        ymin = int(split_line[2])
        xmax = int(split_line[3])
        ymax = int(split_line[4])

        new_box = Box(xmin, ymin, xmax, ymax)
        return new_box










