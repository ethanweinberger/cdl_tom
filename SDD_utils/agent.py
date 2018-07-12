# Utility functions for handling Stanford Drone Data annotations

from enum import Enum
from box import Box
from position import Position
from action import Action

import numpy as np

class Agent_Type(Enum):
    BIKER        = 1
    PEDESTRIAN   = 2
    SKATER       = 3
    CART         = 4
    CAR          = 5
    BUS          = 6

class Agent:
    
    def __init__(self, lines):
        self.type      = self.get_type_from_lines(lines)
        self.boxes     = self.get_boxes_from_lines(lines)
        self.positions = self.get_positions_from_boxes(self.boxes) 
        self.actions   = self.get_actions_from_positions(self.positions)


    def get_type_from_lines(self, lines):
        """
        Extracts the agent type from the annotation lines
        for a given agent

        *For internal use only*

        Args:
            lines (string list): Lines from annotations.txt corresponding to a given agent

        Returns:
            Agent_Type instance

        """
        initial_line        = lines[0]
        agent_type_string   = initial_line[-1]
        agent_type_string   = agent_type_string.strip('"')

        if agent_type_string == "Biker":
            return Agent_Type.BIKER
        elif agent_type_string == "Pedestrian":
            return Agent_Type.PEDESTRIAN
        elif agent_type_string == "Skater":
            return Agent_Type.SKATER
        elif agent_type_string == "Cart":
            return Agent_Type.CART
        elif agent_type_string == "Car":
            return Agent_Type.CAR
        elif agent_type_string == "Bus":
            return Agent_Type.BUS
        else:
            print(agent_type_string)
            raise ValueError("Agent type not one of the six possibilities")

    def get_boxes_from_lines(self, lines):
        """
        Extracts the bounding boxes from each frame
        from the annotation lines for a given agent

        *For internal use only*

        Args:
            lines (string list): Lines from annotations.txt corresponding to 
                   a given  agent

        Returns:
            box_list (Box list): List of Box objects
        """
        box_list = []
        for line in lines:
            xmin = int(line[1])
            ymin = int(line[2])
            xmax = int(line[3])
            ymax = int(line[4])

            new_box = Box(xmin, ymin, xmax, ymax)
            box_list.append(new_box)
        return box_list


    def get_positions_from_boxes(self, boxes):
        """
        Convert box objects into the center coordinate of the box

        *For internal use only*

        Args:
            boxes (Box list): list of boxes describing an agent's locations

        Returns:
            position_list (tuple list): List of positions corresponding to each box
        """

        position_list = []
        for box in boxes:
            x_center = (box.xmin + box.xmax)/2
            y_center = (box.ymin + box.ymax)/2
            position = Position(x_center, y_center)
            position_list.append(position)
        return position_list

    def get_actions_from_positions(self, positions):
        """
        Get the actions that capture the transformations between consecutive positions.
        Actions represented as (Delta_x, Delta_y)

        *For internal use only*

        Args:
            positions (tuple list): tuples describing an agent's locations

        Returns:
            action_list (tuple list): tuples describing changes between positions
        """
        action_list = []
        for i in range(0, len(positions)-1):
            action = Action(positions[i], positions[i+1])
            action_list.append(action)
        return action_list
            
