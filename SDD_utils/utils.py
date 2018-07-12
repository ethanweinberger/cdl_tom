# Utility functions for processing annotations

import sys
from agent import Agent
from matplotlib.path import Path
from collections import namedtuple

Step = namedtuple('Step', 'cur_state action')

def read_annotations(file):
    """
    Reads in an annotations file provided in the SDD dataset,
    and converts the text file into a list of Agent objects

    Args: 
        file: annotation file

    Returns: 
        annotations_by_agent: List of Agent objects
    """

    with open(file) as f:
        lines = f.read().splitlines()

    annotations_by_agent = divide_annotations(lines)
    agent_list = []
    for annotation_set in annotations_by_agent:
        agent_list.append(Agent(annotation_set))

    return agent_list


def divide_annotations(annotation_lines):
    """
    Divides the lines of an annotation file into
    lists based on the different objects.

    *For internal use only*

    Args: 
        annotation_lines: List of annotation lines

    Returns: 
        annotation_lists: List of lists of annotation lines divided by agent
    """

    annotation_lists = []
    current_agent_num = 0

    annotation_lists.append([])
    for line in annotation_lines:
        line = line.split()
        read_agent_num = int(line[0])
        if read_agent_num == current_agent_num:
            annotation_lists[current_agent_num].append(line)
        else:
            annotation_lists.append([])
            current_agent_num += 1
            annotation_lists[current_agent_num].append(line)
    return annotation_lists

def create_video_grid(vid_width, vid_height, block_dim):
    """
    Create a grid of matplotlib Path's that represent
    a downsampling of the positions in the SDD videos.  We
    use this to determine a trajectory for each agent.

    Args:
        vid_width (int): Width (in pixels) of our video
        vid_height (int): Height (in pixels) of our video
	block_dim (int): Length of one side of our (square) blocks
    Returns:
        grid (List of Path lists): List of lists of matplotlib Paths, 
                with each path representing a block in the grid
    """

    grid = []
    for i in range(0, vid_height, block_dim):
        grid_row = []
        for j in range(0, vid_width, block_dim):
            bottom_left_vertex = (j, i)
            bottom_right_vertex = (j + block_dim, i)
            top_right_vertex = (j + block_dim, i + block_dim)
            top_left_vertex = (j, i + block_dim)

            vertex_list = []
            vertex_list.append(bottom_left_vertex)
            vertex_list.append(bottom_right_vertex)
            vertex_list.append(top_right_vertex)
            vertex_list.append(top_left_vertex)

            path = Path(vertex_list)
            grid_row.append(path)
        grid.append(grid_row)
    print(len(grid) * len(grid[0]))
    return grid
              
def get_agent_positions_in_grid(agent, grid):
    """
    Given an agent and a list of Path lists comprising
    a grid, returns the list of position of the agent
    in that grid. 
    
    Args:
	agent (Agent): Agent object
        grid (List of Path lists): List of Path lists representing
           our downsampled coordinate frame
    Returns:
        grid_position_list (tuple list): List of tuples representing
            positions in the new coordinate frame
    """
    grid_position_list = []

    for position in agent.positions:
        for (grid_row, grid_row_index) in zip(grid, range(len(grid))):
            grid_col_index = _get_position_grid_column(position, grid_row)
            if grid_col_index:
                grid_position_list.append((grid_col_index, grid_row_index))
                break

    return grid_position_list     

def _get_position_grid_column(position, grid_row):
    """
    Given a position and a grid_row in which the position may lie,
    returns the grid column containing the position (if any)
    
    Args:
        position (int tuple): Agent position that we're examining
        grid_row (Path list): List of paths representing a row of our grid
    Returns:
        grid_col_index (int/None): None if no column contains the position,
            otherwise returns the number of the column containing our position
    """
     
    for (box, grid_col_index) in zip(grid_row, range(len(grid_row))):
        if box.contains_point((position.x, position.y)):
            return grid_col_index

    return None

def get_steps_from_position_list(position_list):
    """
    Given a list of an agent's positions, returns a list of the actions
    taken by the agent to achieve those positions
    
    Args:
        position_list (tuple list): List of (x,y) tuples representing the agent's position
                at different time steps
    Returns:
        step_list (Step list): List of strings representing the actions taken
                by the agent
    """

    step_list = []

    for i in range(len(position_list) - 1):
        current_position = position_list[i]
        next_position = position_list[i+1]
        if current_position == next_position:
          continue

        #Cardinal directions
        if (next_position[0] - current_position[0] == 1 and 
            next_position[1] - current_position[1] == 0):
            next_action = "right"
        elif (next_position[0] - current_position[0] == -1 and 
            next_position[1] - current_position[1] == 0):
            next_action = "left"
        elif (next_position[0] - current_position[0] == 0 and 
            next_position[1] - current_position[1] == 1):
            next_action = "up"
        elif (next_position[0] - current_position[0] == 0 and 
            next_position[1] - current_position[1] == -1):
            next_action = "down"

        #Extended directions
        elif (next_position[0] - current_position[0] == 1 and 
            next_position[1] - current_position[1] == 1):
            next_action = "up_right"
        elif (next_position[0] - current_position[0] == -1 and 
            next_position[1] - current_position[1] == 1):
            next_action = "up_left"
        elif (next_position[0] - current_position[0] == -1 and 
            next_position[1] - current_position[1] == -1):
            next_action = "down_left"
        elif (next_position[0] - current_position[0] == 1 and 
            next_position[1] - current_position[1] == -1):
            next_action = "down_right"

        next_step = Step(current_position, next_action)
        step_list.append(next_step)

    return step_list
