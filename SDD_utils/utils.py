# Utility functions for processing annotations

from agent import Agent

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
    return annotations_by_agent


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
        read_agent_num = int(line[0])
        if read_agent_num == current_agent_num:
            annotation_lists[current_agent_num].append(line)
        else:
            annotation_lists.append([])
            current_agent_num += 1
            annotation_lists[current_agent_num].append(line)
    return annotation_lists




    
x = read_annotations("annotations.txt")
test = Agent(x[0])
print(repr(test.actions[5]))
