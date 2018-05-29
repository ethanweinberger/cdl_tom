# Class for representing action (change in x/y coordinates between positions)
# Purpose of class is to make code more readable

class Action(object):
    
    def __init__(self, old_pos, new_pos):
        self.x = new_pos.x - old_pos.x
        self.y = new_pos.y - old_pos.y
