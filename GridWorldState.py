'''GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.

class GridWorldState(object):
    ''' Class for Grid World States '''

    def __init__(self, x, y):
        self._is_terminal = False
        self.data = [x, y]
        self.x = round(x, 5)
        self.y = round(y, 5)

    def is_terminal(self):
        return self._is_terminal

    def set_terminal(self, is_term = True):
        self._is_terminal = is_term
    
    def __hash__(self):
        return hash(tuple(self.data))
    
    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + ")"

    def __eq__(self, other):
        return isinstance(other, GridWorldState) and self.x == other.x and self.y == other.y
