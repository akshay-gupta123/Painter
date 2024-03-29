import numpy as np
import cv2
from random import randint, uniform, random,seed,randrange
import time
cv2.useOptimized()

class IndividualBrush:
    """
    Individual Brush is a sample of the genetic population, it represents a
    brush inside the painting, this brush must be as similar as possible
    to the aspect of the painting at its specific position
    (based on the provided fitness function), a brush is in theory
    an immutable object, an operator can only create a new one based on
    this one
    Attributes:
    self.pos (tuple): position of the brush relative to the painting
    self.direction (float): rotation degree of  the brush
    self.color (array): color of the brush
    self.brush (str): filename of the brush to paint
    self.size (float): relative size of the brush
    Static Variables:
    brushes (array): list of strings, each string is a brush type
    min_pos_x (int): the min x position where the brush can be placed
    max_pos_x (int): the max x position where the brush can be placed
    min_pos_t (int): the min t position where the brush can be placed
    max_pos_t (int): the max t position where the brush can be placed
    min_direction (float): the min angle of the brush
    max_direction (float): the max angle of the brush
    min_size (float): max value for the relative size
    max_size (float): min value for the relative size
    """
    brushes = []
    min_pos_x = 0
    max_pos_x = 0
    min_pos_y = 0
    max_pos_y = 0
    min_direction = -90
    max_direction = 90
    min_size = 1
    max_size = 2

    def __init__(self,magnitude,angle):
        self.pos = (0, 0)
        self.direction = 0
        self.color = np.array([0, 0, 0])
        self.brush = None
        self.size = 0
        self.importance = 0
        self.magnitude = magnitude
        self.angle = angle
        
    def randomize(self):
        """
        Given the parameters limits, each parameter of the
        brush is randomize between its possible values
        :return: None
        """
        # select a position
        seed = time.time()
        self.pos = (randint(IndividualBrush.min_pos_x, IndividualBrush.max_pos_x - 1),
                    randint(IndividualBrush.min_pos_y, IndividualBrush.max_pos_y - 1))
        # select a direction
        localMag = self.magnitude[self.pos[1]][self.pos[0]]
        localAngle = self.angle[self.pos[1]][self.pos[0]] + 90 #perpendicular to the dir
        self.direction = randrange(-90, 90)*(1-localMag) + localAngle
        # select a brush
        self.brush = randint(0, len(IndividualBrush.brushes) - 1)
        # select size
        self.size = uniform(IndividualBrush.min_size, IndividualBrush.max_size)

    def randomize_item(self):
        """
        Given the parameters limits, each parameter of the
        brush is randomize between its possible values
        :return: None
        """
        seed = time.time()
        # select a position
        selection = randint(0, 1)
        if selection == 0:
            self.pos = (randint(IndividualBrush.min_pos_x, IndividualBrush.max_pos_x - 1),
                    randint(IndividualBrush.min_pos_y, IndividualBrush.max_pos_y - 1))
        # select a direction
        selection = randint(0, 1)
        if selection == 0:
            localMag = self.magnitude[self.pos[1]][self.pos[0]]
            localAngle = self.angle[self.pos[1]][self.pos[0]] + 90 #perpendicular to the dir
            self.direction = randrange(-90, 90)*(1-localMag) + localAngle
            #self.direction = uniform(IndividualBrush.min_direction,  IndividualBrush.max_direction)
        
        # select a brush
        selection = randint(0, 1)
        if selection == 0:
            self.brush = randint(0, len(IndividualBrush.brushes) - 1)
        # select size
        selection = randint(0, 1)
        if selection == 0:
            self.size = uniform(IndividualBrush.min_size, IndividualBrush.max_size)

    def set_color(self, color):
        # color is not randomized, it is obtained from the
        # original image
        self.color = color

    def __lt__(self, other):
        return self.importance > other.importance

    @staticmethod
    def merge(parent_a, parent_b):
        sibling = IndividualBrush(parent_a.magnitude,parent_a.angle)
        selection = randint(0, 1)
        sibling.pos = ((parent_a.pos[0] + parent_b.pos[0]) // 2, (parent_a.pos[1] + parent_b.pos[1]) // 2)
        
        #selection = randint(0, 1)
        #if selection == 0:
        #    sibling.direction = parent_a.direction
        #else:
        #    sibling.direction = parent_b.direction
        
        localMag = sibling.magnitude[sibling.pos[1]][sibling.pos[0]]
        localAngle = sibling.angle[sibling.pos[1]][sibling.pos[0]] + 90 #perpendicular to the dir
        sibling.direction = randrange(-180, 180)*(1-localMag) + localAngle
        
        selection = randint(0, 1)
        if selection == 0:
            sibling.brush = parent_a.brush
        else:
            sibling.brush = parent_b.brush
        
        sibling.size = uniform(IndividualBrush.min_size, IndividualBrush.max_size)
        sibling.importance = (parent_a.importance + parent_b.importance) // 2
        return sibling

    @staticmethod
    def add_brush(brush):
        IndividualBrush.brushes.append(cv2.imread(brush) / 255.0)