import numpy as np
import cv2
from algorithm.individual import IndividualBrush
from operations.operator import Mutation, Crossover, Selection
from herustic.fitness import IndividualImportance


cv2.useOptimized()

class PaintingPopulation:
    """
    A population contains a lot of individuals that act independently,
    this population updates over time and tries to improve to reach
    its objective. A population represents a canvas of that has the
    same shape as the objective shape, it is initially empty, but
    as the population changes, each individual applies a specific brush.
    A canvas is always changing, although it has has a memory of past
    generations of the population, many individuals may be destroyed
    between generations as consequence of specific operators, but their
    marks are still saved in the canvas
    Attributes:
    self.objective (matrix): original image to be painted, it is the objective
                             to which the algorithm must arrive
    self.size  (float): the initial size of the population
    self.canvas (matrix): represents the white image with all the brushes
    self.individuals (array): list of all the brushes in the canvas in the current generation
    self.operators (array): list of operators that are applied to the population on an update
    """

    def __init__(self, objective, initial_size, total_steps):
        self.objective = objective
        self.size = initial_size
        self.canvas = None
        self.individuals = []
        self.operators = [Mutation(), Crossover(), Selection()]
        self.min_size = 0.05
        self.max_size = 1
        self.total_steps = total_steps
        self.step_size = (self.max_size - self.min_size) / (total_steps * 0.3)
        self.generation = -1
        # calculate importance of the image based on filter
        self.objective_angle,self.objective_magnitude =self. _img_gradient()   
        self.blur = 50
        self.importance = cv2.cvtColor(self.objective_magnitude, cv2.COLOR_GRAY2BGR)
        self.importance = cv2.GaussianBlur(self.importance, (0, 0), self.blur)
        self.importance_heuristic = IndividualImportance()
        self.randomize()
        
    def _img_gradient(self):
        colored = cv2.cvtColor(self.objective, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(colored, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(colored, cv2.CV_32F, 0, 1, ksize=1)
        objective_magnitude, objective_angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        objective_magnitude /= objective_magnitude.max()
        return objective_angle,objective_magnitude
    
    def update_size(self):
        self.generation += 1
        # update size
        # limit min_size

        value = max(self.max_size - self.generation * self.step_size, self.min_size)
        IndividualBrush.max_size = value*0.85
        IndividualBrush.min_size = value
        
        self.blur = max(1, self.blur - 0.1)
        self.importance = cv2.cvtColor(self.objective_magnitude, cv2.COLOR_GRAY2BGR)
        self.importance = cv2.GaussianBlur(self.importance, (0, 0), self.blur)

    def update(self):
        self.update_size()
        # apply error to each mutation
        for ind in self.individuals:
            image = IndividualBrush.brushes[ind.brush]
            width = image.shape[1]
            height = image.shape[0]
            pos_x, pos_y = ind.pos
            ind.importance = self.importance_heuristic.error(self.importance[pos_y:pos_y + height, pos_x:pos_x + width])

        # apply operators
        for operator in self.operators:
            self.individuals = operator.op(self.individuals)
        # update colors
        for ind in self.individuals:
            color = self.objective[ind.pos[1], ind.pos[0]]
            ind.set_color(color)

    def randomize(self):
        for i in range(self.size):
            brush = IndividualBrush(self.objective_magnitude,self.objective_angle)
            brush.randomize()
            color = self.objective[brush.pos[1], brush.pos[0]]
            brush.set_color(color)
            self.individuals.append(brush)

    def image(self):
        """
        Given all the individual brushes
        we can generate an image from all of them
        :return: image of brushes
        """
        if self.canvas is None:
            self.canvas = np.zeros((self.objective.shape[0], self.objective.shape[1], 3), dtype=np.uint8)
            self.canvas.fill(255)
        
        for ind in self.individuals:
            image = IndividualBrush.brushes[ind.brush]
            dim = (int(image.shape[0] * ind.size), int(image.shape[1] * ind.size))
            image = cv2.resize(image, dim)
            direction = self.objective_angle[ind.pos[1], ind.pos[0]]
            image = PaintingPopulation.rotate_image(image, direction)
            self.insert_image(ind.pos, ind.color, image)
        return self.canvas

    def insert_image(self, pos, color, image):
        # calculate size and position
        width = image.shape[1]
        height = image.shape[0]
        pos_x = pos[0]  # x == col
        if pos_x >= self.canvas.shape[1] - width:
            pos_x = self.canvas.shape[1] - width
        pos_y = pos[1]  # y == row
        if pos_y >= self.canvas.shape[0] - height:
            pos_y = self.canvas.shape[0] - height

        alpha_canvas = 1.0 - image
        # apply color to image
        foreground = (1.0 - self.importance[pos_y:pos_y + height, pos_x:pos_x + width]) * (image * color)
        background = alpha_canvas * self.canvas[pos_y:pos_y + height, pos_x:pos_x + width]
        self.canvas[pos_y:pos_y + height, pos_x:pos_x + width] = cv2.add(foreground, background)

    @staticmethod
    def rotate_image(image, angle):
        width, height = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (height, width))