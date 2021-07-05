from genetic.algorithm.population import PaintingPopulation
from genetic.herustic.fitness import imageError
from genetic.algorithm.individual import IndividualBrush
import warnings
warnings.filterwarnings("ignore")
import argparse
import cv2
import sys

cv2.useOptimized()

class Genetic:
    """
    A genetic works by creating a brush population and iterating,
    until a painting condition is met (i.e. error is low enough).
    First, a random sample is generated from which different operators
    are applied, this operators modify the population by creating new
    individuals and destroying old ones.
    Attributes:
    self.objective (matrix): original image to be painted, it is the objective
                             to which the algorithm must arrive
    self.margin (float): the quantity of error that is accepted when searching
                         for a solution, it determines when will the algorithm
                         stop searching
    Static Variables:
    MAX_ITERATIONS (int): given that it is possible that the algorithm loops forever
                          we specify a limit
    """
    MAX_ITERATIONS = 5000

    def __init__(self, objective,max_iterations):
        self.objective = cv2.imread(objective)
        self.MAX_ITERATIONS = max_iterations
        
    def start(self, size):
        # initialize variables
        IndividualBrush.add_brush("brushes/1.jpg")
        IndividualBrush.add_brush("brushes/2.jpg")
        IndividualBrush.add_brush("brushes/3.jpg")
        IndividualBrush.add_brush("brushes/4.jpg")
        IndividualBrush.max_pos_x = self.objective.shape[1]
        IndividualBrush.max_pos_y = self.objective.shape[0]

        it = 0
        population = PaintingPopulation(self.objective, size, self.MAX_ITERATIONS)
        output_per_generation = 30
        # start loop
        while it < self.MAX_ITERATIONS:
            if it % output_per_generation == 0:
                print("Iteration", it)
            # save frame to disk
            sample = population.image()
            if it % output_per_generation == 0 or it==self.MAX_ITERATIONS-1:
                cv2.imwrite("frames/sample_" + Genetic.integer_padding(it, len(str(self.MAX_ITERATIONS))) + ".png", sample)
            # update loop conditions
            it += 1
            # update population
            population.update()

    @staticmethod
    def integer_padding(i, padding):
        i = str(i)
        while len(i) < padding:
            i = '0' + i
        return i

def args_parser():
    parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
    
    parser.add_argument("-img_path", metavar="", type=int, required=True,
                        help="Image path")
    parser.add_argument("-max_iterations", metavar="", type=int, default=1000,
                        help="Number of generation to stimulate. Default is 1000")
    parser.add_argument("-initial_population",type=int,default=50,
                        help="Initial population")
    
    args = parser.parse_args()
    return args

    
def info_print(args):
    """
    This function prints the input arguments from argparse when calling this script via python shell.
    Args:
        args [argparse.Namespace]: argument namespace from main.py
    Returns:
        None
    """
    print(37*"-")
    print("Random Art with Deep Neural Networks:")
    print(37*"-")
    print("Script Arguments:")
    print(17*"-")
    for arg in vars(args):
        print (arg, ":", getattr(args, arg))
    print(17*"-")
    return None


def main():
    args = args_parser()
    ## print out information on shell
    info_print(args)
    print('Starting brushing...')
    
    gen = Genetic(args.img_path,args.max_iterations)
    gen.start(args.initial_population)


if __name__ == '__main__':
    main()