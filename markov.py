import warnings
warnings.filterwarnings("ignore")
import argparse
from PIL import Image
import numpy as np
import random
import os
from collections import defaultdict, Counter
import cv2
import sys


class MarkovChain(object):
    def __init__(self, bucket_size=10, four_neighbour=True):
        self.weights = defaultdict(Counter)
        self.bucket_size = bucket_size
        self.four_neighbour = four_neighbour

    def normalize(self, pixel):
        return pixel // self.bucket_size

    def denormalize(self, pixel):
        return pixel * self.bucket_size

    def get_neighbours(self, x, y):
        if self.four_neighbour:
            return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        else:
            return [(x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y + 1),
                    (x - 1, y - 1),
                    (x - 1, y + 1),
                    (x + 1, y - 1)]

    def train(self, img):
        """
        Train on the input PIL image
        :param img:
        :return:
        """
        width, height = img.size
        img = np.array(img)[:, :, :3]
        for x in range(height):
            for y in range(width):
                # get the left, right, top, bottom neighbour pixels
                pix = tuple(self.normalize(img[x, y]))
                for neighbour in self.get_neighbours(x, y):
                    try:
                        self.weights[pix][tuple(self.normalize(img[neighbour]))] += 1
                    except IndexError:
                        continue
        self.directional = False

    def generate(self, initial_state=None, width=512, height=512):
        fourcc = cv2.VideoWriter_fourcc(*'MP4v')
        writer = cv2.VideoWriter('markov_img.mp4', fourcc, 24, (width, height))
        
        if initial_state is None:
            initial_state = random.choice(list(self.weights.keys()))
        if type(initial_state) is not tuple and len(initial_state) != 3:
            raise ValueError("Initial State must be a 3-tuple")
        img = Image.new('RGB', (width, height), 'white')
        img = np.array(img)
        img_out = np.array(img.copy())

        # start filling out the image
        # start at a random point on the image, set the neighbours and then move into a random, unchecked neighbour,
        # only filling in unmarked pixels
        initial_position = (np.random.randint(0, width), np.random.randint(0, height))
        img[initial_position] = initial_state
        stack = [initial_position]
        coloured = set()
        while stack:
            x, y = stack.pop()
            if (x, y) in coloured:
                continue
            else:
                coloured.add((x, y))
            try:
                cpixel = img[x, y]
                node = self.weights[tuple(cpixel)]  # a counter of neighbours
                img_out[x, y] = self.denormalize(cpixel)
            except IndexError:
                continue

            
            keys = list(node.keys())
            neighbours = self.get_neighbours(x, y)
            counts = np.array(list(node.values()), dtype=np.float32)
            key_idxs = np.arange(len(keys))
            ps = counts / counts.sum()
            np.random.shuffle(neighbours)
            for neighbour in neighbours:
                try:
                    col_idx = np.random.choice(key_idxs, p=ps)
                    if neighbour not in coloured:
                        img[neighbour] = keys[col_idx]
                except IndexError:
                    pass
                except ValueError:
                    continue
                if 0 <= neighbour[0] < width and 0 <= neighbour[1] < height:
                    stack.append(neighbour)
        writer.release()
        return Image.fromarray(img_out)

def args_parser():
    parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
    
    parser.add_argument("-img_path", metavar="", type=int, required=True,
                        help="Image path")
    
    parser.add_argument("-bucket-size", metavar="", type=int, default=10,
                        help="Bucket size for compressing colors. Default is 10")
    parser.add_argument("-four_neighbour",type=bool,default=True,
                        help="Number of neighbours to use")
    
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

def  main():
    ## retrieve arguments and print out in shell
    args = args_parser()
    ## print out information on shell
    info_print(args)
    chain = MarkovChain(bucket_size=args.bucket_size, four_neighbour=args.four_neighbour)
    
    im = Image.open(args.image_path)
    print("Training " + args.image_path)
    chain.train(im)
    
    print("\nGenerating")
    chain.generate().show()

if __name__ == "__main__":
    main()