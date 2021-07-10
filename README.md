# Painter

A collection of techniques to generate art.

# Painter

A collection of techniques to generate art.

## Neural-net-random-art

Create a grayscale or colour image with predefined size `image_height` and `image_width` using fully connected neural networks.The generation of images only requires python `numpy` and `matplotlib`.

### Execution

For the `nn.py` programm `argparse` is used to define several input parameters:

```python 
parser = argparse.ArgumentParser(description="Generate random art with a deep neural network")
parser.add_argument("-img_height", metavar="", type=int, default=512,
                   help="Image height of created random art. Default is 512") 
parser.add_argument("-img_width", metavar="", type=int, default=512,
                   help="Image width of created random art. Default is 512") 
parser.add_argument("-colormode", metavar="", type=str, default="RGB",
                   help="How image color should be generated. Options are ['BW', 'RGB', 'HSV', 'HSL']. By default this
                    value is 'RGB'")    
parser.add_argument("-alpha", metavar="", type=str, default="False",
                   help="Whether or not to add a alpha channel for the image. Default is False")
parser.add_argument("-n_images", metavar="", type=int, default=1,
                   help="Number of images to generate. Default is 1")    
parser.add_argument("-n_depth", metavar="", type=int, default=10,
                   help="Number of layers for the neural network. Default is 10") 
parser.add_argument("-n_size", metavar="", type=int, default=15,
                   help="Number of neurons in each hidden layer. Default is 15")
parser.add_argument("-activation", metavar="", type=str, default="tanh",
                   help="Activation function to apply on the hidden layers. Default is 'tanh'")      
parser.add_argument("-z1", metavar="", type=float, default=-0.618,
                   help="Input variable 1 to insert determinism into the random art. The value should be between -1 and 1. Default 
                    is -0.618")    
parser.add_argument("-z2", metavar="", type=float, default=+0.618,
                   help="Input variable 2 to insert determinism into the random art. The value should be between -1 and 1. Default 
                   is +0.618")
args = parser.parse_args()
```

### Examples

<img src="/assets/nn.png">

## Genetic Brush Painiting

Genetic brush painting is an image painter that mimics the process of painting of an image into a canvas. This artificial intelligence project is powered on genetic algorithms where each brush in the painting represents an individual of a population, this brushes mutate and improve with time, thus creating more realistic paintings overall. To improve the perfomance of the algorithm we provide additional information to the brushes such as the color of the original image are their position, and the importance of that position. The importance of a position of the image is calculated on a convoluted filter of vertical and horizontal edges. The genetic algorithm applies three kind of operators; mutation, selection, and crossover. Mutation works by randomizin a specific feature of the current generation of brushes. After the mutation a crossover between different pairs of individuals. Finally we select the brushes based on their importance as explained before.

### Execution

For the `gen.py` programm `argparse` is used to define several input parameters:

``` python
parser = argparse.ArgumentParser(description="Generate brush painting with genetic algorithm")
    
parser.add_argument("-img_path", metavar="", type=int, required=True,
                    help="Image path")
                    
parser.add_argument("-max_iterations", metavar="", type=int, default=1000,
                    help="Number of generation to stimulate. Default is 1000")
                    
parser.add_argument("-initial_population",type=int,default=50,
                    help="Initial population")
```

### Example
<p float="left">
  <img src="/assets/gen_1.jpg" width="300" />
  <img src="/assets/gen_2.png" width="300" /> 
</p>

## Markov Chain Image Generation

A markov chain is used to store pixel colours as the node values and the count of neighbouring pixel colours becomes the connection weight to neighbour nodes. To generate an image, we randomly walk through the chain and paint a pixel in the output image. The result is images that have a similar colour pallette to the original, but none of the coherence. They still look nice though.

### Execution

For the `markov.py` programm `argparse` is used to define several input parameters:

```python
parser = argparse.ArgumentParser(description="Generate markov chain version of your image")
    
parser.add_argument("-img_path", metavar="", type=int, required=True,
                    help="Image path")
                    
parser.add_argument("-bucket-size", metavar="", type=int, default=10,
                    help="Bucket size for compressing colors. Default is 10")
                    
parser.add_argument("-four_neighbour",type=bool,default=True,
                    help="Number of neighbours to use")

```

### Example

<p float="left">
  <img src="/assets/mc_1.jpg" width="300" />
  <img src="/assets/mc_2.jpg" width="300" /> 
</p>
