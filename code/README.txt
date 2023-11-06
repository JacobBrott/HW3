Description:
This Python script defines and uses a series of functions to perform optimization on a neural network model to find the set of inputs that minimize the output of the network.

Approach:
The script takes a mathematical approach to optimize neural network inputs using gradient descent. The process involves the following key steps:

1. Activation Function:
   - A Rectified Linear Unit (ReLU) function is implemented to introduce non-linearity to the network layers.

2. Network Operations:
   - The network processes inputs through functions for vector dot product, matrix-vector multiplication, and vector addition, which are basic linear algebra operations.

3. Forward Pass:
   - Inputs are propagated forward through the network using the defined linear algebra operations and the ReLU activation function.

4. Gradient Computation:
   - A numerical approach is taken to approximate the gradient of the loss function with respect to the inputs. This involves slightly perturbing the inputs and measuring the change in the output.

5. Optimization Loop:
   - Gradient descent is employed to iteratively adjust the inputs in the direction that reduces the output. The learning rate dictates the step size for each iteration.

6. Network Parsing:
   - The script is capable of reading network configurations from a file, parsing them into a usable structure for the optimization routines.

7. Main Execution:
   - For each network read from the file, the script applies the optimization function and writes the minimizing inputs to an output file.

Note: The 'networks.txt' file should be structured properly with layer counts, weights, biases, and ReLU activation flags for the script to parse correctly.

Usage:
To run the optimization process, execute the script from the command line:
$ python main.py

This will generate a 'solutions.txt' file containing the optimal inputs for each parsed network from 'networks.txt'.

Dependencies:
- Python 3
- No external libraries required; the script uses Python's standard library.

I used ChatGPT to give me comments on my code, help get gradient decent and matrix multiplication working, and to write this readme file. 