# Deep-Compression v0

## Samples:

Training samples : 10000
Validation samples : 2500
Testing samples : 5000


## Data preparation:
* A : 100 x 200 matrix
* X : 200 x 1 vector
* Y : 100 x 1 vector
* s : sparsity value

X is a vector with random numbers at s number of random locations and zero at remaining locations.
A is a randomly generated matrix
Y = A*X
x_{0} : zero vector of length 200

## Training 
Repeat the stages 1 to s

Stage - i:
  true_X, true_Y, true_A will be entering the system
  x_{i} : x_{i-1} vector of length 200
  residue = true_Y - A* x_{1}
  input to Neural network = transpose(A) * residue
  
  Initialize a Neural network = Fully connected network --> you can find the architecture in model.py
  input shape : batch_size x 200
  output shape : batch_size x 200
  Output from neural network -> global max pooling --> get the location l and value v
  Update the vector x_{1} with the value v at the location l

Calculate the MSE loss for x_{s} and true_x
Calculate the gradients based on loss and update the weights for the network.

## Testing
* Load the weights of best model for a given sparsity level
* x_{0} being the zero vector of shape 1 x 200. Predict the recovered x from the model.
* Calculate the MSE loss for predicted x and true x. Normalize the MSE loss and report it.

## Results
Results for the Deep compression v0 is available in results.txt file.


