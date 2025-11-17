# Deep Learning Framework

A lightweight deep learning framework implemented from scratch in NumPy, inspired by TensorFlow v1.

## Features

- **Automatic Differentiation**: Expression-based symbolic differentiation system
- **Neural Network Layers**: Dense (fully connected), Conv2d, RNN, normalization layers
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Optimizers**: SGD with gradient descent
- **Advanced Features**: Skip connections, batch normalization, layer normalization

## Project Structure

```
autodiffAndRegression/    # Automatic differentiation and linear regression
basicNeuralNetworks/      # Feedforward neural networks with basic layers
convolutionalNetworks/    # CNNs with Conv2d and MaxPool layers
recurrentNetworks/        # RNNs with vanilla RNN cells
advancedLayers/           # Normalization and skip connections
```

## Usage

Each module contains training demos that can be run independently:

### Regression Models
```bash
cd autodiffAndRegression/regressors
python train.py
```

### Basic Neural Networks
```bash
cd basicNeuralNetworks/autograd
python train_basic_nn.py
```

### Convolutional Neural Networks
```bash
cd convolutionalNetworks/autograd
python train_cnn.py
```

### Recurrent Neural Networks
```bash
cd recurrentNetworks/autograd
python train_rnn_many_to_one.py
python train_rnn_many_to_many.py
```

### Advanced Layers
```bash
cd advancedLayers/autograd
python train_with_normalization.py
python train_with_skip_connections.py
```

## Requirements

Key dependencies:
- numpy
- matplotlib
- scikit-learn
- tqdm

For detailed requirements, see individual module directories.

## Implementation Details

This framework implements:
- Forward and backward propagation for all layers
- Gradient checking for validation
- Modular architecture with abstract base classes
- Parameter management and optimization

## License

Educational project for deep learning implementation practice.
