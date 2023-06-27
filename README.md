# The History of Neural Networks Part 2: Rosenblatt's Perceptron

[Dr. Kais Dukes](https://github.com/kaisdukes)

*This is the second article in an educational series on the history of the math behind neural networks. For part 1, see [McCulloch-Pitts: The First Computational Neuron](https://github.com/kaisdukes/mcculloch-pitts-neuron). As with the previous article, to appeal to a wider audience, I've minimized technical jargon and provided simplified explanations. This article includes accompanying code in Python. Published on [LinkedIn](https://www.linkedin.com/pulse/history-neural-networks-part-2-rosenblatts-perceptron-dr-kais-dukes) on June 27, 2023.*

![](https://github.com/kaisdukes/rosenblatt-perceptron/blob/main/neurons.jpeg)

## Frank Rosenblatt

In 1957, [Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) published pioneering work on the first machine learning algorithm for artificial neurons, known as the perceptron. He helped revolutionize the field of artificial intelligence through his work on enabling computers to learn from experience. Let's reflect on the significance of this for a moment. Almost all recent breakthroughs in AI are based on neural networks. These AIs, often hosted in the cloud, operate across vast numbers of computers, running networks of neurons with billions, sometimes trillions of parameters. These model parameters (mathematically, weights and biases) are inspired by the structure of the brain and simulate synaptic plasticity, which allows neurons to adjust the strength of their connections to facilitate learning and memory.

The most efficient artificial neural networks run on Graphics Processing Units. Interestingly, GPUs were originally designed for efficient processing of geometry in 3D computer games. The same type of operations used in these games, parallel calculations, turned out to be directly applicable to AI. If you're wondering why Nvidia is now worth over a trillion dollars, this has largely been driven by the extreme demand for GPUs due to recent advancements in artificial intelligence.

Those who knew Rosenblatt personally at Cornell in the 1950s saw him as a shy genius and a polymath (a deep expert in multiple areas), including computing, mathematics, neurology, astronomy, and music. It's remarkable to think that we can now talk with AIs, and for certain tasks, they have surpassed human ability. The AI community highly values Rosenblatt's groundbreaking invention of the first learning algorithm for computational neurons.

## Was the Perceptron the First Machine Learning Algorithm?

The reason it's so hard to be definitive is that defining 'learning' in this context isn't easy. If we take 'machine learning' in its broadest sense to mean automatically constructing models from data, you could argue that earlier and simpler statistical methods such as linear regression, or least squares, also construct models from data and have been known for centuries since the time of [Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss). However, we can confidently say that Rosenblatt's perceptron was the first machine learning algorithm specifically designed for computational neurons. As we shall see later on, it's fascinating that least squares is also one way to understand how the perceptron learns.

## The McCulloch-Pitts Neuron

As we discussed in the [previous article](https://github.com/kaisdukes/mcculloch-pitts-neuron), Rosenblatt's work on perceptrons built on the work of [Warren McCulloch](https://en.wikipedia.org/wiki/Warren_Sturgis_McCulloch) and [Walter Pitts](https://en.wikipedia.org/wiki/Walter_Pitts). The McCulloch-Pitts (MCP) neuron was inspired by the neuroscience known at the time: the brain is composed of neurons interacting via synapses, transmitting either excitatory or inhibitory signals. In the McCulloch-Pitts model, the neuron fires if the sum of excitatory inputs reaches a certain threshold, unless inhibited by other inputs. The MCP neuron can be thought of as a binary classifier using binary inputs.

## Hebbian Learning

In 1949, psychologist [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) made a significant breakthrough in neuroscience. This additional understanding of the brain, not known to McCulloch-Pitts in 1943, directly enabled Rosenblatt to produce an improved mathematical model of the neuron. Hebb's significant contribution is now known as Hebbian learning. In his seminal book, *The Organization of Behavior*, Hebb proposed that when two neurons fire together, their interconnection strengthens. The principle, summarized as *cells that fire together, wire together* provided a vastly improved model of the brain.

Inspired by Hebb's ideas, Rosenblatt aimed to develop a learning algorithm for an artificial neuron: the perceptron. While Hebb's theory didn't specify how connection strength should change, it laid the foundation for Rosenblatt's remarkable insight to adjust the perceptron's weights based on the difference between the desired and actual output.

Rosenblatt, though primarily a psychologist, also had a strong background in mathematics and physics. His interest in the emerging field of cybernetics, the study of automatic control systems in both machines and living organisms, gave him the foundation he needed to develop the perceptron. He wanted to create an improved computational model of the neuron capable of learning from experience, with the eventual goal of building a physical learning machine.

## Rosenblatt's Perceptron

Perceptrons are binary classifiers. They take multiple weighted inputs, sum them, add a bias, and then use an activation function (described below) to predict the class of the input as either 1 or 0. In the simplest form of the perceptron, Rosenblatt made the following key modifications to the McCulloch-Pitts neuron:

**Numeric inputs:** He replaced the binary inhibitory/excitatory inputs with continuous equivalents via real-valued inputs and weights. Unlike the binary inputs of the MCP neuron, the perceptron could process numbers in any range.

**Weighted sum:** He changed the MCP neuron's activation rule, which was based on reaching a threshold, to a rule that fires if a weighted sum plus a bias is reached.

**Machine learning:** In a groundbreaking move, Rosenblatt introduced the first machine learning algorithm for neurons, known as the perceptron learning rule.

The foundational work of Rosenblatt that includes these developments is presented in two of his early papers:

*The Perceptron: A Perceiving and Recognizing Automaton* (1957). This was a technical report commissioned by Cornell, where Rosenblatt was working at the time. The report outlined a proposal for building a physical machine that could recognize patterns and learn from experience.

*The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* (1958). Published in Psychological Review, this paper was more accessible to a wider, non-technical audience. Rosenblatt discusses the philosophical and conceptual aspects of the perceptron and frames it as a model of information processing in the brain. The paper helped place the perceptron within the broader context of cognitive psychology and neuroscience, and now has almost 20,000 citations.

If you read Rosenblatt's original papers, you will find that concepts used more recently have evolved significantly. He describes sensory, association and response units. These correspond to the modern design of an input layer, the process of weighted summation and the output layer in simplified neural networks. Although using different terms, the basic principles he introduced, like learning weights from data and using weights to make predictions, remain central to machine learning and AI.

## How Perceptrons Learn

Compared to the far more advanced machine learning algorithms used today, the perceptron is based on simpler math. The classic perceptron is made up of the following parts:

1. The neuron has multiple inputs. Each input is a real number.
2. Each input also has an associated weight, a strength for that input.
3. As well as the weights, the neuron also includes another number known as the bias. Including a bias term means the neuron can solve a wider range of problems (with a bias, the classifying hyperplane can have an offset from the origin).

To learn from data, examples from the training dataset are shown to the neuron one at a time. The neuron makes a prediction, and an error is calculated. The perceptron learning rule is then applied to update the weights of the model in order to reduce the error for the given example.

Let's now try to understand this rule intuitively. Let `x` and `w` be the input and weight vectors respectively, and `b` the bias. The neuron fires if `w·x + b >= 0`, where the dot represents the vector dot product. Another way to write this would be using the [Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function), defined as `H(x) = 1` if `x >= 0` and 0 otherwise. In the context of neural networks, we call this an activation function. The perceptron's output is then `H(w·x + b)`.

The perceptron learning algorithm proceeds as follows:

1. Initialize the weights to zero, or use random initialization for better performance.
2. For each training example, if the model misclassifies, adjust the weights.

## Loss Functions

We can understand how to minimize errors by introducing a loss function, also known as an error function. Although not always used in practice, for a simplified view one possible choice of loss function for the perceptron is the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE). This is defined using the squared difference between the expected classification and the weighted sum. Other choices for loss functions include the perceptron criterion or hinge loss.

The Python code accompanying this article utilizes a perceptron as a binary classifier for two types of flowers in Fisher's classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). While the code uses the perceptron criterion for practicality, the explanation below uses MSE as a learning exercise. MSE is also more applicable to other machine learning models.

Let `y` be the expected classification (1 or 0) for a specific example with input `x`. We then define the MSE loss function by summing over training examples:

```
L = 1/2 * ∑(y - (w·x + b))^2
```

We can use a form of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to train the perceptron and minimize the error. We find the [gradient](https://en.wikipedia.org/wiki/Gradient) of the loss function with respect to the weights and bias, and adjust them in the direction of the negative gradient. Think of this like trying to find your way down a hill by choosing the steepest path down at each step.

While MSE is a useful loss function for learning about perceptrons, it might not always be the most effective choice. However, it makes gradient descent and weight updates mathematically elegant and efficient. Note that we use the term `1/2` instead of `1/n` to simplify the derivative calculations in the next section. Once again, it's worth mentioning Gauss, the renowned 18th century genius mathematician and polymath, who introduced the method of least squares.

## The Perceptron Learning Rule

Let's now derive the formula for updating the weights and bias. In more advanced training methods, like batch gradient descent, we would consider multiple training examples simultaneously. But to keep things simple, we'll update the weights one example at a time. While it's tempting to consider the standard perceptron learning algorithm as performing Stochastic Gradient Descent (SGD), that's not strictly true. You can think of the perceptron algorithm as a unique variant of SGD. In true SGD, the weights are updated after each example, regardless of whether the example is classified correctly or not.

Focusing on one training example, we calculate the gradient ([grad](https://en.wikipedia.org/wiki/Gradient)) of the loss function, which consists of the partial derivatives with respect to each weight and bias:

```
∂L/∂wi = -(y - (w·x + b)) * xi
∂L/∂b = -(y - (w·x + b))
```

Multivariable calculus, like riding a bicycle, can come back to you once you start practicing. If you're struggling to recall the math, you can see this using the chain rule. In its general form, the chain rule involves [Jacobian matrices](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) when dealing with vector-valued functions. However, this simplifies when both the inner and outer functions are scalars.

Let `u = y - (w·x + b) = y - (∑(wj * xj) + b)`

For the ith component, we have `∂u/∂wi = -xi` since the other terms vanish.

Therefore:

`∂L/∂wi = ∂L/∂u * ∂u/∂wi = u * (-xi) = -(y - (w·x + b)) * xi`

Note that these derivatives, focusing on a single data point at a time, are not the true gradients of the overall loss function, but are a sufficient approximation used in gradient descent-like methods.

We can then update the weights and bias according to the negative gradient of the loss function:

```
wi ← wi - a * ∂L/∂wi
b ← b - a * ∂L/∂b
```

where `a` is the learning rate, determining the step size in the gradient descent update. This results in the following update rules:

```
wi ← wi + a * (y - (w·x + b)) * xi
b ← b + a * (y - (w·x + b))
```

To understand the use of the negative gradient, consider a simple 1-dimensional function such as `y = x^2`, with a minima at `x = 0`. If you were trying to find the minimum of this function using gradient descent, you would start with a guess such as `x = 3`, and compute the gradient at that point: `dy/dx = 6`. Because the slope is positive, it means the function is increasing, so you want to move in the opposite direction of the derivative.

As the learning rate `a` is typically chosen to be small, this update process will gradually adjust the weights and bias, reducing the error incrementally. The lower the learning rate, the more gradual the adjustment process will be, giving the model better stability, but requiring more iterations through the training dataset to reach optimal performance.

## Linear Classification via Hyperplanes

Perceptrons are well known to be linear classifiers. This means that a single perceptron has a simple geometric interpretation. What the perceptron learns during training is how to separate two classes of multidimensional data using a ‘decision boundary’, which in this case is a hyperplane.

In geometry, a hyperplane is a subspace of one dimension less than its ambient space. For example, if the ambient space is 2-dimensional (like a sheet of paper), then its hyperplanes are 1-dimensional lines. If the ambient space is 3-dimensional (like our everyday world), then its hyperplanes are 2-dimensional planes.

The equation for a hyperplane can be written as:

```
w·x + b = 0
```

which is the same formula the perceptron uses to calculate its output before applying the activation function. The perceptron creates a hyperplane decision boundary. The weight vector `w` defines the orientation of the hyperplane, and the bias `b` defines a shift from the origin.

For a given point `x`, when `w·x + b > 0`, we are on one side of the hyperplane, and when `w·x + b < 0`, we are on the other.

## Did Rosenblatt Invent Deep Learning?

It is unfortunate that Rosenblatt's influence on deep learning architectures has been overlooked, and he is now almost always only remembered for the single neuron model. While Rosenblatt didn't invent deep learning as it is known today, his contributions were crucial for the development of modern AI.

The concept of ‘deep learning’ refers to modern neural networks composed of many layers of neurons connected to each other. Deep, in the sense of multiple deep layers.

Typically, the story told is that only with the rediscovery of backpropagation in the 1980s could we begin constructing more complex artificial neural networks. However, this overlooks Rosenblatt's pioneering work on other neural designs, including multilayer perceptrons and models with recurrent connections. He detailed his research on a variety of neural architectures in his 1962 book *Principles of Neurodynamics*. This was not only avidly studied by Rosenblatt's peers, but also served as an important reference for future AI researchers who went on to make pivotal advancements in the 1980s.

While Rosenblatt and his contemporaries faced challenges in training more advanced neural networks due the limitations of computers at the time, their efforts laid the foundations for the advanced AI we have today.

# Python Code

The code in this repository demonstrates how to implement a single perceptron in Python without relying on machine learning libraries. We define a simple perceptron class and functions for training, prediction, and evaluation.

## Getting Started

This project uses [Poetry](https://python-poetry.org).

First, clone the repository:

```
git clone https://github.com/kaisdukes/rosenblatt-perceptron.git
cd rosenblatt-perceptron
```

Install Poetry using [Homebrew](https://brew.sh):

```
brew install poetry
```

Next, set up the virtual environment:

```
poetry install
```

Use the Poetry shell:

```
poetry shell
```

Test the perceptron:

```
python tests/perceptron_test.py
```