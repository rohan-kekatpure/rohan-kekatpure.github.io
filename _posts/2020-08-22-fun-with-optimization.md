---
layout: post
title: "Fun with optimization"
author: "Rohan"
categories: journal
tags: [documentation,sample]
image:
---
## Solving old problems with new tricks

One great way learn a new algorithm is to pick a known problem apply the new algorithm to it. A 'known problem' is any
problem whose solution is known or can be readily guessed. This way it is easier to build intuition for the steps that
the algorithm takes in its solution process.

In this post we will apply this trick to learn three simple but common numerical optimization algorithms with the help
of a known problem. As a side benefit, we get to watch some visually satisfying animations of an optimization algorithm
arrive to its solution.

Optimization is all around us. Various physical entities are constantly solving some form of optimization problem: water
running downhill (finding state of least energy), light rays trying traveling on fastest routes between two points,
electric currents finding paths of least resistance. Given the omnipresence of optimization problems, there is a large
pool of simple and intuitive problem candidates for our exploration. We're want a known problem that is conceptually
rich. Once such problem is the [isoperimetric problem](https://bit.ly/2ErbGK2) whose statement reads:

>Among all closed curves in the plane of fixed perimeter, which curve maximizes the area of its enclosed region?

In other words, given a closed loop of an inelastic thread the problem asks us to find the the shape of that loop that
will maximize the given area. In an [earlier post]({% post_url 2020-08-11-calculus-of-variations %}) we saw a
theoretical solution to this problem. In the present post we will solve it using code.

## Constructing the cost function

To solve with code, a problem needs to be specified precisely. In particular, we need to carefully specify the inputs,
the objective function and the constraints.

Our input is a list of  points on an arbitrary loop of a given perimeter. We can easily generate a random list of
points by generating pairs of random real numbers $(x, y)$ drawn from some distribution. How can we ensure they form
a closed loop?

Points on a loop one characteristic. Vaguely, for (a list of  discrete) points on a loop, we can always tell if given two
points are next to each other. I.e. points on a loop have a notion of _ordering_. The way to make sure our random point
set belongs to a loop is to _order_ them. A simple way to get an ordered list of  random points on a plane is to loop
through angles between $0$ through $360^\circ$ and assign a random radius to each point. Another way is to
generate random $(x, y)$ tuples and sort them by their angle $\theta = \tan^{-1}\frac{y}{x}$. The following
code shows our point-generation.

```python
def generate_points(n_points):
    theta = np.linspace(-np.pi, np.pi, n_points)
    radius = np.random.uniform(0.5, 1.5, size=(n_points,))
    return radius, theta
```

The above function returns the list of  points in the polar $(r, \theta)$ coordinates. We can convert them into cartesian
$(x, y)$ coordinates using a convenience function as follows:

```python
def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
```

The $(x, y)$ representation is easier to plot and when plotted our initialized loop looks like this:
<figure>
    <img src="{{site.url}}/assets/img/loop.png" alt='hello' width='800' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>

Next we need to define the computation of the perimeter and the area. Mathematically the perimeter $S$ and the area $A$
of a curve $y = f(x)$ are given by $S = \int \sqrt{1 + y'^2}dx$ and $A = \int y dx$. Next we will show how to code
these formulas

### Computation of perimeter

The mathematical formula for perimeter is complicated because it involves a derivative. Fortunately, if we have a
table of $(x, y)$ values we can avoid the use of derivative and directly obtain the perimeter. We will first state
the formula and then deconstruct it to write the code.

$$
\begin{equation*}
S = \sum_{0}^{N}\sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}
\end{equation*}
$$

The above equation states that if we have $N$ points in the set, then we loop over the index and add up the Pythagorean
distance between consecutive points. Translated into code:

```python
def perimeter(x_list, y_list):
    N = len(points)
    S = 0.0
    for i in range(N - 1):
        x1 = x_list[i]  # current x
        y1 = y_list[i]  # current y
        x2 = x_list[i+1]  # next x
        y2 = y_list[i+1]  # next y
        S += sqrt((x2 - x1) ** 2  + (y2 - y1) ** 2)

    return S
```

Of course, since `for` loops in Python are slow, a more efficient implementation uses vectorized operations in Numpy.
The version actually used (but functionally equivalent to the above code) is:

```python
def perimeter(x_list, y_list):
    dx = np.diff(x_list)
    dy = np.diff(y_list)
    perim = np.sqrt((dx * dx + dy * dy)).sum()
    return perim
```

### Computation of area

Remember that we have _two_ equivalent version of our list of points. One is as a list of $(x, y)$ tuples and the other
is as a set of $(r, \theta)$ tuples. The function `pol2cart()` listed above can overt the $(r, \theta)$ list to the $(x,
y)$ list. Because our points are generated to be uniformly spaced spaced in $\theta$ (by the `generate_points()`
function), it turns out that the $(r, \theta)$ version is easier for area computation than the $(x, y)$ version. In
the $(r, \theta)$ representation the formula for area is $A = \int \frac{1}{2}r^2d\theta$. Numerical integration
functions can easily integrate this function if given a list of $(r, \theta)$ points. A popular numerical
integration technique is [Simpsons rule](https://en.wikipedia.org/wiki/Simpson%27s_rule) whose
[Python implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simps.html) is available
in the popular Scipy library. With the use of this ready-made function, our area computation code is particularly
simple:

```python
from scipy.integrate import simps
def area(r, theta):
    return simps(r * r, theta) / 2.0
```

### Enforcing the perimeter constraint

The constraint of our problem is that the perimeter of the loops is constant to a given value. For sake of
concreteness lets assume this constant value is $C = 2\pi \approx 6.28$. A simple way to enforce this constraint is
to compute the perimeter $S$ at each iteration and scale all points in the list by the ratio of computed perimeter to
the desired value. For the $(x, y)$ representation, both $x$ and $y$ need this scaling. For the $(r, \theta)$ version
only $r$ needs to be scaled, since $\theta$ is the angle and doesnt change with scaling of length.

We have now specified how to compute the objective function (i.e the area) the constraint (i.e. the perimeter) and
shown how to enforce the constraint. We're now ready to apply different algorithms to see how each algorithm
evolves to the area-maximizing shape.

## Solution using greedy algorithm

The greedy algorithm is simple iterative algorithm. In each iteration, we modify the shape slightly by adding small
random values to coordinate of a randomly selected point in the loop and recompute the area. If the new area is larger
than the previous area, we replace the old set of points with the new (perturbed) set of points. If not, we retain the
old set of points.

```python
def greedy(r, theta, n_iterations=1000):
    n_points = r.shape[0]
    constraint_val = 2 * np.pi
    best_r = r.copy()
    best_area = area(best_r, theta)
    x0, y0 = pol2cart(r, theta)
    for i in range(n_iterations):
        old_r = r.copy()
        j = np.random.randint(0, n_points)  # index of selected point
        r[j] += np.random.uniform(-.01, .01) # small perturbation to r_j
        new_perim = perimeter(r, theta)
        r *= constraint_val / new_perim  # ensuring perimeter is unchanged
        new_area = area(r, theta)
        if new_area > best_area:
            best_area = new_area
            best_r = r
        else:
            r = old_r

    return best_r, theta
```

The implementation closely follows the description. One only needs to remember that the perturbed list of points
needs to be scaled to make sure the perimeter remains unchanged. Other than that the logic is straightforward.
Following animation shows how the greedy algorithm evolves to its optimum shape.

<figure>
    <img src="{{site.url}}/assets/img/greedy.gif" alt='hello' width='800' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>

After 25000 iterations, the shape is pretty close to a circle (but not exactly). Letting it run for a while longer
will gets it closer to a circle. With the constraint value $\approx 2\pi$ the final area is $\approx \pi$.

## Solution using hand-coded gradient descent

## Neural network

Finally we will attempt to solve the isoperimetric problem using a neural network. We will use Pytorch as to code
our neural network, though we will only be needing a simple 1 hidden-layer-network, not a deep architecture. To begin
we create our 1-layer network architecture:

```python
class Inet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Inet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=False),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out, bias=False)
        )

    def forward(self, input_):
        return self.main(input_)
```

Usually neural networks are used to solve supervised learning problems. We have a list of input-output pairs:
Image/labels, audio/words, or sentence/sentiment and we seek to minimize the prediction error. The isoperimetric
problem, on the other hand, is a pure constrained optimization problem. There are no inputs; we simply seek to arrange
$N$ ordered points to maximize the area formed by their loop while keeping the perimeter fixed. Therefore our input will
be a constant scalar of value `x = 1.0` and output will be `n_points` radius values (one for each value of $\theta$
between $0^\circ$ and $360^\circ$).

The loss function will be defined on output values only. Since Pytorch functions are differentiable, we can construct
very sophisticated loss functions. For instance, we can directly construct a constrained loss function **without** the
need to explicitly ensure that the perimeter constrained is satisfied at every iteration. In other words, in Pytorch we
can construct a loss function with Lagrange multipliers directly (We will look at how to translate this into code
shortly):

$$
\begin{align*}
J(r) = -A(r, \theta) + \lambda_1 \Big[S(r, \theta) - C\Big]^2
\end{align*}
$$

The loss function above says that we have an optimization objective $J$ (also called as the loss function) which depends
on the list of radius values $r = [r_1, r_2, \ldots, r_N]$. It is the sum of the (negative) area $-A$ and the deviation
between the computed perimeter $S$ and its constraint value $C$. Remember that we already have code for $A$ and $S$.

Before moving to code, lets convince ourselves why solving that formula is equivalent to solving our problem. The
function $J$ will be minimized when $-A$ is minimized (i.e. $A$ is maximized) and the deviation between computed and
constrained perimeters is $0$. A list $r$ of radius values that maximizes $A$ and minimizes deviation between $A$ and
$C$ is exactly what we're after. So if we get such a list, then it means we've solved the problem.

Once we understand this, the coding part is straightforward:

```python
def loss(r, theta, lambda1):
    C = torch.tensor(2 * np.pi)
    x, y = pol2cart(r, theta)
    area = 0.5 * torch.trapz(r * r, theta)
    perim = perimeter(x, y)
    J = -area + lambda1 * (perim - C) ** 2
    return J
```

With the loss function in hand, we can put everything together: we instantiate our neural net, fix our learning rate
and iterate till the loss is not minimized (or it stops changing). The following code accomplishes this.

```python
def nnet_optimizer(r, theta, learning_rate, lambda1):
    num_points = r.shape[0]
    theta = torch.tensor(theta)
    r = torch.tensor(r)
    model = Inet(1, 100, num_points)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    input = torch.tensor([1.])
    for i in range(n_iterations):
        r = model(input)
        loss = iso_loss(r, theta, lambda1, lambda2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return r, theta
```

Following animation shows how the greedy algorithm evolves to its optimum shape.

<figure>
    <img src="{{site.url}}/assets/img/deepnet.gif" alt='hello' width='800' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>



