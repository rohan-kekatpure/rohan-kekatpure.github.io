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
electric currents finding paths of least resistance. The optimizing propensity of entities is so ingrained is us that it
causes us sometimes to flip the direction of causality (e.g. in forensics). Instead of analysing the behavior to
guess the objective, we theorize the objective and try to predict the behavior. This strategy is well summarized in the
cultural aphorism "follow the money".

Given the omnipresence of optimization problems, there is a large pool of simple and intuitive problem candidates for
our exploration. We're want a known problem that is conceptually rich. Once such problem is the [isoperimetric
problem](https://bit.ly/2ErbGK2) whose statement reads:

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
        j = np.random.randint(0, n_points)
        r[j] += np.random.uniform(-.01, .01)
        new_perim = perimeter(r, theta)
        r *= constraint_val / new_perim
        new_area = area(r, theta)
        if new_area > best_area:
            best_area = new_area
            best_r = r
        else:
            r = old_r

    return best_r, theta
```

The implementation closely follows the description. One only needs to remember that the perturbed list of points
needs to be scaled to make sure the perimeter remains unchanged. The Python implementation isn't too different from

<figure>
    <img src="{{site.url}}/assets/img/greedy.gif" alt='hello' width='800' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>


## Solution using hand-coded gradient descent
## Solution using deep learning in Pytorch


