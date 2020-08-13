---
layout: post
title: "Fun with optimization"
author: "Rohan"
categories: journal
tags: [documentation,sample]
image:
---
### Solving old problems with new tricks

One great way to build intuition for a new algorithm is to pick a problem whose solution we can easily guess and then
test if and how the algorithm find the right solution. In this post we will dive into a couple of simple but common
numerical optimization algorithms with the help of a simple problem. As a side benefit, we get to watch some
satisfying visualizations of an optimization algorithm arrive to its solution.

There is a large pool of problems to choose from for testing a optimization algorithms. This is because mathematical
optimization is an old field and also because optimization problems are all around us. Various natural entities are
constantly solving some form of optimization problem. One way to understand someone's behavior is to ask what
optimization problem they are solving (e.g. 'follow the money').

Even inanimate entities are solving optimization problems of their own: objects are trying to find configurations of
least energy, light rays are trying to find fastest routes between two points, electric currents are trying to
find paths of least resistance.

### The isoperimetric problem

We will start with a much simpler problem than the ones mentioned above. We're want a problem
that is easy to think about and yet is conceptually rich. Once such problem is the [isoperimetric problem](https://en
.wikipedia.org/wiki/Isoperimetric_inequality#The_isoperimetric_problem_in_the_plane){:target="_blank"}. The problem
statement reads

>Among all closed curves in the plane of fixed perimeter, which curve maximizes the area of its enclosed region?

In other words, given a closed loop of an inelastic thread the problem asks us to find the the shape of that loop
that will maximize the given area. Most of us would rightly guess the answer to be a circle. Yet, it is a non trivial
exercise to actually _prove_ that circle is indeed the correct shape.

We will therefore start by first mathematically deriving the solution using calculus of variations. This section is
not at all necessary to understand the code portion which follows and can be safely skipped. However, if you dont
mind a bit of math, then the derivation is not quite as bad.

### Solution using calculus of variations

The solution to the isoperimetric problem will give us a two dimensional curve $y = f(x)$ whose perimeter is a fixed
number $p$ and whose area $A$ is maximum. The recipe in calculus of variations is to express our
objective function (with constraints) in the standard form:

$$
\begin{equation}
\label{eq:cov}
 J[y] = \int L(x, y, y')dx
\end{equation}
$$

Then the minimizing function $y(x)$ is found by solving the Euler-Lagrange equation:

$$
\begin{equation}
\label{eq:el}
 \frac{\partial L}{\partial y} = \frac{d}{dx}\frac{\partial L}{\partial y'}
\end{equation}
$$

There a bit of theory leading to the above equations, which is outside our scope. But once the suffice it to say that a
wide array of optimization problems fall within the framework of \eqref{eq:cov} and \eqref{eq:el} With this recipe, the
only task remaining is to express our problem in the standard form and solve it. In practice this is pretty
mechanical task, which we carry out below.

The formulas for the perimeter and area of a generic curves are well known:

$$
\begin{align}
 & A = \int y dx \\
 & P = \int \sqrt{1 + y'^2} dx
\end{align}
$$

which allows us to express our objective function for area as:

$$
 J[y] = \int y dx
$$

To add the constant perimeter constraint, we use the method of Lagrange multipliers and write the modified objective
as:

$$
\begin{equation}
 J[y] = \int y dx - \lambda \int \sqrt{1 + y'^2} dx
      = \int \left(y - \lambda \sqrt{1 + y'^2}\right) dx
\end{equation}

$$

which allows us to identify our $L$ function as

$$
\begin{equation}
    L(x, y, y') = y - \lambda \sqrt{1 + y'^2}
\end{equation}
$$

with partial derivatives

$$
\begin{equation}
    \frac{\partial L}{\partial y} = 1, \quad
    \frac{\partial L}{\partial y'} = \frac{-\lambda y'}{\sqrt{1 + y'^2}}
\end{equation}
$$

Substituting into the Euler-Lagrange equations we get:

$$
\begin{equation}
    1 = \frac{d}{dx} \frac{-\lambda y'}{\sqrt{1 + y'^2}}
\end{equation}
$$

After a bit of manipulations, the solution of this second-order ODE comes out to

$$
\begin{equation}
    (x + c_1)^2 + (y + c_2)^2 = \lambda^2
\end{equation}
$$

which is the equation of a circle, thus proving that circle is the solution to the isoperimetric problem.

### Solution using greedy algorithm
### Solution using hand-coded gradient descent
### Solution using deep learning in Pytorch


