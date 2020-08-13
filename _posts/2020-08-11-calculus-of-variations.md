---
layout: post
title: "Classic optimization problems"
author: "Rohan"
categories: journal
tags: [documentation,sample]
image:
---
### Variational techniques

I've been fascinated with the isoperimetric problem ever since hearing about it my 12th grade. Despite the name, the
statement of the isoperimetric problem is simple. One from [Wikipedia](https://en
.wikipedia.org/wiki/Isoperimetric_inequality#The_isoperimetric_problem_in_the_plane){:target="_blank"} reads

>Among all closed curves in the plane of fixed perimeter, which curve maximizes the area of its enclosed region?

Most folks will intuitively and correctly think that the answer is a circle.

There are many problems of this flavor. We include a few more examples below.

1. [Fermat's principle](https://en.wikipedia.org/wiki/Fermat%27s_principle)
    path taken by a ray between two given points is the path that can be traversed in the least time

2. Path of shortest distance connecting a two points is a straight line

3. Among all probability distributions of known mean and variance, the normal distributions has the least entropy

In addition to being fun mathematical problems, these minimization principles play a crucial role in Physics. Expressing
laws of Physics as minimization principles leads to a uniform treatment of many different areas and has potential to
provide new insights. Classical mechanics can be reformulated as Lagrangian mechanics based on a suitable
minization principle. Field equations of gravitation can be derived from a minimization principle. The pattern is to
identify a suitable quantity whose minimization leads to a desired physical law.

Minimization principles also play an important role computationally. If a suitable minimization principle is known
for a problem, it can open up ways to compute solutions to otherwise difficult-to-handle problems. Such
techniques are known as variational techniques. Calculation of orbitals of Helium molecule are a fine example of
application of variational techniques.

Hopefully the above examples have provided a (admittedly vague) sense of why the calculus of variations (COV) and the
variational techniques what it begets are important. In addition to being important, the solution to variational
and optimization problems have a certain magic about them **explain why**

### First principles solution of elementary variational problems

In the remainder of this post, we will solve two classic variational problems in closed form from first principles. The
two problems of choice are the isoperimetric problem and the minimum entropy problem. The presentation below is part of
standard treatment of this subject. Yet going over it in our own way is beneficial for internalizing the subject.

#### Standard form of calculus of variations problems

The standard minimization problems is to find the _point_ $x_0$ which minimizes a given scalar function $f(x)$. The
input variable $x$ can be scalar or a vector, but the function value must be scalar. The way to solve this problem in
standard calculus is to solve the equation $f'(x) = 0$ for $x$.

The minimization problem in calculus of variations is to find the _function_ $f(x)$ which minimizes a scalar
_functional_ $J[f]$. A functional is a function of functions. A standard function $f$ takes an input $x$ and outputs
a number. A functional, on the other hand, takes an entire function $f$ and outputs a single number.

A functional may seem like a new concept, but it is not. Any subroutine which calculates the mean, median, variance, or
area, of an input function between two values is a functional. We have all seen examples of such subroutines in
scientific libraries. For calculus of variations, we just abstract the concept for a mathematical treatment. We can
write the action of a functional $J$ on a function $f$ as

$$
\begin{equation*}
J[f] = \text{a scalar value}
\end{equation*}
$$

Finding an $f(x)$ that minimizes $J[f(x)]$ is a harder problem than finding an $x$ that minimizes $f(x)$. For
a calculus of variations problem, therefore we cannot write a general recipe like solve $f'(x) = 0$. We have to
proceed by identifying special cases. The most common special case is one where $J$ depends on the _integral_ of a
function $L()$ which is itself a function only of $x$, $y =f(x)$ and $y'= f'(x)$. That is

$$
\begin{equation*}
J[f] = \int L(x, y, y')dx
\end{equation*}
$$

This may seem like a hopelessly specific case whose analysis will not be worth the effort. But it turns out that a
variety of practically useful optimization problems can be expressed in the above form or its minor variations.
Both isoperimetric problem as well as a max entropy problem (and many others) are of the above form. An important
point to note is that the function $L$ can encore our optimization objective as well as the constraints. This point
will become clear when we look at the examples.

### Recipe for solving standard COV problem

Once the optimization problem is expressed in the above form, the recipe for solving it to form a differential equation
for $y = f(x)$ and solve that differential equation. The differential equation for $y$ is obtained from the standard
form via the famous [Euler-Lagrange equations](https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation):

$$
\begin{equation}
\label{eq:el}
 \boxed{\frac{\partial L}{\partial y} = \frac{d}{dx}\frac{\partial L}{\partial y'}}
\end{equation}
$$

In computing the derivatives of $L$, we treat $y$ and $y'$ as independent variables. That is, when computing
$\partial L/\partial y$ we do not worry about derivative of $y'$ with respect to $y$. Similarly when computing
$\partial L/\partial y'$ we do not worry about derivative of $y$ with respect to $y'$. Equation \eqref{eq:el} yields
a differential in terms of $x$, $y$ and $y'$ whose solution gives us a $y(x)$.

To add constraints, we use a the method of [Lagrange multipliers](https://en.wikipedia org/wiki/Lagrange_multiplier).
This method modifies the $L(x, y, y')$ function using the constraints and allows us to still use the Euler-Lagrange
equations. Once again, the method of Lagrange multipliers will become clear when we work with examples.

### Isoperimetric problem

To solve the isoperimetric problem we're given a loop of an inelastic wire of perimeter $S$ and asked for a shape
$y = f(x)$ that will maximize the area $A$ of the shape. To be able to use the Euler-Lagrange equation, we need to
derive the $L$ function for the problem. There are standard formulas for the area and perimeter of any curve.

$$
\begin{align*}
 & A = \int y dx \\
 & S = \int \sqrt{1 + y'^2} dx
\end{align*}
$$

These formulas allow the following mathematical statement of the isoperimetric problem:

$$
\begin{align*}
 \text{maximize}&\qquad \int y dx \\
 \text{subject to}&\qquad \int \sqrt{1 + y'^2} dx - P = 0
\end{align*}
$$

Using the method of Lagrange multipliers, we can convert this into

$$
\begin{equation*}
 \text{maximize}\qquad J[y] = \int \left(y - \lambda \sqrt{1 + y'^2} \right) dx,
\end{equation*}
$$

where we have ignored the constant term $\lambda P$ since it wont survive the various differentiations.

This is the standard form if we identify

$$
\begin{equation*}
 L(x, y, y') = y - \lambda \sqrt{1 + y'^2}
\end{equation*}
$$

This form is ready be plugged into the Euler-Lagrange equation. We obtain

$$
\begin{align*}
 \frac{\partial L}{\partial y} &= \frac{d}{dx}\frac{\partial L}{\partial y'} \\
 \Rightarrow 1 &= \lambda \frac{d}{dx} \frac{y'}{\sqrt{1 + y'^2}} \\
 \Rightarrow \frac{x + h}{\lambda} &= \frac{y'}{\sqrt{1 + y'^2}}\\
 \Rightarrow y' = \frac{dy}{dx} &= \frac{\pm(x + h)}{\sqrt{\lambda^2 - (x+h)^2}}
\end{align*}
$$

This is a simple differential equation whose solution is

$$
\begin{equation*}
 (x + h)^2 + (y + k)^2 = \lambda^2
\end{equation*}
$$

which is the standard equation of a circle centered at $(-h, -k)$ and radius $\lambda$.

### Max entropy problem

The max entropy problem is simpler calculus-wise but involves multiple constraints in a single problem. One version
of the max entropy read as follows:

> Among all probability distributions $p(x)$ of known mean $\mu$ and known variance $\sigma^2$ find the one with
largest entropy.

Entropy of a (continuous) random variable $X$ with PDF $p(x)$ is given by $H = -\int p(x)\log p(x) dx\$. The entropy
is our optimization objective and the known mean, variance and probability normalization condition are our
constraints for this problem. As before, we can express the problem mathematically as

$$
\begin{align*}
 \text{maximize}&\qquad \int -p(x)\, \log p(x)\, dx \\
 \text{subject to}&\qquad \int p(x)\, dx = 1 \qquad \ldots \text{normalization}\\
                  &\qquad \int x\, p(x)\, dx = \mu \qquad \ldots \text{known mean}\\
                  &\qquad \int x^2\, p(x)\, dx = \sigma^2 + \mu^2 \qquad \ldots \text{known variance}
\end{align*}
$$

Using the recipe for Lagrange multipliers we can transform the above optimization problem with constraints into the
standard form:

$$
\begin{equation*}
    J[p] = \int L(x, p, p') dx
\end{equation*}
$$

where

$$
\begin{equation*}
    L(x, p, p') = -p \log p - \lambda_1 p - \lambda_2 xp -\lambda_3 x^2p
\end{equation*}
$$

The derivatives of $L$ with respect to $p$ and $p'$ are

$$
\begin{align*}
    \frac{\partial L}{\partial p} &= -1 - \log p -\lambda_1 - \lambda_2 x -\lambda_3 x^2\\ \\
    \frac{\partial L}{\partial p'} &= 0
\end{align*}
$$

Substituting into Euler-Lagrange equation we get

$$
\begin{align*}
    -\log p &= 1 + \lambda_1 + \lambda_2 x + \lambda_3 x^2 \\ \\
    \Rightarrow p(x) &= A e^{-(a x + b x^2)}
\end{align*}
$$

In the last equation we have somewhat simplified the last expression by identifying $A \leftarrow e^{-(1+\lambda_1)}$,
$a \leftarrow \lambda_2$ and $b \leftarrow \lambda_3$. We have the standard integrals (e.g. from Wolfram Alpha)

$$
\begin{align*}
    \int p(x)\, dx &= A\, \sqrt{\frac{\pi}{b}} e^{a^2/4b} = 1 \\
    \int x\,p(x)\, dx &= A\, \sqrt{\frac{\pi}{b}} e^{a^2/4b} \left(\frac{-a}{2b}\right) = \mu \\
    \int x^2\,p(x)\, dx &= A\, \sqrt{\frac{\pi}{b}} e^{a^2/4b} \left(\frac{a^2}{4b^2} + \frac{1}{2b}\right)
                         = \mu^2 + \sigma^2
\end{align*}
$$

The above equations allow us to write the undetermined constants $A$, $a$ and $b$ in terms of given quantities $\mu$
and $\sigma$ as

$$
\begin{equation}
    \label{eq:coeffs}
    A = \sqrt{b/\pi}, \qquad \mu = -a/(2b), \qquad \sigma^2 = 1/(2b)
\end{equation}
$$

With these equations, $p(x)$ can be written as

$$
\begin{equation*}
    p(x) = \sqrt{\frac{b}{\pi}}\, e^{-a^2/4b}\, e^{-(ax + bx^2)}
         = \sqrt{\frac{b}{\pi}}\, e^{-b(x + \frac{a}{2b})^2}
\end{equation*}
$$

Finally substituting the coefficients $A$, $a$ and $b$ from equation \eqref{eq:coeffs} we get

$$
\begin{equation*}
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \, e^{-\frac{(x-\mu)^2}{2\sigma^2}}
\end{equation*}
$$

which is a normal distribution with mean $\mu$ and variance $\sigma^2$.

### Generalizations of Euler-Lagrange equations

The two problems we solved so far used the $L$ function that depended on $x$, $y$ and $y'$. There are many
generalizations of $L$ functions. Three practically important generalizations that immediately follow the one we've
considered are the following.

1. $L(x, y, y', y'')$:

    The $L$ function depends on a single function of a single variable, but the dependence involves higher order
    derivatives. This case could arise in variational problems involving curvature. The Euler-Lagrange equation for
    this case has a second derivative term:

    $$
    \begin{equation*}
        \frac{\partial L}{\partial y}
        - \frac{d}{dx}\frac{\partial L}{\partial y'}
        +\frac{d^2}{dx^2}\frac{\partial L}{\partial y''} = 0
    \end{equation*}
    $$

2. $L(t, f(t), g(t), f'(t), g'(t))$:

    The $L$ function depends on two functions $f$ and $g$, both of which depend on a single variable $t$. This
    case arise for a variational problem defined on a parametric curve. In this case we get two coupled first order
    Euler-Lagrange equations:

    $$
    \begin{align*}
        \frac{\partial L}{\partial f} = \frac{d}{dt}\frac{\partial L}{\partial f'} \\ \\
        \frac{\partial L}{\partial g} = \frac{d}{dt}\frac{\partial L}{\partial g'}
    \end{align*}
    $$

3. $L(x, y, z(x, y), z_x, z_y)$:

    The $L$ function depends on a function of two variables $z(x, y)$ and their first derivatives. This case could
    arise for variational problems in higher dimensions. E.g. isoperimetric problem for a sphere. In this case the
    Euler-Lagrange equation has a term that looks like total derivative:

    $$
    \begin{equation*}
        \frac{\partial L}{\partial z} =
        \frac{\partial}{\partial x}\frac{\partial L}{\partial z_x} +
        \frac{\partial}{\partial y}\frac{\partial L}{\partial z_y}
    \end{equation*}
    $$

Further generalization are possible and the reader is referred to [Wikipedia](https://en.wikipedia
.org/wiki/Euler%E2%80%93Lagrange_equation#Generalizations) for more.

### Isoperimetric problem in parametrized coordinates

We will explore generalization #2 in more detail using the isoperimetric problem again. Instead of directly
seeking the function $y(x)$ we will parametrize the $x$ and the $y$ coordinates using a single dependent variable $t$
and using _two_ functions: $x = x(t)$ and $y = y(t)$. To derive the new $L$ function, we need the formulas for the
area and the perimeter of this parametrized curve. To that end we will consider two successive points along this
curve separated by an infinitesimal distance $\delta t$. The situation is depicted in the figure below.

<figure>
    <img src="{{site.url}}/assets/img/triangle.png" alt='hello' width='500' heigh='500' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>

The area of this infinitesimal triangle is

$$
\begin{align*}
    \delta A &= \frac{1}{2}
    \begin{vmatrix}
        0 & 0 & 1 \\
        x(t) & y(t) & 1 \\
        x(t + \delta t) & y(t + \delta t) & 1
    \end{vmatrix} \\
    &= \frac{1}{2}\left[ x(t) y(t + \delta t) - y(t) x(t + \delta t)\right] \\
    &= \frac{1}{2}[x(t) y'(t) - y(t) x'(t)]\delta t
        \qquad\ldots \text{using } f(t + \delta t) \approx f(t) + f'(t)\delta t  \\
    dA &= \frac{1}{2}(x y' - x'y)dt \qquad \ldots \text{differential notation}
\end{align*}
$$

The total area $A$ enclosed by the curve is the sum over areas of all infinitesimal triangles as the parameter $t$ is
varied over its range: $A = \int \frac{1}{2}(x y' - x' y) dt$.

The perimeter $\delta s$ of the exterior segment of the infinitesimal triangle is

$$
\begin{align*}
    \delta s^2 &= (x(t+\delta t) - x(t))^2 + (y(t+\delta t) - y(t))^2 \\
         &= (x'(t)^2 + y'(t)^2)\delta t^2\\
    ds &= \sqrt{x'^2 + y'^2} dt \qquad \ldots \text{differential notation}
\end{align*}
$$

The total perimeter $S$ is the sum of lengths of all such segments: $S = \int \sqrt{x'^2 + y'^2}dt$.

With the total area and total perimeter in hand, the $L$ function for our constrained optimization problem in the
parametrized coordinates can be written as

$$
\begin{equation*}
    L(t, x, y, x', y') = \frac{1}{2}(xy' - x'y) + \lambda \sqrt{x'^2 + y'^2}
\end{equation*}
$$

As mentioned before, we have two sets of Euler-Lagrange equations, one each for $x$ and $y$. From the $x$ equation we
get

$$
\begin{align}
    \frac{\partial L}{\partial x} &= \frac{d}{dt} \frac{\partial L}{\partial x'} \notag \\
    \frac{y'}{2} &= \frac{d}{dt} \left(-\frac{y}{2} + \lambda\frac{x'}{\sqrt{x'^2 + y'^2}}\right) \notag \\
    y' &= \lambda \frac{d}{dt} \frac{x'}{\sqrt{x'^2 + y'^2}} \notag \\
    \Rightarrow \frac{y + k}{\lambda} &= \frac{x'}{\sqrt{x'^2 + y'^2}} \label{eq:elx}
\end{align}
$$

Similarly for the $y$ equation we get

$$
\begin{align}
    \frac{\partial L}{\partial y} &= \frac{d}{dt} \frac{\partial L}{\partial y'} \notag \\
    \frac{-x'}{2} &= \frac{d}{dt} \left(\frac{y}{2} + \lambda\frac{y'}{\sqrt{x'^2 + y'^2}}\right) \notag \\
    -x' &= \lambda \frac{d}{dt} \frac{y'}{\sqrt{x'^2 + y'^2}} \notag \\
    \Rightarrow \frac{-(x + h)}{\lambda} &= \frac{y'}{\sqrt{x'^2 + y'^2}} \label{eq:ely}
\end{align}
$$

Dividing equation \eqref{eq:ely} by equation \eqref{eq:elx} we get

$$
\begin{align*}
    \frac{y'}{x'} &= \frac{dy}{dx} = \frac{-(x + h)}{y + k} \\
    \Rightarrow (y + k)^2 &= -(x + h)^2 + C
\end{align*}
$$

which is once again the equation of a circle.








