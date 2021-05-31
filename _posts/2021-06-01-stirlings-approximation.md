---
layout: post
title: "Factorial and Stirling's approximation"
author: "Rohan"
categories: journal
tags: [documentation,sample]
image:
---

## Factorial function

Factorial of a number $$n$$ is the result of multiplication of numbers from $$1$$ through $$n$$. "$$n$$ factorial" is
 denoted by $$n!$$ and in mathematical notation we write:

$$
\begin{equation}
n! = 1 \times 2 \times \cdots \times n
\end{equation}
$$

The factorial function shows in discussions about arranging a set of items. The number of ways of arranging $$n$$
distinct objects equals $$n!$$. Since arranging things is universal, it is not surprising
that the factorial function appears in computational complexity, combinatorics (arrangements of abstract objects), and
statistical physics (arrangements of atoms).

While the mathematical definition of factorial is easy enough to understand, its numerical computation is a different
matter. 32- and 64-bit unsigned integers will overflow for $$13!$$ and $$21!$$ respectively. Most scientific
calculators will return floating point answers up to $$69!$$. But in real world, we need to estimate
factorials of numbers much larger than these. In statistical physics, for example, one needs to [accurately estimate](https://en.wikipedia.org/wiki/Fermi%E2%80%93Dirac_statistics#Microcanonical_ensemble)
factorial of $$\sim 6\times 10^{23}$$ of atoms. How do we go about estimating large factorials? The answer is a special
function called the Gamma function.

## Gamma function: Generalization of the factorial

[Gamma function](https://en.wikipedia.org/wiki/Gamma_function) is defined as

$$
\begin{equation}
\Gamma(z + 1) = \int_0^\infty x^z e^{-x}dx
\label{eq:gamma}
\end{equation}
$$

In the above expression, note that the answer of the above integral must be a function of $$z$$ since the $$x$$ will
integrate out. The surprising property of the above function is that and for any positive integer $$n$$,
$$\Gamma(n + 1)$$ equals $$n!$$.

$$
\begin{equation}
\Gamma(n + 1) = n!
\end{equation}
$$

If you're familiar with integration by parts (from high school or freshman calculus),
you can demonstrate this fact quite easily.

From the form of gamma function, it may seem that we have made an easy problem unnecessarily complicated. But the
opposite is true. Once we generalize an integer-valued problem to real numbers, the latter can be attacked with the
full power of calculus. Tools from calculus and analysis allows us to perform not only approximation analysis, but to
discover surprising connections to other special functions or entirely different mathematical areas.

In number theory this game is played quite often. Famous example are

 1. [Newton's generalization](https://en.wikipedia.org/wiki/Binomial_theorem#Newton's_generalized_binomial_theorem) of
 the binomial theorem to negative integers, rational numbers and real numbers

 2. [Riemann's generalization](https://en.wikipedia.org/wiki/Riemann_zeta_function) of the zeta function to
 the complex plane

In fact the fascinating field of
[analytic number theory](https://en.wikipedia.org/wiki/Analytic_number_theory)
deals with applying methods of analysis (i.e. calculus) to number theory.

Coming back to the factorial, we note that the Gamma function provides a generalization of the factorial and
furthermore allows us to use tools from calculus to estimate it accurately. When we carry out the steps, it will turn
out that the factorial of any positive integer $$n$$ can be accurately approximated by the following function.

$$
\begin{equation}
n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n
\end{equation}
$$

and the approximation gets better as n gets bigger. The above approximation to the factorial is known as
the [Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation). It is this approximation
that makes various derivations in statistical physics and combinatorics tractable.

Lets see next how to derive this remarkable formula.

## Stirling's approximation

There are a number of ways to derive this formula, but we will use one which I find most intuitive. Consider the
integrand in equation $$\eqref{eq:gamma}$$

$$
\begin{equation}
A_z(x) = x^z e^{-x}
\end{equation}
$$

$$A_z(x)$$ is a combination of two factors $$x^z$$ and $$e^{-x}$$. For $$z > 0$$, the first factor ($$x^z$$) increases
with $$x$$ and the second factor ($$e^{-x}$$) decreases with $$x$$. Thus we should expect $$A(x)$$ to have a maximum
at some value of $$x$$. When we plot the function $$A_z(x)$$ for various values of $$z$$ we find a maximum as
expected. But we also find a lot more. Not  only does $$A_z(x)$$ have a maximum, but for larger and larger values of
$$z$$, $$A_z(x)$$ increasingly resembles the Gaussian $$\mathcal{N}(z, z)$$ (i.e a Gaussian distribution centered at
$$\mu=z$$ and having variance $$\sigma^2=z$$).

<figure>
    <img src="{{site.url}}/assets/img/gamma_evolution.png" alt='map' style='margin: 10px;' height="300"/>
    <figcaption></figcaption>
</figure>

### Gamma function to Gaussian

Let us now prove informally that the integrand of the gamma function resembles a scaled Gaussian function. To do that
we first define

$$
\begin{equation}
B_z(x) = \log A_z(x) = z \log x - x
\end{equation}
$$

We will now obtain a polynomial expansion for $$B_z(x)$$ around $$x = z$$ using Taylor expansion

$$
\begin{align}
\begin{split}
 B_z'(x)\bigg\rvert_{x = z} &= \frac{z}{x} - 1\bigg\rvert_{x = z} = 0 \\
B_z''(x)\bigg\rvert_{x = z} &= \frac{-z}{x^2}\bigg\rvert_{x = z} =\frac{-1}{z}
\end{split}
\label{eq:t2}
\end{align}
$$

Thus our Taylor expansion is, up to the quadratic term

$$
\begin{align}
\begin{split}
B_z(x) &\simeq B_z(z) + B_z'(z)(x - z) + \frac{B_z''(z)}{2}(x-z)^2 \\
&= z \log z - z + 0 - \frac{(x - z)^2}{2z}
\end{split}
\end{align}
$$

This means our integrand of the Gamma function $$A_z(x)$$ is

$$
\begin{align}
A_z(x) &= e^{B_z(x)} \notag \\
&\simeq e^{z\log z - z} \,\, e^{-\frac{(x - z)^2}{2 z}} \notag \\
&= \left(\frac{z}{e}\right)^z \,\, e^{-\frac{(x - z)^2}{2 z}} \label{eq:gaussian}
\end{align}
$$

The second factor in equation $\eqref{eq:gaussian}$ is precisely a scaled Gaussian function with mean and variance
both equal to $$z$$.

The final step is integrating the integrand $$A_z(x)$$ to obtain an approximation for $$\Gamma(z + 1)$$:

$$
\begin{align}
\begin{split}
\Gamma(z + 1) &= \int_0^\infty A_z(x) dx \\
&= \left(\frac{z}{e}\right)^z \int_0^\infty e^{-\frac{(x - z)^2}{2 z}} dx \\
&= \left(\frac{z}{e}\right)^z \int_{-z}^\infty e^{-\frac{h^2}{2 z}} dh \qquad\ldots\text{putting  } h = x - z \\
&\simeq \left(\frac{z}{e}\right)^z \int_{-\infty}^{\infty} e^{-\frac{h^2}{2 z}} dh \\
&= \sqrt{2\pi z} \left(\frac{z}{e}\right)^z
\end{split}
\end{align}
$$

This completes our informal proof of the Stirling's approximation.




