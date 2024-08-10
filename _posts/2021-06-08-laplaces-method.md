---
layout: post
title: "Laplace's method"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

Laplace's method is a general technique for approximating functions of the form $g(x) = e^{M f(x)}$ around
global maximum of $f(x)$. It may seem like we're studying a rare special case of functions, but functions of
this particular form appear in combinatorics and probability theory (and related areas in physics which make use
of probability, such as quantum electrodynamics and quantum statistics). After understanding Laplace's method, we
will be able to derive asymptotic forms of common probability distributions and easily prove special cases of the
[Central Limit Theorem](https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation).

The intuition behind Laplace's method is easy to understand. Expand $f(x)$ in a Taylor series around $x_0$:

$$
\begin{equation}
f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2} (x-x_0)^2
\label{eq:texp1}
\end{equation}
$$

We now use the fact that $f(x)$ has a global maximum at $x_0$. This implies two things: (1) $f'(x_0) = 0$ and (2)
$$f''(x_0) < 0$$. These facts allow us to refine equation $\eqref{eq:texp1}$ as

$$
\begin{equation}
f(x) \approx f(x_0) - \frac{\vert f''(x_0) \vert}{2} (x-x_0)^2
\end{equation}
$$

Function $g(x) = e^{Mf(x)}$ becomes

$$
\begin{equation}
g(x) = e^{M f(x)} \approx e^{Mf(x_0)} \,\,\,e^{-\frac{M\vert f''(x_0) \vert}{2} (x-x_0)^2}
\label{eq:texp2}
\end{equation}
$$

Once $$g(x)$$ is brought in this form, the next steps depend on the application. One of the most common operations is
to estimate the integral of $$g(x)$$ within some limits $$[a, b]$$. If $$a < x < b$$ then the integral is
approximated by the Gaussian integral

$$
\begin{equation}
\int_a^b g(x)dx = \int_a^b e^{Mf(x_0)} \, e^{-\frac{M\vert f''(x_0) \vert}{2} (x-x_0)^2} dx
\approx \sqrt{\frac{2\pi}{M\vert f''(x_0)\vert}} e^{Mf(x_0)}
\end{equation}
$$

Let us restate and box this remarkable formula

<div class="boxed">
$$
\begin{equation}
\int_a^b e^{Mf(x)} dx \approx \sqrt{\frac{2\pi}{M\vert f''(x_0)\vert}} e^{Mf(x_0)}
\label{eq:lm}
\end{equation}
$$
</div>

If I'd encountered this formula out of the blue, it would have seemed like magic. My thinking would go like this: We
started with  generic exponential function with a global maximum. And suddenly we got an _exact_ formula for its
integral and that formula involves square root of $$\pi$$! How is that possible? But now we know the
straightforward reasoning behind its derivation. By the way, $$\sqrt{2\pi}$$ is always a hint that a Gaussian is
somewhere nearby.

As a final note, we will see how Laplace's method lets us derive some pretty cool results. If you're familiar with
the Gamma function, you probably know that it is a generalization of the factorial:

$$
\begin{equation}
n! = \Gamma(n + 1) = \int_0^\infty x^n e^{-x} dx
\end{equation}
$$

The integrand $$g(x) = x^n e^{-x}$$ can be written as $$g(x) = e^{f(x)}$$ where $$f(x) = n\log x - x$$. Using
equation $$\eqref{eq:lm}$$ we can immediately approximate $$\Gamma(n + 1) = \sqrt{2\pi n}\,(n/e)^n$$, a result which is
famous as the Stirling's approximation. The
[Stirling's approximation post]({% post_url 2021-06-01-stirlings-approximation %}) has more details about applying
Laplace's method to the Gamma function.
