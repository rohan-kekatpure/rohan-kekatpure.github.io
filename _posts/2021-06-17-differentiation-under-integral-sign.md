---
layout: post
title: "Deriving the differentiation under the integral sign"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

_Surely you're joking, Mr Feynman!_
[brought](https://en.wikipedia.org/wiki/Leibniz_integral_rule#In_popular_culture) differentiation under the integral
sign (DUI) into from relative obscurity into folklore. Most of us get introduced to DUI in the context of
[Evaluation of tricky definite integrals](https://en.wikipedia.org/wiki/Leibniz_integral_rule#Applications). But
outside of this magical application, DUI is an important theoretical tool in the theory
[integral transforms](https://en.wikipedia.org/wiki/Integral_transform). Integral transforms themselves are central
to Physics and Mathematics. Characteristic function in probability theory, the Fourier
and Laplace transforms in signal processing, the Convolution operation, the Mellin transform in Number theory are
all special cases of integral transforms. DUI can be applied to each of them to derive important results.

In the present post we will derive the DUI rule for simple one-dimensional case. The DUI, also known as the Leibniz
Integration Rule is a significantly deeper result than what we will derive here. The
[Wikipedia](https://en.wikipedia.org/wiki/Leibniz_integral_rule) article states:

>The general statement of the Leibniz integral rule requires concepts from differential geometry, specifically
differential forms, exterior derivatives, wedge products and interior products.

My mathematical level permits me to merely scratch the surface of this remarkable result. The Wikipedia article
linked above is an excellent place to start exploring more.

## Derivation of one-dimensional DUI rule

Consider a general definite integral of the form

$$
\begin{equation}
g(x) = \int_{a(x)}^{b(x)}f(t, x) dt
\end{equation}
$$

Notice that we have recognized the integral as being a function of $$x$$. This is because, the dependence with
respect to $$t$$ will be integrated out once the limits of the integration are substituted. The DUI rule states that

<div class="boxed">
$$
\begin{equation}
g'(x) = f(b(x), x)\,b'(x) - f(a(x), x)\,a'(x) - \int_{a(x)}^{b(x)}f_x(t, x) dt
\label{eq:result}
\end{equation}
$$
</div>

To derive this result, we will use first principles definition of derivative of a function:

$$
\begin{equation}
g'(x) = \lim_{\epsilon\to 0} \frac{g(x + \epsilon) - g(x)}{\epsilon}
\end{equation}
$$

Let us evaluate $$g(x + \epsilon)$$. We will need to make use of the first order expansions of various functions. The
general rule for first order expansion of one- and two-variable functions is:

$$
\begin{align}
u(x+ \epsilon) &= u(x) + \epsilon u'(x) + O(\epsilon^2) \\
u(t+ \delta, x) &= u(t, x) + \delta u_t(t, x) + O(\delta^2) \label{eq:exp2} \\
u(t, x + \epsilon) &= u(t, x) + \epsilon u_x(t, x) + O(\epsilon^2) \label{eq:exp3}
\end{align}
$$

Now lets us compute $$g(x+\epsilon)$$ by making use of the first-order expansion rule on $$a(x)$$, $$b(x)$$ and
$$f(t, x)$$:

$$
\begin{align}
g(x + \epsilon) &= \int_{a(x + \epsilon)}^{b(x + \epsilon)}f(t, x + \epsilon) dt \notag \\[0.1in]
&= \int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)}\left[f(t, x) + \epsilon f_x(t, x) \right]dt \notag \\[0.1in]
&= \int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)}f(t, x) dt
 + \epsilon \int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)} f_x(t, x) dt \label{eq:1} \\[0.1in]
\end{align}
$$

Now we use the following identity on limits of definite integrals:

$$
\int_{p+r}^{q+s} = \int_p^q -\int_p^{p+r} + \int_q^{q + s}
$$

The identity is read intuitively as follows: the area under any curve from $$p+r$$ to $$q + s$$ is area from
$$p$$ to $$q$$ minus area from $$p$$ to $$p+r$$ plus additional area from $$q$$ to $$q+s$$. Using this identity we can
simplify the first term in equation $$\eqref{eq:1}$$ as:

$$
\begin{align}
\int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)}f(t, x) dt
&= \int_{a(x)}^{b(x)}f(t,x)dt - \int_{a(x)}^{a(x)+\epsilon a'(x)} f(t, x) dt
+ \int_{b(x)}^{b(x)+\epsilon b'(x)} f(t, x) dt \notag \\[0.1in]
&= g(x) - \bigg[h(t, x)\bigg]_{a(x)}^{a(x) + \epsilon a'(x)}
+\bigg[h(t, x)\bigg]_{b(x)}^{b(x) + \epsilon b'(x)} \notag \\[0.1in]
&= g(x) - \big[h((a(x) + \epsilon a'(x), x) - h(a(x), x)\big]
+ \big[h((b(x) + \epsilon b'(x), x) - h(b(x), x)\big] \notag \\[0.1in]
&= g(x) - \epsilon h_t(a(x), x) a'(x) + \epsilon h_t(b(x), x) b'(x)
\quad \text{ using equation }\eqref{eq:exp2}\notag \\[0.1in]
&= g(x) - \epsilon f(a(x), x) a'(x) + \epsilon f(b(x), x) b'(x)
\quad \text{ since } h_t(t,x) = f(t, x)
\end{align}
$$

Similarly the second term in the equation $$\eqref{eq:1}$$ can be simplified as

$$
\begin{align}
\int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)} f_x(t, x) dt
&=\int_{a(x)}^{b(x)} f_x(t, x) dt - \int_{a(x)}^{a(x) + \epsilon a'(x)} f_x(t, x) dt
+ \int_{b(x)}^{b(x) + \epsilon b'(x)} f_x(t, x) dt \notag \\[0.1in]
&= \int_{a(x)}^{b(x)} f_x(t, x) dt - \big[q(a(x) + \epsilon a'(x), x) - q(a(x), x)\big]
+ \big[q(b(x) + \epsilon b'(x), x) - q(b(x), x)\big] \notag \\[0.1in]
&= \int_{a(x)}^{b(x)} f_x(t, x) dt - \epsilon q_t(a(x), x) a'(x)
+ \epsilon q_t(b(x), x) b'(x)
\quad \text{ using equation }\eqref{eq:exp2}\notag \\[0.1in]
&= \int_{a(x)}^{b(x)} f_x(t, x) dt - \epsilon f_x(a(x), x) a'(x)
+ \epsilon f_x(b(x), x) b'(x)
\quad \text{ since } q_t(t, x) = f_x(t, x)
\end{align}
$$


Thus,

$$
\begin{align}
g(x+\epsilon) &= \int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)}f(t, x) dt
 + \epsilon \int_{a(x) + \epsilon a'(x)}^{b(x) + \epsilon b'(x)} f_x(t, x) dt \notag \\[0.1in]
&= g(x) - \epsilon f(a(x), x) a'(x) + \epsilon f(b(x), x) b'(x) \notag \\[0.1in]
&+\epsilon\int_{a(x)}^{b(x)} f_x(t, x) dt - \epsilon^2 f_x(a(x), x) a'(x)
+ \epsilon^2 f_x(b(x), x) b'(x) \notag \\[0.2in]
\frac{g(x + \epsilon) - g(x)}{\epsilon} &=
f(b(x), x)\, b'(x) - f(a(x), x)\, a'(x) + \int_{a(x)}^{b(x)} f_x(t, x) dt + \epsilon O(1)
\label{eq:2}
\end{align}
$$

The result $$\eqref{eq:result}$$ follows upon taking limit of $$\eqref{eq:2}$$ as $$\epsilon\to 0$$.

The derivation of the single-variable DUI rule is cumbersome. However, the exercise in helpful in showing us
how to manipulate integrals where limits are a function of another variable. Also note that, throughout the derivation,
we used nothing more than definition of differentiation. At the end, we have a powerful rule and the insight for why
it works.

## Applications

Let us now use the rule to derive a few interesting facts.



