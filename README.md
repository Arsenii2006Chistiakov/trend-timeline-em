# Problem Definition

If you have a sample of videos posted under a trend, sound, or key word, you might want to estimate the **timeline of the trend**.

In principle, you could model this with some probability density function

$$
f(t; \theta)
$$

where `t` is time and `\theta` are the parameters. In this exploration I use a normal-family model, written informally as

$$
N(t; \theta)
$$

because I want a simple shape for how the trend evolves over time.

## Why MLE is not enough here

The problem is that you only observe the data **up to today**. You do **not** observe the full eventual volume of the trend, so the future mass is missing. Because of that, a direct MLE setup is not enough for the version of the problem I care about.

So instead, EM should be used. I decided to make this mini project public, because I was really proud to find the the application for the algorithm I learned before. I faced the challenge in relation to my main project Hugo, and got and intution that EM should be used. 

## Setup

The setup is simple.

Assume some distribution family. Here I use $N(t; \theta)$.

Define the observed variables:

$$
X_i = \text{number of observed posts by timestamp } t_i
$$

Create a latent variable $Z$:

$$
Z_i = \text{the unseen total volume of the whole sample as } t \to \infty
$$

This formulation of $Z$ is not canonical. I use it because this is my project and this is the solution which I derived myself.

Define the joint probability:

$$
P(X_i, Z_i \mid \theta) = \bigl(N_{\mathrm{cdf}}(t_i; \theta)\bigr)^{x_i}\bigl(1 - N_{\mathrm{cdf}}(t_i; \theta)\bigr)^{z_i - x_i}
$$

where `N_cdf` is the CDF of the chosen normal-family model.

Define the EM objective:

$$
Q(\theta) = \mathbb{E}_{Z \mid X, \theta^{(old)}} \left[ \sum_{i=1}^{N} \log P(X_i, Z_i \mid \theta) \right]
$$

Informally: take the expectation over the latent $Z$, then maximize with respect to the parameters.

In the maximization step, get the next set of parameters maximizing the log-likelihood.

Repeat until it converges.

## Intuition

The whole idea is: I only see the partial trend today, but I want to infer the full shape of the trend as if I could observe it all the way out. EM gives a clean way to treat that missing future volume as latent structure instead of pretending it is already observed.