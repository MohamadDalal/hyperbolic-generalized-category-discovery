# Writing a question to post on forums

Title: Hyperbolic geometry: Finding the exponential map for the Klein model using exponential maps from Lorentz and Poincare model

Hello. This is my first time posting a question, so I humbly ask that you go easy on me. I will start with first describing the background behind my questions:

I am trying to train a neural network with hyperbolic embeddings, the idea is to map the vector embeddings into a hyperbolic manifold before performing contrastive learning and classification. Here is an example of a paper that does contrastive learning in hyperbolic space https://proceedings.mlr.press/v202/desai23a.html, and I am taking a lot of inspiration from it.

Following the paper I am mapping to the Lorentz model, which is working fine for contrastive learning, but I also have to perform K-Means on the hyperbolic embedding vectors. For that I am trying to use the Einstein midpoint, which requires transforming to the Klein model and back.

I have followed the transformation from equation 9 in this paper https://ieeexplore.ieee.org/abstract/document/9658224:

$$x_K=\frac{x_{space}}{x_{time}}$$

However, the book assumes a constant curvature of -1, and I need the model to be able to work with variable curvature, as it is a learnable variable of the model. Would this transformation still work? If not does anyone have the formula for transforming from Lorentz to Klein model and back in arbitrary curvature?

