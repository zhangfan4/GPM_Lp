# Bisection Method for subproblem solving

This file tends to interpret how to use the bisection method to solve the subproblem:
$$
\begin{align}
\min_{\mathbf{x}}&\quad \frac12\|\mathbf{x}-\mathbf{y}\|_2^2\\
\mathrm{s.t.}&\quad \sum_{i=1}^nw_i^kx_i\leq \gamma^k\\
&\quad \mathbf{x}\in\mathbb{R}_+^n
\end{align}
$$
 where $\mathbf{y}\in\mathbb{R}_+^n$ is given.

## Case I: 

​	If $\sum_{i=1}^nw_i^ky_i\leq \gamma^k$, then the solution is given by
$$
\mathbf{x}^*=\mathbf{y}
$$


## Case II:

If $\sum_{i=1}^nw_i^ky_i> \gamma^k$, then the Lagrangian function is defined as
$$
L(\mathbf{x},\lambda)= \frac12\|\mathbf{x}-\mathbf{y}\|_2^2+\lambda\left(\sum_{i=1}^nw_i^kx_i- \gamma^k\right)
$$






optimality condition is defined as
$$
\begin{align}

\end{align}
$$
