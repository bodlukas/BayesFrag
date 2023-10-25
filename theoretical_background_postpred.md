# Theoretical background on the posterior predictive distribution of IM values at new sites

The posterior is conditioned on damage data, station data and rupture characteristics, $p(\mathbf{im}_\mathcal{T}|\mathbf{ds}, \mathbf{im}_\mathcal{S}, \mathbf{rup})$. Because the target sites are not identical to the building survey sites, we obtain this posterior predictive by marginalizing over the posterior IM values at the survey sites, $\mathbf{im}_\mathcal{B}$.

$p(\mathbf{im}_\mathcal{T}|\mathbf{ds}, \mathbf{im}_\mathcal{S}, \mathbf{rup}) = \int p(\mathbf{im}_\mathcal{T}|\mathbf{im}_\mathcal{B}, \mathbf{im}_\mathcal{S}, \mathbf{rup}) p(\mathbf{im}_\mathcal{B}|\mathbf{ds}, \mathbf{im}_\mathcal{S}, \mathbf{rup}) \, \mathrm{d}\mathbf{im}_\mathcal{B}$

The distribution $p(\mathbf{im}_\mathcal{B}|\mathbf{ds}, \mathbf{im}_\mathcal{S}, \mathbf{rup})$ is the posterior of IM values at the survey sites from which we obtained samples $\mathbf{im}_\mathcal{B}$ with MCMC (see also the previous section). The distribution $p(\mathbf{im}_\mathcal{T}|\mathbf{im}_\mathcal{B}, \mathbf{im}_\mathcal{S}, \mathbf{rup})$ is a multivariate log-normal with parameters $\boldsymbol{\mu}_{\mathcal{T}|\mathcal{B,S}}$ and $\boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{B,S}}$ computed as 

$\boldsymbol{\mu}_{\mathcal{T}|\mathcal{B,S}} = \boldsymbol{\mu}_{\mathcal{T}|\mathcal{S}} + \boldsymbol{\Sigma}_{\mathcal{TB}|\mathcal{S}} \boldsymbol{\Sigma}_{\mathcal{BB}|\mathcal{S}}^{-1} (\ln \mathbf{im}_\mathcal{B} - \boldsymbol{\mu}_{\mathcal{B}|\mathcal{S}}) $

$\boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{B,S}} = \boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{S}} + \boldsymbol{\Sigma}_{\mathcal{TB}|\mathcal{S}} \boldsymbol{\Sigma}_{\mathcal{BB}|\mathcal{S}}^{-1} \boldsymbol{\Sigma}_{\mathcal{BT}|\mathcal{S}} $

The marginalization is approximated by sampling $\mathbf{im}_\mathcal{T}$ from $p(\mathbf{im}_\mathcal{T}|\mathbf{im}_\mathcal{B}, \mathbf{im}_\mathcal{S}, \mathbf{rup})$ with parameters computed for each posterior sample $\mathbf{im}_\mathcal{B}$ obtained from MCMC. 

The `PosteriorPredictiveIM` object performs this marginalization in a computationally efficient manner. 


## Implementation with whitening variables z

\begin{equation}
p()
\end{equation}

Step 1: $\mathbf{A} = \mathbf{L}_\mathcal{SS}^{-1} \boldsymbol{\Sigma}_{\mathcal{SB}}$

Step 2: Compute $\boldsymbol{\Sigma}_{\mathcal{TB}}$ and $\boldsymbol{\Sigma}_{\mathcal{ST}}$, 

Step 3: $\mathbf{B} = \mathbf{L}_\mathcal{SS}^{-1} \boldsymbol{\Sigma}_{\mathcal{ST}}$

Step 4: $\boldsymbol{\Sigma}_{\mathcal{TB}|\mathcal{S}} = \boldsymbol{\Sigma}_{\mathcal{BT}} - \mathbf{B}^\top \mathbf{A}$

Step 5: $\mathbf{C} = \mathbf{L}_{\mathcal{BB}|\mathcal{S}}^{-1} \boldsymbol{\Sigma}_{\mathcal{TB}|\mathcal{S}}^\top$

To obtain correlated samples from $p(\mathbf{im}_\mathcal{T}|\mathbf{ds}, \mathbf{im}_\mathcal{S}, \mathbf{rup})$ we compute the covariance matrix $\boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{B,S}} = \boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{S}} - \mathbf{C}^\top \mathbf{C}$

To obtain independent samples we compute the variance at each target site $j\in \mathcal{T}$: $\sigma_{\mathcal{T}_j|\mathcal{B,S}}^2 = \sigma_{\mathcal{T}_j|\mathcal{S}}^2 - \sum_{i\in\mathcal{B}} [\mathbf{C}]_{ij}^2$

Step 7: For each sample of $\mathbf{z}$: $\boldsymbol{\mu}_{\mathcal{T}|\mathcal{B,S}} = \boldsymbol{\mu}_{\mathcal{T}|\mathcal{S}} + \mathbf{C}^\top\mathbf{z} $

Step 8: Generate a sample from $\mathcal{N}(\boldsymbol{\mu}_{\mathcal{T}|\mathcal{B,S}}, \boldsymbol{\Sigma}_{\mathcal{TT}|\mathcal{B,S}})$
