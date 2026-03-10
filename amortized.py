
# Encode CH + Weather sequence 
class ObservationEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, y):
        # x: (N, x_dim), y: (N, y_dim)
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)  # (N, out_dim)
    
def aggregate(encoded_obs):
    # encoded_obs: (N, out_dim) — N observations
    return encoded_obs.mean(dim=0)  # (out_dim,)

# Encode observation with text prior 
class InferenceNetwork(nn.Module):
    def __init__(self, text_dim, obs_out_dim, embed_dim, hidden_dim):
        super().__init__()
        self.obs_encoder = ObservationEncoder(...)
        
        # Takes text embed + aggregated obs → posterior params
        self.combiner = nn.Sequential(
            nn.Linear(text_dim + obs_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 2)  # outputs μ and log σ
        )

    def forward(self, text_embed, observations=None):
        if observations is not None:
            x, y = observations
            encoded = self.obs_encoder(x, y)   # (N, out_dim)
            agg = encoded.mean(dim=0)          # (out_dim,)
        else:
            agg = torch.zeros(obs_out_dim)     # zero-shot fallback

        combined = torch.cat([text_embed, agg], dim=-1)
        out = self.combiner(combined)

        mu, log_sigma = out.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-4   # ensure positivity

        return mu, sigma

# Compute loss with KL divergence to stay "close" to the text embedding  
def loss(model, inference_net, site_data, text_embed):
    # Split observations
    context, target = random_split(site_data)

    # Get posterior embedding
    mu, sigma = inference_net(text_embed, context)

    # Reparameterization trick
    eps = torch.randn_like(mu)
    e_site = mu + sigma * eps

    # 1. Task loss: can we predict target observations?
    y_pred = model(target.x, site_embed=e_site, ...)
    task_loss = criterion(y_pred, target.y)

    # 2. KL toward text prior (acts as regularizer)
    prior_mu = text_projection(text_embed)
    prior_sigma = torch.ones_like(sigma)  # or also learned
    kl = kl_divergence(Normal(mu, sigma), Normal(prior_mu, prior_sigma))

    return task_loss + β * kl.mean()
```

The KL term does two things simultaneously: it regularizes the posterior toward the text prior, and it ensures the zero-shot case (`context = ∅`) doesn't degrade too far from the text embedding.

---

## Handling Zero to Many Observations Gracefully

A key property you want is **monotonic improvement** as you add more context:
```
0 observations  →  text prior (μ = f(text_embed))
1 observation   →  slight update toward data
N observations  →  converges toward fully data-driven embedding




# Predict NNL loss for CH
import torch
import torch.nn.functional as F

def gaussian_nll_loss(mu, log_var, y):
    # Predict log σ² for numerical stability (unconstrained output)
    var = torch.exp(log_var)
    loss = 0.5 * log_var + 0.5 * (y - mu)**2 / var
    return loss.mean()



class ColdHardinessModel(nn.Module):
    def forward(self, x):
        ...
        mu = self.mu_head(h)        # unconstrained
        log_var = self.var_head(h)  # unconstrained → exp() gives σ² > 0
        return mu, log_var
```

Never output `σ²` directly from a linear layer — it can go negative. Alternatives:

| Output | Transform | Notes |
|---|---|---|
| `log σ²` | `exp()` | Most common, numerically stable |
| `log σ` | `exp()` | Equivalent |
| unconstrained | `softplus() + ε` | Avoids very small variances |

---

## The Variance Collapse Problem

A known failure mode: the model ignores `μ` and just predicts very large `σ²` to make the loss small regardless of prediction quality. The NLL loss has a **natural tension** that prevents this:
```
L = ½ log σ²      ← penalizes large variance
  + (y-μ)²/2σ²   ← penalizes large residuals, scaled by 1/σ²


  log_var = torch.clamp(log_var, min=-10, max=10)

  def total_loss(model, inference_net, batch, text_embeds, β=1.0):
    # 1. Get amortized site embedding
    mu_e, sigma_e = inference_net(text_embeds[batch.site_id], batch.context_obs)
    
    eps = torch.randn_like(mu_e)
    e_site = mu_e + sigma_e * eps          # reparameterized sample

    # 2. Predictive loss
    mu_y, log_var_y = model(batch.x, site_embed=e_site)
    nll = 0.5 * log_var_y + 0.5 * (batch.y - mu_y)**2 / log_var_y.exp()

    # 3. KL from embedding posterior toward text prior
    prior_mu = text_projection(text_embeds[batch.site_id])
    kl = -0.5 * (1 + log_var_e - (mu_e - prior_mu)**2 - sigma_e**2)
    #          ↑ standard KL(N(μ,σ²) || N(prior_mu, 1))

    return nll.mean() + β * kl.mean()