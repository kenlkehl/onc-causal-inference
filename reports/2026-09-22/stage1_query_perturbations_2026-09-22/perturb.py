"""Constrained training-only query edits, with equal-distance random controls."""
import numpy as np
import torch


def activation(chunks, mask, queries, temperature):
    score = (chunks @ queries.T) / temperature
    score = score.masked_fill(~mask[:, :, None], -torch.inf)
    return temperature * (torch.logsumexp(score, dim=1) - mask.sum(1).clamp(min=1).to(score.dtype).log()[:, None])


def moments(values, contribution, weights):
    centers = (values * weights[:, None]).sum(0) / weights.sum()
    rows = (values - centers) * contribution[:, None]
    return rows.mean(0), rows.std(0, unbiased=False)


def project_cap(candidate, original, radius):
    """Exact spherical cap for Euclidean distance between unit vectors."""
    candidate = torch.nn.functional.normalize(candidate, dim=1)
    cosine = (candidate * original).sum(1, keepdim=True).clamp(-1, 1)
    bound = 1 - radius[:, None].square() / 2
    tangent = candidate - cosine * original
    tangent = torch.nn.functional.normalize(tangent, dim=1)
    boundary = bound * original + torch.sqrt((1 - bound.square()).clamp(min=0)) * tangent
    return torch.where(cosine < bound, boundary, candidate)


def random_at_distance(query, distance, seed):
    rng = np.random.default_rng(seed)
    q = np.asarray(query, dtype=np.float64)
    q = q / np.linalg.norm(q)
    tangent = rng.normal(size=q.shape)
    tangent -= (tangent @ q) * q
    tangent /= np.linalg.norm(tangent)
    cosine = 1 - float(distance) ** 2 / 2
    return (cosine * q + np.sqrt(max(0, 1 - cosine ** 2)) * tangent).astype(np.float32)


def optimize(chunks, mask, originals, contribution, weights, policy, seed, progress=None):
    """No validation inputs accepted. Minimize a fixed-scale cohort moment."""
    torch.manual_seed(seed)
    q0 = torch.nn.functional.normalize(torch.as_tensor(originals, dtype=torch.float32), dim=1)
    c = torch.as_tensor(contribution, dtype=torch.float32)
    w = torch.as_tensor(weights, dtype=torch.float32)
    with torch.no_grad():
        base_a = activation(chunks, mask, q0, policy["temperature"])
        _, base_scale = moments(base_a, c, w)
        base_sd = base_a.std(0, unbiased=False)
    indexes = [(j, r, restart) for j in range(len(q0)) for r in policy["radii"] for restart in range(policy["restarts"])]
    source = torch.stack([q0[j] for j, _, _ in indexes])
    radii = torch.tensor([r for _, r, _ in indexes])
    scales = torch.stack([base_scale[j].clamp(min=1e-8) for j, _, _ in indexes])
    sds = torch.stack([base_sd[j].clamp(min=1e-8) for j, _, _ in indexes])
    initial = np.stack([q0[j].numpy() if restart == 0 else random_at_distance(q0[j].numpy(), r * 0.1, seed + k)
                        for k, (j, r, restart) in enumerate(indexes)])
    candidate = torch.nn.Parameter(torch.as_tensor(initial))
    optimizer = torch.optim.Adam([candidate], lr=policy["learning_rate"])
    best = source.clone()
    best_loss = torch.full((len(indexes),), float("inf"))
    lo, hi = policy["relative_sd_bounds"]
    history = []

    def objective(q):
        values = activation(chunks, mask, q, policy["temperature"])
        mean, _ = moments(values, c, w)
        ratio = values.std(0, unbiased=False) / sds
        distance = torch.linalg.vector_norm(q - source, dim=1)
        criterion = (mean / scales).square() + policy["distance_penalty"] * (distance / radii).square()
        penalty = 10 * (torch.relu(lo - ratio).square() + torch.relu(ratio - hi).square())
        return criterion, penalty, ratio

    with torch.no_grad():
        best_loss = objective(source)[0]
    for epoch in range(policy["epochs"] + 1):
        optimizer.zero_grad()
        q = torch.nn.functional.normalize(candidate, dim=1)
        criterion, penalty, ratio = objective(q)
        with torch.no_grad():
            eligible = (ratio >= lo) & (ratio <= hi)
            better = eligible & (criterion < best_loss)
            best[better] = q[better]
            best_loss[better] = criterion[better]
            if epoch % 10 == 0:
                history.append({"epoch": epoch, "best_mean_objective": float(best_loss.mean())})
                if progress:
                    progress(history[-1])
        if epoch == policy["epochs"]:
            break
        (criterion + penalty).mean().backward()
        torch.nn.utils.clip_grad_norm_([candidate], 5.0)
        optimizer.step()
        with torch.no_grad():
            candidate.copy_(project_cap(candidate, source, radii))
    selected = []
    for j in range(len(q0)):
        for radius in policy["radii"]:
            choices = [k for k, (jj, rr, _) in enumerate(indexes) if jj == j and rr == radius]
            winner = min(choices, key=lambda k: (float(best_loss[k]), indexes[k][2]))
            selected.append({"query": j, "radius": radius, "restart": indexes[winner][2],
                             "criterion": float(best_loss[winner]), "vector": best[winner].numpy(),
                             "distance": float(torch.linalg.vector_norm(best[winner] - q0[j]))})
    return selected, history
