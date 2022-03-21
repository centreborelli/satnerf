"""
This script renders the input rays that are used to feed the NeRF model
It discretizes each ray in the input batch into a set of 3d points at different depths of the scene
Then the nerf model takes these 3d points (and the ray direction, optionally, as in the original nerf)
and predicts a volume density at each location (sigma) and the color with which it appears
"""

import torch

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Args:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Returns:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models, args, rays, ts):

    # get config values
    N_samples = args.n_samples
    N_importance = args.n_importance
    variant = args.model
    use_disp = False
    perturb = False

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]

    # sample depths for coarse model
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    # discretize rays into a set of 3d points (N_rays, N_samples_, 3), one point for each depth of each ray
    xyz_coarse = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # run coarse model
    typ = "coarse"
    if variant == "s-nerf":
        from models.snerf import inference
        sun_d = rays[:, 8:11]
        # render using main set of rays
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_ = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d)
            result['weights_sc'] = result_["weights"]
            result['transparency_sc'] = result_["transparency"]
            result['sun_sc'] = result_["sun"]
    elif variant == "sat-nerf":
        from models.satnerf import inference
        sun_d = rays[:, 8:11]
        rays_t = models['t'](ts) if ts is not None else None
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
        if args.sc_lambda > 0:
            # solar correction
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result_tmp = inference(models[typ], args, xyz_coarse, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            result['weights_sc'] = result_tmp["weights"]
            result['transparency_sc'] = result_tmp["transparency"]
            result['sun_sc'] = result_tmp["sun"]
    else:
        # classic nerf
        from models.nerf import inference
        result = inference(models[typ], args, xyz_coarse, z_vals, rays_d=rays_d)
    result_ = {}
    for k in result.keys():
        result_[f"{k}_{typ}"] = result[k]

    # run fine model
    if N_importance > 0:

        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, result_['weights_coarse'][:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples+N_importance, 3)

        typ = "fine"
        if variant == "s-nerf":
            sun_d = rays[:, 8:11]
            # render using main set of rays
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=None)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]
        elif variant == "sat-nerf":
            sun_d = rays[:, 8:11]
            rays_t = models['t'](ts) if ts is not None else None
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
            if args.sc_lambda > 0:
                # solar correction
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result_ = inference(models[typ], args, xyz_fine, z_vals, rays_d=None, sun_d=sun_d, rays_t=rays_t)
                result['weights_sc'] = result_["weights"]
                result['transparency_sc'] = result_["transparency"]
                result['sun_sc'] = result_["sun"]
        else:
            result = inference(models[typ], args, xyz_fine, z_vals, rays_d=rays_d)
        for k in result.keys():
            result_["{}_{}".format(k, typ)] = result[k]

    return result_
