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


def inference(model, conf, rays_xyz, z_vals, rays_d=None, sun_d=None, rays_t=None):
    """
    Runs the nerf model using a batch of input rays
    Args:
        model: NeRF model (coarse or fine)
        conf: the NeRF configuration
        rays_xyz: (N_rays, N_samples_, 3) sampled positions in the object space
                  N_samples is the number of sampled points in each ray;
                            = conf.n_samples for coarse model
                            = conf.n_samples+conf.n_importance for fine model
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        rays_d: (N_rays, 3) direction vectors of the rays
        sun_d: (N_rays, 3) sun direction vectors
    Returns:
        result: dictionary with the output magnitudes of interest
    """
    N_rays = rays_xyz.shape[0]
    N_samples = rays_xyz.shape[1]
    xyz_ = rays_xyz.view(-1, 3)  # (N_rays*N_samples, 3)

    # check if there are additional inputs, which are used or not depending on the nerf variant
    rays_d_ = None if rays_d is None else torch.repeat_interleave(rays_d, repeats=N_samples, dim=0)
    sun_d_ = None if sun_d is None else torch.repeat_interleave(sun_d, repeats=N_samples, dim=0)
    rays_t_ = None if rays_t is None else torch.repeat_interleave(rays_t, repeats=N_samples, dim=0)

    # the input batch is split in chunks to avoid possible problems with memory usage
    chunk = conf.training.chunk
    variant = conf.name
    batch_size = xyz_.shape[0]

    # run model
    out_chunks = []
    for i in range(0, batch_size, chunk):
        out_chunks += [model(xyz_[i:i+chunk],
                             input_t=None if rays_t is None else rays_t_[i:i + chunk],
                             input_dir=None if rays_d_ is None else rays_d_[i:i+chunk],
                             input_sun_dir=None if sun_d_ is None else sun_d_[i:i+chunk])]
    out = torch.cat(out_chunks, 0)

    # retreive outputs
    out_channels = model.outputs_per_variant[variant]
    out = out.view(N_rays, N_samples, out_channels)
    rgbs = out[..., :3]  # (N_rays, N_samples, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples)
    if variant == "s-nerf":
        sun_v = out[..., 4:5]  # (N_rays, N_samples, 1)
        sky_rgb = out[..., 5:8]  # (N_rays, N_samples, 3)
    if variant == "s-nerf-w":
        sun_v = out[..., 4:5] # (N_rays, N_samples, 1)
        ambient_a = out[..., 5:8] # (N_rays, N_samples, 3)
        ambient_b = out[..., 8:11] # (N_rays, N_samples, 3)

    # define deltas, i.e. the length between the points in which the ray is discretized
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples)

    # compute alpha as in the formula (3) of the nerf paper
    noise_std = conf.training.noise_std
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples)
    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
    weights = alphas * transparency # (N_rays, N_samples)
    weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # return outputs
    if variant == "s-nerf":
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
        # equation 2 of the s-nerf paper
        irradiance = sun_v + (1 - sun_v) * sky_rgb
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs * irradiance, -2)  # (N_rays, 3)
        rgb_final = torch.clamp(rgb_final, min=0., max=1.)
        result = {'rgb': rgb_final,
                  'depth': depth_final,
                  'weights': weights,
                  'transparency': transparency,
                  'albedo': rgbs,
                  'sun': sun_v,
                  'sky': sky_rgb}
    elif variant == "s-nerf-w":
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
        rgb_final = torch.sum(weights.unsqueeze(-1) * (rgbs * sun_v * ambient_a + ambient_b), -2) # (N_rays, 3)
        rgb_final = torch.clamp(rgb_final, min=0., max=1.)
        result = {'rgb': rgb_final,
                  'depth': depth_final,
                  'weights': weights,
                  'transparency': transparency,
                  'albedo': rgbs,
                  'sun': sun_v,
                  'ambient_a': ambient_a,
                  'ambient_b': ambient_b}
    else:
        # classic NeRF outputs
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        result = {'rgb': rgb_final,
                  'depth': depth_final,
                  'weights': weights,
                  'transparency': transparency}
    return result


def render_rays(models,
                conf,
                rays,
                ts):

    # get config values
    N_samples = conf.n_samples
    N_importance = conf.n_importance
    use_disp = conf.training.use_disp
    perturb = conf.training.perturb
    variant = conf.name

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
    # keep rays direction only if it is part of the inputs of the model
    rays_d_ = rays_d if conf.input_sizes[1] > 0 else None

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
    result_ = {}
    if variant == "s-nerf":
        sun_d = rays[:, 8:11]
        # render using main set of rays
        result = inference(models[typ], conf, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d)
        if conf.lambda_s > 0:
            # predict transparency/sun visibility from a secondary set of solar correction rays
            xyz_coarse = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
            result = inference(models[typ], conf, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d)
            result_[f'weights_sc_{typ}'] = result["weights"]
            result_[f'transparency_sc_{typ}'] = result["transparency"]
            result_[f'sun_sc_{typ}'] = result["sun"]
    elif variant == "s-nerf-w":
        sun_d = rays[:, 8:11]
        rays_t_ = models['t'](ts) if ts is not None else None
        result = inference(models[typ], conf, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d, rays_t=rays_t_)
    else:
        result = inference(models[typ], conf, xyz_coarse, z_vals, rays_d=rays_d_)
    result_ = {}
    for k in result.keys():
        result_["{}_{}".format(k, typ)] = result[k]

    # run fine model
    if N_importance > 0:

        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples+N_importance, 3)

        typ = "fine"
        result_ = {}
        if variant == "s-nerf":
            sun_d = rays[:, 8:11]
            # render using main set of rays
            result = inference(models[typ], conf, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d)
            if conf.lambda_s > 0:
                # predict transparency/sun visibility from a secondary set of solar correction rays
                xyz_fine = rays_o.unsqueeze(1) + sun_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
                result = inference(models[typ], conf, xyz_fine, z_vals, rays_d=sun_d_, sun_d=sun_d)
                result_[f'weights_sc_{typ}'] = result["weights"]
                result_[f'transparency_sc_{typ}'] = result["transparency"]
                result_[f'sun_sc_{typ}'] = result["sun"]
        elif variant == "s-nerf-w":
            sun_d = rays[:, 8:11]
            rays_t_ = models['t'](ts) if ts is not None else None
            result = inference(models[typ], conf, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d, rays_t=rays_t_)
        else:
            result = inference(models[typ], conf, xyz_fine, z_vals, rays_d=rays_d_, sun_d=None)
        for k in result.keys():
            result_["{}_{}".format(k, typ)] = result[k]

    return result_
