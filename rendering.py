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


def render_rays(models,
                rays,
                N_samples=64,
                N_importance=0,
                use_disp=False,
                perturb=0,
                noise_std=1,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                variant="classic"
                ):

    def inference(model, xyz_, z_vals, rays_d=None, sun_d=None, weights_only=False, variant="classic"):
        """
        Helper function that performs model inference
        Args:
            model: NeRF model (coarse or fine)
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            rays_d: (N_rays, 3) direction vectors of the rays
            sun_d: (N_rays, 3) sun direction vectors associated to the rays
            weights_only: do inference on sigma only or not
            variant: NeRF variant that is used  (classic or s-nerf)
        Returns:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_rays = xyz_.shape[0]
        N_samples_ = xyz_.shape[1]
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)

        # check if the ray directions and the sun ray directions are given as input
        # these inputs may be used or not depending on the nerf variant
        if rays_d is not None:
            ray_dirs_ = torch.repeat_interleave(rays_d, repeats=N_samples_, dim=0)
        if sun_d is not None:
            sun_dirs_ = torch.repeat_interleave(sun_d, repeats=N_samples_, dim=0)

        # the input batch is split in chunks to avoid possible problems with memory usage
        batch_size = xyz_.shape[0]
        out_chunks = []

        # run model
        for i in range(0, batch_size, chunk):
            out_chunks += [model(xyz_[i:i+chunk],
                                 input_dir=ray_dirs_[i:i+chunk] if rays_d is not None else None,
                                 input_sun_dir=sun_dirs_[i:i+chunk] if sun_d is not None else None,
                                 sigma_only=weights_only)]
        out = torch.cat(out_chunks, 0)

        if weights_only:
            # predict only sigma
            sigmas = out.view(N_rays, N_samples_)
        else:
            # predict all
            channels = model.outputs_per_variant[variant]
            out = out.view(N_rays, N_samples_, channels)
            rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = out[..., 3]  # (N_rays, N_samples_)
            if variant == "s-nerf":
                sun_v = out[..., 4]  # (N_rays, N_samples_)
                sky_rgb = out[..., 5:8]  # (N_rays, N_samples_, 3)

        # define deltas, i.e. the length between the points in which the ray is discretized
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha as in the formula (3) of the nerf paper
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        transparency = torch.cumprod(alphas_shifted, -1)[:, :-1]  # T in the paper
        weights = alphas * transparency # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        if variant == "s-nerf":
            # equation 2 of the s-nerf paper
            white_source = torch.ones_like(rgbs)
            irradiance = white_source + (1 - sun_v.view(N_rays, N_samples_, 1)) * sky_rgb
            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs * irradiance, -2)  # (N_rays, 3)
            depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)
            return rgb_final, depth_final, weights, transparency, sun_v
        else:
            # classic NeRF outputs
            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
            depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

    # get rays
    rays_o, rays_d, near, far = rays[:, 0:3],  rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
    # keep rays direction only if it is part of the inputs of the model
    rays_d_ = rays_d if models[0].input_sizes[1] > 0 else None
    # sun rays direction is used by the s-nerf vairant
    sun_d_ = rays[:, 8:] if variant == "s-nerf" else None

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
    model_coarse = models[0]
    if test_time:
        weights_coarse = \
            inference(model_coarse, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d_, weights_only=True)
    else:
        if variant == "s-nerf":
            rgb_coarse, depth_coarse, weights_coarse, transparency_coarse, sun_visibility_coarse = \
                inference(model_coarse, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d_, weights_only=False)
            result = {'rgb_coarse': rgb_coarse,
                      'depth_coarse': depth_coarse,
                      'opacity_coarse': weights_coarse.sum(1),
                      'weights_coarse': weights_coarse,
                      'transparency_coarse': transparency_coarse,
                      'sun_visibility_coarse': sun_visibility_coarse}
        else:
            rgb_coarse, depth_coarse, weights_coarse = \
                inference(model_coarse, xyz_coarse, z_vals, rays_d=rays_d_, sun_d=sun_d_, weights_only=False)
            result = {'rgb_coarse': rgb_coarse,
                      'depth_coarse': depth_coarse,
                      'opacity_coarse': weights_coarse.sum(1)}

    # run fine model
    if N_importance > 0:

        # sample depths for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here
        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        # discretize rays for fine model
        xyz_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

        # run fine model
        model_fine = models[1]
        if variant == "s-nerf":
            rgb_fine, depth_fine, weights_fine, transparency_fine, sun_visibility_fine = \
                inference(model_fine, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d_, weights_only=False)
            result = {'rgb_fine': rgb_fine,
                      'depth_fine': depth_fine,
                      'opacity_fine': weights_fine.sum(1),
                      'weights_fine': weights_fine,
                      'transparency_fine': transparency_fine,
                      'sun_visibility_fine': sun_visibility_fine}
        else:
            rgb_fine, depth_fine, weights_fine = \
                inference(model_fine, xyz_fine, z_vals, rays_d=rays_d_, sun_d=sun_d_, weights_only=False)
            result = {'rgb_fine': rgb_fine,
                      'depth_fine': depth_fine,
                      'opacity_fine': weights_fine.sum(1)}

    return result