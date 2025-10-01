# Flow Matching on CIFAR-10
Please follow the instructions for this part of the assignment in THIS order!
First, download the pre-trained checkpoint from https://drive.google.com/drive/folders/1hypaGqB69SxL7U5ZTDSrwU-bRZBaXE9R?usp=drive_link

Flow matching learns a time-conditioned velocity field that continuously transports a base Gaussian into the data distribution. This section mirrors the diffusion homework setup but switches to the flow-matching formulation described in Lipman et al. (2022) and Liu et al. (2022).

## Mathematical Background

### Linear Interpolation Path
We prescribe a linear interpolant between Gaussian noise $z \sim \mathcal{N}(0, I)$ and a clean image $x_0$:

$$x_t = (1 - t) z + t \cdot x_0, \quad t \in [0,1]$$

This creates **straight-line paths** in image space: at $t=0$ we have pure noise, at $t=1$ we have the data, and intermediate times linearly blend the two.

### Ground-Truth Velocity Field
The velocity along this path is simply the time derivative:

$$u_t = \frac{\mathrm{d} x_t}{\mathrm{d} t} = x_0 - z$$

**Key insight**: The velocity $u_t = x_0 - z$ is **constant** (independent of $t$) along each trajectory. It points directly from the noise sample to the data sample. This is much simpler than diffusion models, where the target noise $\epsilon_t$ depends on the time-dependent schedule $\bar{\alpha}_t$.

### Why Straight Lines? (Optimal Transport)
This construction coincides with the **Benamou-Brenier formulation** of optimal transport. We seek a time-dependent velocity field that transports the base Gaussian distribution $\mathcal{N}(0,I)$ to the data distribution while minimizing kinetic energy. The solution is the **displacement interpolation** (straight lines in data space), which are geodesics in the 2-Wasserstein metric space. By learning to match these optimal velocities, the network learns to follow the most efficient transport paths.

### Conditional Flow Matching Loss
The UNet receives $(x_t, t)$ and predicts a velocity field $v_\theta(x_t, t)$ of the same shape as the image. We optimize the **conditional flow matching (CFM)** loss:

$$
\mathcal{L}_{\mathrm{CFM}}
= \mathbb{E}_{x_0 \sim \mathcal{D}, z \sim \mathcal{N}(0, I), t \sim \mathcal{U}(0,1)}
\left[ || v_{\theta}(x_t, t) - (x_0 - z) ||_2^2 \right]
$$



This is called *conditional* flow matching because we supervise the network on individual trajectories conditioned on each $(x_0, z)$ pair. The network learns velocities that work for all such conditional paths, allowing it to generalize to the entire probability flow.

### Sampling via ODE Integration
Once trained, sampling integrates the learned ODE **forward in time** from noise to data:

$$\frac{\mathrm{d} x}{\mathrm{d} t} = v_\theta(x, t), \quad x(0) \sim \mathcal{N}(0, I), \quad t: 0 \to 1$$

We use an explicit integrator for this assignment:
- **Euler**: $x_{k+1} = x_k + \Delta t \cdot v_\theta(x_k, t_k)$

### 4.1 Euler Inference (# TODO 4.1)
- Start from Gaussian noise $x(0) \sim \mathcal{N}(0, I)$; our sampler integrates the learned velocity field $v_\theta(x, t)$ forward from $t = 0$ to $t = 1$.
- **Forward pass**: `FlowModel.forward(x_t, t)` should convert floating times to integer indices via `_prepare_t` before calling the UNet.
- **Euler micro-step**: `FlowModel.ode_euler_step(x, t, dt)` evaluates $v_\theta(x, t)$ with the forward pass and applies the explicit Euler update $x \leftarrow x + dt \cdot v$.
- **Time grid**: In both `sample` and `sample_given_z`, build a uniform grid `ts` and compute the spacing `dt`. The provided loop calls `ode_euler_step`, and iterates to the final time.

Run inference using the following command:
```
python inference.py --ckpt fm_final.pth --solver euler --steps 50
```
This saves a 10x10 grid to `flow_euler/samples_euler.png`.

### 4.2 FID Evaluation (# TODO 4.2)
Finish the fid function in `inference.py`. Compute FID for flow matching inference using Euler solver using the following command:

```
python inference.py --ckpt fm_final.pth --solver euler --steps 50 --compute-fid
```

## References
- Lipman et al., "Flow Matching for Generative Modeling," NeurIPS 2022.
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," 2022.