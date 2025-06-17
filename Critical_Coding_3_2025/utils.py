import numpy as np
import torch
from torch.nn.functional import interpolate
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def slerp(val, low, high):
    '''
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    '''
    if len(low.shape) == 1:
        omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
        so = np.sin(omega)
        return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high
    elif len(low.shape) == 2:
        ws = []
        for i in range(low.shape[0]):
            omega = np.arccos(np.dot(low[i,:]/np.linalg.norm(low[i,:]), high[i,:]/np.linalg.norm(high[i,:])))
            so = np.sin(omega)
            w = np.sin((1.0-val)*omega) / so * low[i,:] + np.sin(val*omega)/so * high[i,:]
            ws.append(w)
        return torch.tensor(np.array(ws))


def get_perceptual_loss(synth_image, target_features, perceptual_model):
    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    synth_image = (synth_image + 1) * (255/2)
    if synth_image.shape[2] > 256:
        synth_image = interpolate(synth_image, size=(256, 256), mode='area')

    # Features for synth images.
    synth_features = perceptual_model(synth_image, resize_images=False, return_lpips=True)
    return (target_features - synth_features).square().sum()

def get_target_features(target, perceptual_model, device):
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = interpolate(target_images, size=(256, 256), mode='area')
    return perceptual_model(target_images, resize_images=False, return_lpips=True)


def run_projector(projection_target, g_model, steps, perceptual_model, device, save_path = None):
  zs = torch.randn([10000, g_model.mapping.z_dim], device=device)
  w_stds = g_model.mapping(zs, None).std(0)

  target_features = get_target_features(projection_target, perceptual_model, device)

  with torch.no_grad():
    qs = []
    losses = []
    for _ in range(8):
      q = (g_model.mapping(torch.randn([4,g_model.mapping.z_dim], device=device), None, truncation_psi=0.75) - g_model.mapping.w_avg) / w_stds
      images = g_model.synthesis(q * w_stds + g_model.mapping.w_avg)
      loss = get_perceptual_loss(images, target_features, perceptual_model)
      i = torch.argmin(loss)
      qs.append(q[i])
      losses.append(loss)
    qs = torch.stack(qs)
    losses = torch.stack(losses)
    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0).requires_grad_()

  # Sampling loop
  q_ema = q
  opt = torch.optim.AdamW([q], lr=0.20, betas=(0.0,0.999))
  target_images = projection_target.unsqueeze(0).to(device).to(torch.float32)/255*2-1

  for i in range(steps):
    opt.zero_grad()
    w = q * w_stds
    image = g_model.synthesis(w + g_model.mapping.w_avg, noise_mode='const')
    # in the first quarter of the steps, use MSE loss, then switch to perceptual loss
    if i > (steps / 4):
      loss = get_perceptual_loss(image, target_features, perceptual_model)
      if i % 10 == 0:
        print(f"image {i}/{steps} | perceptual_loss: {loss.item()}")
    else:
      loss = torch.nn.functional.mse_loss(image, target_images)
      if i % 10 == 0:
        print(f"image {i}/{steps} | mse_loss: {loss}")

    loss.backward()
    opt.step()

    q_ema = q_ema * 0.9 + q * 0.1
    image = g_model.synthesis(q_ema * w_stds + g_model.mapping.w_avg, noise_mode='const')

    if save_path is not None:
        pil_image = to_pil_image(image[0].add(1).div(2).clamp(0,1))
        pil_image.save(f'{save_path}/{i:04}.jpg')

  return q_ema * w_stds + g_model.mapping.w_avg

def image_path_to_tensor(target_image_filename, model_resolution):
    target_pil = Image.open(target_image_filename).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((model_resolution, model_resolution), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_tensor = torch.tensor(target_uint8.transpose([2, 0, 1]))
    return target_tensor




def slerp_interpolation(z1, z2, num_interp):
    # create intervals
    interp_vals = np.linspace(1./num_interp, 1, num=num_interp)

    # Convert latent vectors to numpy arrays
    latent_a_np = z1.cpu().numpy().squeeze()
    latent_b_np = z2.cpu().numpy().squeeze()

    # Create our spherical interpolation between two points
    latent_interp = torch.tensor(np.array([slerp(v, latent_a_np, latent_b_np) for v in interp_vals], dtype=np.float32))
    return latent_interp

def batch_norm(x, min=None, max=None):
    min = x.min() if min is None else min
    max = x.max() if max is None else max
    return (x - min) / (max - min)
