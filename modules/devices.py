import contextlib

import torch

from backend import memory_management

cpu: torch.device = torch.device("cpu")
fp8: bool = False
device: torch.device = memory_management.get_torch_device()
device_gfpgan = device_esrgan = device_codeformer = device
dtype_vae: torch.dtype = memory_management.vae_dtype()
dtype_unet: torch.dtype = memory_management.unet_dtype()
dtype_inference: torch.dtype = memory_management.inference_cast(dtype_unet, device)
dtype: torch.dtype = torch.float32 if dtype_unet is torch.float32 else torch.float16
unet_needs_upcast: bool = False


def has_xpu() -> bool:
    return memory_management.is_device_xpu(device)


def has_mps() -> bool:
    return memory_management.is_device_mps(device)


def get_cuda_device_id() -> int:
    return device.index


def get_cuda_device_string() -> str:
    return str(device)


def get_optimal_device_name() -> str:
    return device.type


def get_optimal_device() -> torch.device:
    return device


def get_device_for(*args, **kwargs) -> torch.device:
    return device


def torch_gc():
    memory_management.soft_empty_cache()


def autocast(*args, **kwargs):
    return contextlib.nullcontext()


def without_autocast(*args, **kwargs):
    return contextlib.nullcontext()


class NansException(Exception):
    pass


def report_nonfinite_denoised(step: int, total_steps: int):
    """
    Explain a sampling step whose output is not finite, and what to do about it.

    Left alone this surfaces as a black image and nothing else: the NaN is
    carried through the remaining steps, the VAE decodes it to NaN, and the
    pipeline turns that into zeros. Which step it first appears on separates two
    quite different faults, so it is worth saying.
    """

    when = f"step {step} of {total_steps}"
    lines = [f"The diffusion model produced NaN at sampling {when}; the image cannot recover from here."]

    if step == 0:
        lines.append("That is the very first step, so nothing accumulated into it -- the weights are already")
        lines.append("wrong by the time they are used, which points at the checkpoint, its quantisation, or")
        lines.append("the precision it is being run in rather than at the sampler.")
    else:
        lines.append("Earlier steps were finite, so values grew until they left the range of the compute dtype")
        lines.append("(fp16 stops at 65504).")

    lines.append("Worth trying, in this order:")
    lines.append("  * --bf16-unet             run the model in its native bf16 -- slower, and the dependable fix")
    lines.append("  * a GGUF or INT8 build of the same checkpoint")
    lines.append("  * --force-upcast-attention  if it only shows up at larger sizes")

    memory_management.logger.warning("\n".join(lines))


def test_for_nans(x: torch.Tensor, *args, **kwargs):
    if torch.isnan(x).any():
        memory_management.logger.warning("Encountered NaN in Latent" + ("; Try --disable-sage" if memory_management.sage_enabled() else ""))
        x.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)
        # raise NansException
