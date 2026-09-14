# Running on AMD Radeon

This fork runs on AMD GPUs as well as NVIDIA ones. The compute backend is
detected automatically at launch; nothing has to be configured by hand for the
common cases.

---

## Quick start (Windows)

Reference configuration this was set up for:

| | |
|---|---|
| CPU | AMD Ryzen 5 3600 (6C/12T) |
| RAM | 64 GB DDR4 |
| GPU | AMD Radeon RX 6800 (16 GB, `gfx1030`, RDNA 2) |
| OS | Windows 10 x64 |

1. Install **64-bit Python 3.13** from [python.org](https://www.python.org/downloads/).
   Tick **"Add python.exe to PATH"** during setup.
   *(3.12 also works. 3.11 and older do not — the AMD wheels are `cp312`/`cp313` only.)*
2. Install [Git for Windows](https://git-scm.com/download/win).
3. Update the **AMD Adrenalin** driver to a current release.
4. Double-click **`webui-user-amd.bat`**.

The first launch downloads a ROCm build of PyTorch (several GB) and creates the
`venv` folder. Later launches skip straight to the UI.

There is nothing else to install — the ROCm runtime ships inside the Python
wheels. You do **not** need the HIP SDK unless you are using the ZLUDA fallback.

---

## Without the `.bat` launcher (conda / venv)

`webui-user-amd.bat` only sets `COMMANDLINE_ARGS` and calls `webui.bat`, which
manages a `venv` for you. If you would rather bring your own environment,
activate it and run `launch.py` directly:

```bash
conda create -n imgen python=3.13
conda activate imgen
python launch.py --gpu-backend rocm --pin-shared-memory
```

**Do not `pip install -r requirements.txt` first.** `requirements.txt` lists
`torch` unpinned, so pip resolves it from PyPI — and the PyPI Windows wheels are
**CPU-only**. (`facexlib` pulls in a CPU `torchvision` the same way.) The result
looks like a complete install but has no GPU support:

```
RuntimeError: PyTorch (ROCm) cannot see your Radeon GPU.
```

`launch.py` installs the correct PyTorch build itself, before the requirements,
and it inspects the *build* rather than just checking that torch is importable.
So if you already have CPU wheels in your environment, just run `launch.py` —
it detects the mismatch and replaces them:

```
Replacing PyTorch: torch is a 'cpu' build, but the 'rocm' backend needs 'rocm*'
```

The same check runs after the requirements step, so an extension that reinstalls
PyTorch behind your back gets reported instead of silently disabling your GPU.

---

## The backends

`--gpu-backend` selects one explicitly; the default (`auto`) picks the first
that applies.

| Backend | Hardware | PyTorch build | Notes |
|---|---|---|---|
| `cuda` | NVIDIA | `cu130` | unchanged from upstream |
| `rocm` | AMD | `+rocm` | **recommended for Radeon**; native, no extra install |
| `zluda` | AMD, Windows | `cu118` | fallback; needs the AMD HIP SDK |
| `directml` | any DX12 GPU | `torch-directml` | last resort, see caveats below |
| `cpu` | — | `+cpu` | very slow; for testing only |

### `rocm` — native ROCm (recommended)

On Windows this uses AMD's ["TheRock"](https://github.com/ROCm/TheRock) wheels,
which are published per GPU family. Your card's family is detected from the HIP
runtime that ships with the Adrenalin driver:

| Family | Cards |
|---|---|
| `gfx103X-dgpu` | RX 6800 / 6800 XT / 6900 XT / 6950 XT / 6700 / 6600 / 6500 |
| `gfx110X-all` | RX 7900 / 7800 / 7700 / 7600 |
| `gfx120X-all` | RX 9070 / 9060 |
| `gfx1151` | Ryzen AI Max (Strix Halo) |

On Linux the first-party wheels from `download.pytorch.org/whl/rocm7.1` are used
instead.

If detection gets it wrong, force it:

```bat
set SD_AMD_ARCH=gfx1030
```

or point at a different wheel index entirely:

```bat
set ROCM_INDEX_URL=https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/
```

### `zluda` — CUDA translation

For Radeon cards AMD does not publish ROCm wheels for. ZLUDA implements the
CUDA driver API on top of HIP, so the normal `cu118` PyTorch works.

1. Install the [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
   — **version 6.2 or 6.4**. ZLUDA does not support HIP SDK 7.x.
2. Launch with `--gpu-backend zluda`.

ZLUDA itself is downloaded automatically into `.zluda` on first use.

Add `--zluda-nightly` to get the nightly build, which enables hipBLASLt and
MIOpen (faster, but less stable). `--reinstall-zluda` wipes and re-downloads it.

The first generation after each launch is slow — ZLUDA compiles kernels on the
fly and caches them.

### `directml`

`torch-directml` pins PyTorch to 2.4.1 and Python ≤ 3.12, which is at the very
bottom of what this codebase supports. It is much slower than ROCm, does not
support `bf16` or `fp8` weights, and several features are disabled under it.
Use it only if neither `rocm` nor `zluda` works on your machine.

---

## Tuning for a 16 GB card with plenty of RAM

`webui-user-amd.bat` ships with:

```bat
set COMMANDLINE_ARGS=--gpu-backend rocm --pin-shared-memory
```

`--pin-shared-memory` page-locks system RAM (up to 45% of it on Windows) so that
weights offloaded out of VRAM stream back quickly. With 64 GB installed this is
close to free, and it is what makes large models usable on 16 GB of VRAM.

Other options worth knowing:

| Flag | Effect |
|---|---|
| `--cuda-stream 2` | overlaps weight transfers with compute |
| `--reserve-vram 1.0` | leaves 1 GB of VRAM for the desktop |
| `--lowvram` / `--novram` | more aggressive offloading |
| `--use-pytorch-cross-attention` | SDPA instead of the sliced fallback |
| `--bf16-unet` | keep bf16 weights (see below) |
| `--fp16-unet` | force fp16 weights |
| `--disable-gpu-warning` | silence low-VRAM warnings |

### Precision on RDNA 2

RDNA 2 has no bf16 matrix hardware, so weights are computed in **fp16** by
default even for models that ship as bf16 (Flux, Krea 2, Qwen-Image, Wan). This
is the right trade on speed — bf16 on `gfx1030` is emulated and several times
slower — but fp16 tops out at 65504, and these models produce attention scores
well past that. The overflow becomes `inf`, the softmax becomes `NaN`, and the
pipeline zeroes it: a black image.

So on a card with no bf16, attention scores and the softmax are computed in fp32
by default. The rest of the model stays fp16, and the cost is small next to
running everything in bf16. You will see:

```
Upcasting attention to fp32: this GPU has no bf16, and fp16 attention overflows on bf16-native models
```

`--no-upcast-attention` turns it off if you would rather have the speed.

If a model still comes out black or corrupted, `--bf16-unet` runs it in its
native bf16 throughout — slow, but numerically faithful.

### Attention

`gfx1030` and older have no FlashAttention kernels (AOTriton, which provides
them on ROCm, starts at `gfx90a`/`gfx1100`). The default is the sliced
`attention_basic` implementation, which adapts its slice size to free VRAM and
therefore degrades gracefully at high resolutions.

`--use-pytorch-cross-attention` switches to PyTorch SDPA, pinned to the math
backend. It is faster at ordinary resolutions but allocates an O(N²) attention
matrix, so it can run out of memory during hires-fix on large images.

### Large models (Krea 2, Flux, Qwen-Image, Wan)

These do not fit in 16 GB at full precision, and are handled by keeping the
weights in a compact format and casting them per-layer as they are used:

* **GGUF** checkpoints — fully supported, and the best option on 16 GB.
* **fp8 (`e4m3fn` / `e5m2`)** checkpoints — the weights stay fp8 in memory and
  are cast to fp16 for the matmul. RDNA 2 has no fp8 hardware, so `--fast-fp8`
  has no effect and is ignored.
* **INT8** checkpoints (`...Int8.safetensors`, and INT8 Krea 2 in particular) —
  supported, and the memory saving is real, but the matmul is not accelerated
  on RDNA 2. hipBLASLt ships no INT8 GEMM kernel for `gfx103X`, and calling the
  one PyTorch exposes crashes the process outright, so startup detects that and
  substitutes an fp32 matmul:

  ```
  INT8 matmul: using the fp32 fallback (this GPU's ROCm libraries have no INT8 GEMM kernel)
  ```

  Expect roughly fp16 speed with int8 memory use. A GGUF or fp8 build of the
  same model will generally be faster.
* Anything that still does not fit is offloaded to system RAM, which is what
  `--pin-shared-memory` accelerates.

---

## Flags that do nothing on AMD

These depend on CUDA-only libraries. Passing them prints a message and
continues, rather than failing the install:

`--xformers` · `--sage` · `--flash` · `--nunchaku` · `--onnxruntime-gpu` ·
`--cuda-malloc` · `--fast-fp8` (on RDNA 2)

INT8 checkpoints load and run, but without an INT8 speedup — see above.

MIOpen is disabled by default (`ENABLE_MIOPEN=1` re-enables it); see
Troubleshooting for why.

---

## Troubleshooting

**"PyTorch (ROCm) cannot see your Radeon GPU"**
Update the Adrenalin driver, then delete `venv` and relaunch. If your card is
older than RDNA 2, try `--gpu-backend zluda`.

**"AMD does not publish Windows ROCm PyTorch wheels for your GPU architecture"**
Your card has no ROCm wheel family. Use `--gpu-backend zluda`.

**"ZLUDA needs the AMD HIP SDK, which was not found"**
Install HIP SDK 6.2 or 6.4, or set `HIP_PATH` to an existing install.

**"ZLUDA could not run a trivial matmul on your GPU"**
Usually a HIP SDK version mismatch (7.x is not supported), or a card whose
`rocBLAS` libraries are missing from the SDK.

**"RuntimeError: operator torchvision::nms does not exist"**, or a
**"python.exe - Entry Point Not Found"** dialog naming `torchvision\_C.pyd`
torchvision's compiled extension failed to load, almost always because it was
built against a different PyTorch than the one installed. torchvision hides that
load error and only fails later, on the first missing operator, which is why the
traceback points at `_meta_registrations.py`.

The launcher now checks for this on startup and reinstalls a matching build by
itself. If you hit it in an environment you manage by hand, install the
torchvision that pairs with your torch — AMD's nightly channel publishes several
under the same date stamp, and the newest is *not* the matching one:

| torch | torchvision |
|---|---|
| 2.9 | 0.24.0 |
| 2.10 | 0.25.0 |
| 2.11 | 0.26.0 |
| 2.12 | 0.27.0a0 |

```bat
set TORCH_VERSION=2.9.1+rocm7.13.0a20260421
set TORCHVISION_VERSION=0.24.0+rocm7.13.0a20260421
```

then relaunch with `--reinstall-torch`. To see the underlying load error, set
`TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS=1`.

**The warning about a mismatched PyTorch build**
Something replaced PyTorch — usually an extension's `install.py`, or a manual
`pip install -r requirements.txt`. Relaunch with `--reinstall-torch`.

**A 0xC0000005 crash dump mentioning `_int_mm` or `torch_hip.dll`**
An INT8 checkpoint reached an INT8 matmul kernel that does not exist for your
GPU. This is detected automatically on the first launch after the venv is built;
if you are seeing it, delete `tmp\int-mm-*.ok` and relaunch so the check runs
again.

**A black image**
The log names the stage that went NaN and what to try — the diffusion model and
the VAE are separate problems with separate fixes:

```
NaN in the output: the diffusion model produced NaN, ... Worth trying:
  * --bf16-unet          run the model in its native bf16 (slower, numerically faithful)
  ...
```

**"The GPU stopped responding and its context was lost"**
A kernel faulted or the driver reset the card. Nothing works afterwards until
the Web UI is restarted — everything printed after that message is fallout, not
new information.

Two causes on this hardware:

* **MIOpen convolutions.** MIOpen benchmarks candidate kernels at runtime, and
  on `gfx1030` a candidate can take the GPU down with it, usually during VAE
  decode and usually preceded by `CK grouped conv library not found`. MIOpen is
  therefore off by default here and PyTorch's own convolutions are used instead
  — sometimes slower, but they come back. `ENABLE_MIOPEN=1` turns it back on.
* **TDR** (Windows' driver timeout). Windows resets a GPU that spends more than
  two seconds inside one kernel. Generating at a lower resolution or with tiled
  VAE decoding keeps individual kernels short; failing that, raise the timeout:

  ```
  reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 20 /f
  ```

  Reboot afterwards. This is a global Windows setting — it delays the safety net
  that recovers a genuinely hung GPU, so raise it, don't disable it.

**Out of memory during hires-fix**
Drop `--use-pytorch-cross-attention` if you added it, add `--reserve-vram 1.0`,
or switch to a GGUF/fp8 checkpoint.

---

## Reporting problems

`--dump-sysinfo` writes a report that includes the detected backend, GPU
architecture and PyTorch build. Attach it to any issue.
