"""
Reproduce, on any GPU, the initial noise that IndicF5 draws on a Colab/Kaggle T4.

Why: the only random step in IndicF5 inference is `torch.randn(duration, 100, device="cuda")` in
CFM.sample(). PyTorch's CUDA sampler (Philox4x32-10 + Box-Muller, grid-stride kernel) assigns
random sub-streams to elements according to the launch grid, and the grid is sized from the GPU's
SM count. So the same seed gives different noise on a T4 (40 SMs) and on, e.g., an RTX 4050
(20 SMs), and the kit's data/baseline_val/ (made on a T4) cannot be reproduced elsewhere.

This module re-implements that sampler in numpy for a chosen grid, verifies it against the local
GPU, and can patch f5_tts so that synthesis uses the T4's noise. Nothing about the model, the
seed or the settings changes; only the hardware dependence of the noise is removed.

    import t4_noise
    t4_noise.selftest()          # emulation == torch.randn on this GPU (bit-exact or ~1e-7)
    t4_noise.install("T4")       # from now on CFM.sample draws the noise a T4 would draw
    t4_noise.reset(seed)         # call right after torch.manual_seed(seed), before each item
"""
import numpy as np

# device profiles: (multiProcessorCount, maxThreadsPerMultiProcessor)
DEVICES = {"T4": (40, 1024)}

_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
_MASK = np.uint64(0xFFFFFFFF)
_INV32 = np.float32(2.3283064e-10)           # CURAND_2POW32_INV
_INV32_2PI = np.float32(2.3283064e-10 * 6.2831855)  # CURAND_2POW32_INV_2PI
BLOCK = 256        # block_size_bound in ATen DistributionTemplates.h
UNROLL = 4         # curand4_engine_calls for float normal


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """curand_Philox4x32_10 on uint64 arrays holding 32-bit values."""
    k0 = np.uint64(k0) & _MASK
    k1 = np.uint64(k1) & _MASK
    for r in range(10):
        p0 = _M0 * c0
        p1 = _M1 * c2
        hi0, lo0 = p0 >> np.uint64(32), p0 & _MASK
        hi1, lo1 = p1 >> np.uint64(32), p1 & _MASK
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _MASK, lo1, (hi0 ^ c3 ^ k1) & _MASK, lo0
        if r < 9:
            k0 = (k0 + _W0) & _MASK
            k1 = (k1 + _W1) & _MASK
    return c0, c1, c2, c3


def _box_muller(x, y):
    u = x.astype(np.float32) * _INV32 + np.float32(_INV32 / 2)
    v = y.astype(np.float32) * _INV32_2PI + np.float32(_INV32_2PI / 2)
    s = np.sqrt(np.float32(-2.0) * np.log(u))
    return (np.sin(v) * s).astype(np.float32), (np.cos(v) * s).astype(np.float32)


def grid_size(numel, sms, max_threads_per_sm):
    blocks = (numel + BLOCK - 1) // BLOCK
    return min(sms * (max_threads_per_sm // BLOCK), blocks)


def randn(numel, seed, offset, sms, max_threads_per_sm):
    """Return (values float32[numel], offset_increment) exactly as ATen's normal_ kernel."""
    g = grid_size(numel, sms, max_threads_per_sm)
    threads = g * BLOCK
    e = np.arange(numel, dtype=np.int64)
    it = e // (threads * UNROLL)                 # grid-stride iteration = curand4 call index
    rem = e % (threads * UNROLL)
    comp = rem // threads                        # which of the 4 outputs of curand_normal4
    idx = rem % threads                          # thread index = Philox subsequence
    # counter: x = offset/4 + call, y = 0 (no carry for our sizes), z = subsequence, w = 0
    c0 = (np.uint64(offset // 4) + it.astype(np.uint64)) & _MASK
    zeros = np.zeros(numel, dtype=np.uint64)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    r0, r1, r2, r3 = philox4x32_10(c0, zeros, idx.astype(np.uint64), zeros, seed & 0xFFFFFFFF, seed >> 32)
    s01, c01 = _box_muller(r0, r1)
    s23, c23 = _box_muller(r2, r3)
    out = np.choose(comp, [s01, c01, s23, c23]).astype(np.float32)
    counter_offset = ((numel - 1) // (BLOCK * g * UNROLL) + 1) * UNROLL
    inc = ((counter_offset + 3) // 4) * 4
    return out, inc


def randn_torch(numel, seed, offset, sms, max_threads_per_sm, device="cuda"):
    """The same sampler as randn(), computed with int64/float32 torch ops on `device` (used by the
    Task 4 engine: about 1 ms on the GPU instead of ~50 ms in numpy). Philox needs 32x32->64-bit
    products; int64 multiplication wraps modulo 2^64, so the bit pattern is the unsigned product and
    the high word is (p >> 32) & 0xFFFFFFFF."""
    import torch
    M = 0xFFFFFFFF
    g = grid_size(numel, sms, max_threads_per_sm)
    threads = g * BLOCK
    e = torch.arange(numel, device=device, dtype=torch.int64)
    it = e // (threads * UNROLL)
    rem = e % (threads * UNROLL)
    comp = rem // threads
    c0 = (it + offset // 4) & M
    c1 = torch.zeros_like(e)
    c2 = rem % threads
    c3 = torch.zeros_like(e)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    k0, k1 = seed & M, seed >> 32
    for r in range(10):
        p0 = c0 * int(_M0)
        p1 = c2 * int(_M1)
        c0, c1, c2, c3 = ((p1 >> 32) & M) ^ c1 ^ k0, p1 & M, ((p0 >> 32) & M) ^ c3 ^ k1, p0 & M
        if r < 9:
            k0 = (k0 + int(_W0)) & M
            k1 = (k1 + int(_W1)) & M

    def box_muller(x, y):
        u = x.to(torch.float32) * float(_INV32) + float(np.float32(_INV32 / 2))
        v = y.to(torch.float32) * float(_INV32_2PI) + float(np.float32(_INV32_2PI / 2))
        s = torch.sqrt(-2.0 * torch.log(u))
        return torch.sin(v) * s, torch.cos(v) * s

    s01, c01 = box_muller(c0, c1)
    s23, c23 = box_muller(c2, c3)
    out = torch.where(comp == 0, s01, torch.where(comp == 1, c01, torch.where(comp == 2, s23, c23)))
    counter_offset = ((numel - 1) // (BLOCK * g * UNROLL) + 1) * UNROLL
    inc = ((counter_offset + 3) // 4) * 4
    return out, inc


class _Gen:
    def __init__(self, sms, mt):
        self.sms, self.mt, self.seed, self.offset = sms, mt, 0, 0
        self.device = None          # None: numpy on the CPU; "cuda": randn_torch on the GPU

    def reset(self, seed):
        self.seed, self.offset = int(seed), 0

    def randn(self, *shape):
        numel = int(np.prod(shape))
        if self.device is not None:
            vals, inc = randn_torch(numel, self.seed, self.offset, self.sms, self.mt, self.device)
            self.offset += inc
            return vals.reshape(shape)
        vals, inc = randn(numel, self.seed, self.offset, self.sms, self.mt)
        self.offset += inc
        return vals.reshape(shape)


_GEN = None


def install(device="T4"):
    """Patch f5_tts.model.cfm so CFM.sample draws its noise from the emulated `device`."""
    global _GEN
    import torch
    import f5_tts.model.cfm as cfm
    sms, mt = DEVICES[device]
    _GEN = _Gen(sms, mt)
    real_torch = torch

    class _TorchProxy:
        def __getattr__(self, name):
            return getattr(real_torch, name)

        @staticmethod
        def randn(*size, device=None, dtype=None, **kw):
            if len(size) == 1 and isinstance(size[0], (tuple, list)):
                size = tuple(size[0])
            vals = _GEN.randn(*[int(s) for s in size])
            if isinstance(vals, np.ndarray):
                vals = real_torch.from_numpy(vals)
            return vals.to(device=device, dtype=dtype or real_torch.float32)

    cfm.torch = _TorchProxy()
    return _GEN


def reset(seed):
    if _GEN is not None:
        _GEN.reset(seed)


def selftest(shapes=((1000, 100), (517, 100), (3, 7))):
    """Compare the emulation with torch.randn on the local GPU (same seed, consecutive calls)."""
    import torch
    p = torch.cuda.get_device_properties(0)
    mt = getattr(p, "max_threads_per_multi_processor", None) or 1536
    torch.manual_seed(1234)
    gen = _Gen(p.multi_processor_count, mt)
    gen.reset(1234)
    worst = 0.0
    for shp in shapes:
        ref = torch.randn(*shp, device="cuda").cpu().numpy()
        emu = gen.randn(*shp)
        worst = max(worst, float(np.abs(ref - emu).max()))
    return {"gpu": p.name, "sms": p.multi_processor_count, "max_threads_per_sm": mt, "max_abs_diff": worst}


if __name__ == "__main__":
    print(selftest())
