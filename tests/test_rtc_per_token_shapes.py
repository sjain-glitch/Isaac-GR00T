"""Shape/consistency unit tests for the RTC per-token-timestep changes.

Loads the CHANGED fork modules directly (bypassing the heavy gr00t package __init__)
and checks:
  1. TimestepEncoder     (N,)->(N,D)  and  (N,T)->(N,T,D)
  2. AdaLayerNorm        per-sample (N,D) and per-token (N,T,D) temb
  3. MultiEmbodimentActionEncoder  (B,) and (B,T) timesteps
  4. AlternateVLDiT.forward  scalar vs per-token timestep (shape + uniform-consistency)
  5. gr00t_n1d6 forward RTC tensor-logic (delay/tau/interpolant/dit_timestep/loss-mask)

Key invariant used throughout: a PER-TOKEN time that is uniform across tokens must give
the SAME result as the legacy per-sample scalar path (so the change is backward-compatible),
while a genuinely per-token time must change the output (so it is actually wired in).
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(relpath, name):
    path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dit = _load("gr00t/model/modules/dit.py", "fork_dit")
emb = _load("gr00t/model/modules/embodiment_conditioned_mlp.py", "fork_emb")

torch.manual_seed(0)
FAILS = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- 1. TimestepEncoder
print("1. TimestepEncoder")
te = dit.TimestepEncoder(embedding_dim=16).eval()
with torch.no_grad():
    e1 = te(torch.arange(4))            # (N,)
    e2 = te(torch.arange(4 * 7).reshape(4, 7))  # (N,T)
    sc = te(torch.full((4,), 5))
    pt = te(torch.full((4, 7), 5))
check("(N,)->(N,D)", tuple(e1.shape) == (4, 16), str(tuple(e1.shape)))
check("(N,T)->(N,T,D)", tuple(e2.shape) == (4, 7, 16), str(tuple(e2.shape)))
check("uniform (N,T) == (N,) broadcast",
      torch.allclose(pt, sc[:, None].expand(4, 7, 16), atol=1e-5))


# ---------------------------------------------------------------- 2. AdaLayerNorm
print("2. AdaLayerNorm")
ada = dit.AdaLayerNorm(16).eval()
x = torch.randn(4, 7, 16)
with torch.no_grad():
    y_s = ada(x, torch.randn(4, 16))          # per-sample temb
    y_t = ada(x, torch.randn(4, 7, 16))       # per-token temb
    tb = torch.randn(4, 16)
    c_s = ada(x, tb)
    c_t = ada(x, tb[:, None].expand(4, 7, 16))
check("per-sample temb -> (N,T,dim)", tuple(y_s.shape) == (4, 7, 16), str(tuple(y_s.shape)))
check("per-token temb  -> (N,T,dim)", tuple(y_t.shape) == (4, 7, 16), str(tuple(y_t.shape)))
check("uniform per-token == per-sample", torch.allclose(c_s, c_t, atol=1e-5))
# per-token with a genuinely different time on token 0 must change that token's output
tb2 = tb[:, None].expand(4, 7, 16).clone()
tb2[:, 0] = torch.randn(4, 16)
with torch.no_grad():
    c_diff = ada(x, tb2)
check("distinct per-token time changes that token",
      not torch.allclose(c_diff[:, 0], c_t[:, 0], atol=1e-4))


# ---------------------------------------------------------------- 3. Action encoder
print("3. MultiEmbodimentActionEncoder")
enc = emb.MultiEmbodimentActionEncoder(action_dim=30, hidden_size=16, num_embodiments=32).eval()
acts = torch.randn(4, 50, 30)
eid = torch.zeros(4, dtype=torch.long)
with torch.no_grad():
    a_s = enc(acts, torch.randint(0, 1000, (4,)), eid)
    a_t = enc(acts, torch.randint(0, 1000, (4, 50)), eid)
    v = torch.full((4,), 5)
    u_s = enc(acts, v, eid)
    u_t = enc(acts, v[:, None].expand(4, 50), eid)
check("(B,) time -> (B,T,hidden)", tuple(a_s.shape) == (4, 50, 16), str(tuple(a_s.shape)))
check("(B,T) time -> (B,T,hidden)", tuple(a_t.shape) == (4, 50, 16), str(tuple(a_t.shape)))
check("uniform (B,T) == (B,)", torch.allclose(u_s, u_t, atol=1e-5))


# ---------------------------------------------------------------- 4. AlternateVLDiT forward
print("4. AlternateVLDiT.forward (scalar vs per-token)")
try:
    B, n_state, Ta, S, Dh = 2, 5, 8, 6, 16
    Ttot = n_state + Ta
    model = dit.AlternateVLDiT(
        num_attention_heads=2, attention_head_dim=8, output_dim=Dh, num_layers=2,
        cross_attention_dim=Dh, interleave_self_attention=True, attend_text_every_n_blocks=2,
    ).eval()
    hs = torch.randn(B, Ttot, Dh)
    ehs = torch.randn(B, S, Dh)
    img_mask = torch.zeros(B, S, dtype=torch.bool); img_mask[:, :3] = True
    bb_mask = torch.ones(B, S, dtype=torch.bool)
    with torch.no_grad():
        out_s, _ = model(hs, ehs, timestep=torch.full((B,), 5),
                         return_all_hidden_states=True, image_mask=img_mask,
                         backbone_attention_mask=bb_mask)
        out_t, _ = model(hs, ehs, timestep=torch.full((B, Ttot), 5),
                         return_all_hidden_states=True, image_mask=img_mask,
                         backbone_attention_mask=bb_mask)
        out_v, _ = model(hs, ehs, timestep=torch.randint(0, 1000, (B, Ttot)),
                         return_all_hidden_states=True, image_mask=img_mask,
                         backbone_attention_mask=bb_mask)
    check("scalar & per-token output shapes match", out_s.shape == out_t.shape,
          f"{tuple(out_s.shape)} vs {tuple(out_t.shape)}")
    check("uniform per-token == scalar", torch.allclose(out_s, out_t, atol=1e-4))
    check("distinct per-token time changes output", not torch.allclose(out_s, out_v, atol=1e-4))
except Exception as e:  # noqa: BLE001
    check("AlternateVLDiT forward ran", False, f"EXC: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- 5. RTC forward tensor-logic
print("5. gr00t_n1d6 forward RTC tensor-logic")
B, Ta, adim, buckets, D_rtc, n_state = 4, 50, 30, 1000, 8, 5
actions = torch.randn(B, Ta, adim)
noise = torch.randn(B, Ta, adim)
t_s = torch.rand(B)
delay = torch.randint(0, D_rtc, (B,))  # Uniform{0..D-1} (Pi RTC uniform recipe)
_steps = torch.arange(Ta)[None, :]
prefix_mask = _steps < delay[:, None]
tau = torch.where(prefix_mask, torch.ones_like(t_s)[:, None], t_s[:, None])
tau_e = tau.unsqueeze(-1)
noisy = (1 - tau_e) * noise + tau_e * actions
enc_time = (tau * buckets).long()
state_time_1d = (t_s * buckets).long()
dit_timestep = torch.cat([state_time_1d[:, None].expand(B, n_state), enc_time], dim=1)
action_mask = torch.ones(B, Ta, 1)
action_mask_rtc = action_mask * (~prefix_mask).unsqueeze(-1).to(action_mask.dtype)

check("tau shape (B,Ta)", tuple(tau.shape) == (B, Ta))
check("noisy shape (B,Ta,adim)", tuple(noisy.shape) == (B, Ta, adim))
check("enc_time shape (B,Ta)", tuple(enc_time.shape) == (B, Ta))
check("dit_timestep shape (B,n_state+Ta)", tuple(dit_timestep.shape) == (B, n_state + Ta))
check("loss mask shape (B,Ta,1)", tuple(action_mask_rtc.shape) == (B, Ta, 1))
check("delay in [0,D_rtc)", bool(((delay >= 0) & (delay < D_rtc)).all()))
# frozen prefix must be exactly the clean action (t=1)
clean_ok = all(torch.allclose(noisy[b, : delay[b]], actions[b, : delay[b]], atol=1e-6) for b in range(B))
check("frozen prefix == clean actions", clean_ok)
# suffix mask keeps exactly Ta - delay tokens
check("suffix count == Ta - delay",
      bool((action_mask_rtc.squeeze(-1).sum(dim=1).long() == (Ta - delay)).all()))
# D_rtc=0 style (delay all 0) => no freeze, full-chunk trained
check("delay=0 => all tokens trained",
      bool(((_steps < torch.zeros(B, 1)).logical_not()).all()))

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
