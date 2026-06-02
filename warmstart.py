"""Warm-start utilities: remap a DiffMamba Lightning checkpoint for MMDiMamba.

DiffMamba Lightning state keys look like 'backbone.model...' and
'backbone.sigma_map...'. MMDiMamba nests the same DiMamba under
`self.backbone`, so the MM key == original key (both start 'backbone.').
Drop noise schedule, EMA, optimizer, and any other non-backbone buffers.
"""
import torch


def remap_diffmamba_backbone_state(ckpt_state: dict) -> dict:
    """Keep only backbone weights; drop noise/EMA/optimizer keys.

    Args:
        ckpt_state: flat state_dict from a DiffMamba Lightning checkpoint.

    Returns:
        Filtered dict with only keys starting with 'backbone.' and no 'ema' keys.
    """
    out = {}
    for k, v in ckpt_state.items():
        if not k.startswith('backbone.'):
            continue
        if 'ema' in k:
            continue
        out[k] = v
    return out


def load_warmstart(mm_model, ckpt_path: str) -> dict:
    """Load DiffMamba backbone weights into an MMDiffusion/MMDiMamba module.

    Args:
        mm_model: the MMDiffusion or MMDiMamba instance to load into.
        ckpt_path: path to a DiffMamba Lightning .ckpt file.

    Returns:
        {'missing': n, 'unexpected': n} — projector/vision keys are expected
        in 'missing' because they are newly initialised.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    remapped = remap_diffmamba_backbone_state(state)
    missing, unexpected = mm_model.load_state_dict(remapped, strict=False)
    return {'missing': len(missing), 'unexpected': len(unexpected)}


def load_vlm_warmstart(mm_diffusion, ckpt_path: str) -> dict:
    """Warm-start an MMDiffusion from either checkpoint type.

    - DiffMamba text ckpt (keys 'backbone.sigma_map.*'/'backbone.model.*'):
      load the LM backbone into the nested MMDiMamba (mm_diffusion.backbone);
      the projector stays newly initialised. Used by the align phase.
    - MMDiffusion ckpt (keys 'backbone.backbone.*'/'backbone.projector.*'): load
      backbone + projector (+ noise) into the full module. Used by the SFT phase
      to pick up the align phase's trained projector.

    Returns {'mode', 'missing', 'unexpected'}; callers should assert
    unexpected == 0 (a non-zero count means a key-layout mismatch).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    is_mm = any(k.startswith('backbone.backbone.')
                or k.startswith('backbone.projector.') for k in state)
    if is_mm:
        sub = {k: v for k, v in state.items()
               if (k.startswith('backbone.') or k.startswith('noise.'))
               and 'ema' not in k}
        missing, unexpected = mm_diffusion.load_state_dict(sub, strict=False)
        return {'mode': 'mm', 'missing': len(missing),
                'unexpected': len(unexpected)}
    info = load_warmstart(mm_diffusion.backbone, ckpt_path)
    info['mode'] = 'diffmamba'
    return info
