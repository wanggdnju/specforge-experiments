"""Route sglang's EAGLE tree ops to the Triton implementations on this box.

Why this exists
---------------
sgl_kernel 0.4.7 here is a CUDA-13 build running on a CUDA-12.9 driver. Three
of its speculative kernels do not work on that driver, and they fail in two
different ways:

  tree_speculative_sampling_target_only  raises  "CUDA driver version is
                                                  insufficient for CUDA runtime
                                                  version"   (handled by the
                                                  existing SGLANG_COMPAT_HIDE_
                                                  SPEC_SAMPLING path)
  build_tree_kernel_efficient            SILENT no-op: returns normally and
  verify_tree_greedy                     writes nothing. positions keeps its
                                         uninitialised torch.empty contents,
                                         retrieve_index/retrieve_next_token
                                         keep their -1 fill.

The silent pair is what breaks EAGLE3. Serving any EAGLE3 checkpoint dies on
the first request inside run_eagle_verify, because eagle_prepare_for_verify
fills the verify batch with `positions` read out of uninitialised memory
(measured: min=-4.5e18), and the target attention then computes RoPE angles
from those and reads out of bounds. Nothing raises at the tree-build step, so
the traceback points at whatever kernel happens to be running when the context
is already poisoned.

sglang ships pure-Triton versions of both ops (kernels/ops/speculative/
spec_tree.py). Triton compiles for the *actual* device at runtime, so the
SDK-version mismatch that makes the prebuilt cubins inert does not apply. They
are already wired up for XPU; this module points the CUDA branch at them too.

Verified before trusting them, by calling both with known inputs and reading
the buffers back:

  sgl_build_tree_kernel_efficient_triton -> positions [6,7,8,9,10,11]
                                            retrieve_index [[0,1,2,3,4,5]]
                                            retrieve_next_token [[1,2,3,4,5,-1]]
  verify_tree_greedy_kernel_triton       -> writes predicts/accept_index, and
                                            accepts only the matching prefix
                                            (correct greedy-tree semantics)

What it does NOT do
-------------------
No file in the venv is modified. This is a runtime rebind of two module-level
names in sglang.srt.speculative.eagle_utils, installed from sitecustomize.
Only the greedy tree path (T=0) is affected -- the non-greedy tree sampler is
still unavailable on this driver, which is the pre-existing T=0-only limit.

Loaded via PYTHONPATH=<this dir>, env SGLANG_COMPAT_TRITON_TREE=1.
"""
import os


def _install():
    import sglang.srt.speculative.eagle_utils as eu

    orig_build = eu.build_tree_kernel_efficient
    orig_verify = eu.verify_tree_greedy_func

    if getattr(orig_build, "_sglang_compat_triton", False):
        return  # already installed

    def build_tree_kernel_efficient(*args, **kwargs):
        """Same signature as eagle_utils.build_tree_kernel_efficient, but the
        tree construction runs on the Triton kernel."""
        import torch

        # Mirror the original's buffer setup, then call the Triton builder
        # instead of the inert sgl_kernel op.
        (bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens,
         seq_lens_sum, topk, spec_steps, num_verify_tokens) = args[:9]
        tree_mask_mode = args[9] if len(args) > 9 else kwargs.get("tree_mask_mode", eu.TreeMaskMode.FULL_MASK)
        tree_mask_buf = args[10] if len(args) > 10 else kwargs.get("tree_mask_buf")
        # Any trailing kwargs are the original's tuning knobs (fill_prefix_mask);
        # the Triton builder has no equivalent and does not need them.

        return _build_like_original(
            eu, orig_build, torch, bonus_tokens, parent_list, top_scores_index,
            draft_tokens, seq_lens, seq_lens_sum, topk, spec_steps,
            num_verify_tokens, tree_mask_mode, tree_mask_buf,
        )

    def verify_tree_greedy_func(predicts, accept_index, accept_token_num,
                                candidates, retrieve_index, retrieve_next_token,
                                retrieve_next_sibling, target_predict, topk=-1):
        eu.verify_tree_greedy_triton(
            predicts, accept_index, accept_token_num, candidates,
            retrieve_index, retrieve_next_token, retrieve_next_sibling,
            target_predict,
        )
        return predicts, accept_index, accept_token_num

    build_tree_kernel_efficient._sglang_compat_triton = True
    verify_tree_greedy_func._sglang_compat_triton = True
    eu.build_tree_kernel_efficient = build_tree_kernel_efficient
    eu.verify_tree_greedy_func = verify_tree_greedy_func

    # Modules that did `from ... import build_tree_kernel_efficient` hold their
    # own binding; rebind those too.
    for modname in ("sglang.srt.speculative.eagle_worker_common",
                    "sglang.srt.speculative.frozen_kv_mtp_worker_v2"):
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        if hasattr(mod, "build_tree_kernel_efficient"):
            mod.build_tree_kernel_efficient = build_tree_kernel_efficient


def _build_like_original(eu, orig_build, torch, bonus_tokens, parent_list,
                         top_scores_index, draft_tokens, seq_lens, seq_lens_sum,
                         topk, spec_steps, num_verify_tokens, tree_mask_mode,
                         tree_mask_buf):
    """Reproduce the original's allocation, but fill it with the Triton kernel.

    Rather than reimplementing buffer sizing (which depends on tree_mask_mode
    and on whether a preallocated mask is in play), call the original once to
    get correctly shaped, correctly allocated buffers, then overwrite them with
    the Triton result. The original's sgl_kernel call is a no-op on this driver,
    so its returned buffers are exactly the empty shells we need to refill.
    """
    tree_mask, positions, retrieve_index, retrieve_next_token, \
        retrieve_next_sibling, out_tokens = orig_build(
            bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens,
            seq_lens_sum, topk, spec_steps, num_verify_tokens, tree_mask_mode,
            tree_mask_buf, True,
        )
    # The original concatenated the bonus token onto draft_tokens; rebuild the
    # same flattened order for the Triton call (it takes candidate ids by
    # index, and retrieve_index indexes into this flattened array).
    flat_draft = torch.cat(
        (bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    eu.sgl_build_tree_kernel_triton(
        parent_list, top_scores_index, seq_lens, tree_mask, positions,
        retrieve_index, retrieve_next_token, retrieve_next_sibling,
        topk, spec_steps, num_verify_tokens, tree_mask_mode,
    )
    return (tree_mask, positions, retrieve_index, retrieve_next_token,
            retrieve_next_sibling, flat_draft)


if os.environ.get("SGLANG_COMPAT_TRITON_TREE") == "1":
    try:
        _install()
    except Exception as _e:  # never break interpreter startup
        import sys
        print(f"[sglang_compat] triton tree install failed: {_e}",
              file=sys.stderr)
