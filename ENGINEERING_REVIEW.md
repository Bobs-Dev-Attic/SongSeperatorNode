# Engineering Review (April 27, 2026)

This review focuses on memory usage, security posture, operational robustness, and optimization opportunities for the current `SongSeparator` ComfyUI node implementation.

## Key Findings

1. **Phase distortion bug in high-pass filter implementation (fixed).**
   The code comment described a zero-phase filter, but the implementation used `sosfilt` (single-pass causal filter), which introduces phase shift.

2. **Memory pressure from unnecessary stem materialization (fixed).**
   Demucs returns all stems, but downstream logic only needs guitar + selected `keep_*` stems. The previous code converted every stem to NumPy arrays, increasing CPU RAM and copy overhead.

3. **Potential invalid filter configuration edge case (fixed).**
   If cutoff is outside `(0, Nyquist)`, SciPy filtering fails. The new implementation exits safely by returning the original signal.

4. **Mixing implementation created avoidable temporary arrays (fixed).**
   `np.sum(arrays, axis=0)` can allocate larger temporaries depending on array stack behavior. Incremental accumulation reduces temporary memory churn.

5. **Missing guard when model does not expose expected `guitar` stem (fixed).**
   Access pattern could throw a raw `KeyError`; now emits a clearer runtime error message.

6. **Inference-mode optimization (fixed).**
   `torch.inference_mode()` replaces `torch.no_grad()` for reduced autograd-related overhead and minor runtime improvements.

## Security & Best-Practice Notes (recommended next steps)

- **Dependency supply-chain hardening:** pin direct dependencies to exact versions and add hash-based locking (`pip-tools` or Poetry lock + hashes).
- **Model integrity verification:** optionally verify model artifact hashes when loading pretrained weights in sensitive environments.
- **Path handling hardening:** if this node is shared in multi-tenant environments, consider restricting `audio_path` to configured directories.
- **Telemetry/logging:** replace `print()` warnings with structured logging to integrate with ComfyUI logs and observability tooling.
- **Graceful import diagnostics:** catch `ImportError` for demucs/scipy with an actionable message for end users.

## Service/Architecture Upgrade Ideas

- **Faster alternatives for production pipelines:**
  - `mdx23c`/hybrid separator workflows for vocal-centric tasks,
  - UVR-style models in ensemble for better separation quality in noisy masters,
  - GPU batching service wrapper if running this at scale (queue + worker process + warm model).
- **Performance scaling:**
  - optional model caching between invocations to avoid repeated load cost,
  - chunked long-audio processing + overlap-add at node level for bounded memory profiles.

## Summary of Implemented Code Fixes

- Correct zero-phase filtering behavior with `sosfiltfilt` + short-clip fallback.
- Added filter parameter validation guard.
- Converted separation output only for required stems.
- Improved stem mixing to reduce transient allocations.
- Added explicit runtime guard for missing `guitar` stem.
- Switched to `torch.inference_mode()`.
