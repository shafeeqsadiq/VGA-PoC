# Lightweight Vision-Geometry-Action (VGA) Model - Proof of Concept (PoC)

Official reference implementation for the ~373M parameter VGA policy benchmarked against SmolVLA-450M on 3 spatial manipulation tasks from the LIBERO suite.

---

## Technical Highlights
* **Compact Vision-Language Core**: Pairs lightweight vision model (SigLIP) with compact language model (SmolLM2, ~181M params). A custom projector compresses visual tokens from 576 down to 64, cutting memory footprint while preserving critical manipulation features.
* **Built-in 3D Spatial Awareness**: Embeds real-world 3D camera angles and depth directly into the image features using Centroid Ray-RoPE, giving accurate spatial positioning with zero extra delay during deployment.
* **Fast, Smooth Action Generation**: Uses a 12-layer Diffusion Transformer (DiT) that generates 16-step arm trajectories in just 4 quick calculation steps. The action solver runs in ~7.2 ms, keeping total decision latency under 18 ms for real-time 50 Hz control.
* **Physics-Informed Training**: Trains model to match expert demonstrations while directly penalizing sudden acceleration spikes and jerky motor commands. An auxiliary 3D depth helper guides visual learning during training and is dropped completely at test time.
---