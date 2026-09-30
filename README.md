# Lightweight Vision-Geometry-Action (VGA) Model - Proof of Concept (PoC)

Official reference implementation for the ~373M parameter VGA policy benchmarked against SmolVLA-450M on 3 spatial manipulation tasks from the LIBERO suite.

---

## Technical Highlights
* **Lightweight Backbone**: SigLIP-B/16 (LoRA blocks 6–12) compressed via a $3\times$ Space-to-Depth Projector ($576 \to 64$ tokens) coupled to a pruned 12-layer SmolLM2 spine (~181M params).
* **Zero-Overhead 3D Grounding**: Centroid Ray-RoPE projects metric unit rays ($\mathbf{d} \in \mathbb{S}^2$) and camera origin translation into visual tokens with zero test-time latency penalty.
* **Continuous Action Expert**: 12-layer Diffusion Transformer (DiT) executing 4-step Euler ODE integration ($\le 7.2\text{ ms}$, total policy latency $\le 18\text{ ms}$ at 50 Hz).
* **Multi-Objective Loss**: Error-weighted flow matching regularized with annealed kinematic acceleration ($\mathcal{L}_{acc}$), jerk penalties ($\mathcal{L}_{jerk}$), and an auxiliary VQ-depth cross-entropy head ($\mathcal{L}_{vq}$) that is detached at test time.

---

## Cloud Deployment Guide (RunPod / Lambda Labs / Vast.ai)

### 1. Launch & Connect to Instance
Rent an on-demand instance with **1× NVIDIA RTX 4090 (24GB VRAM)**, $\ge 16$ vCPUs, and a persistent volume mounted at `/workspace`.

```bash
ssh root@<instance-ip> -p <port>
cd /workspace