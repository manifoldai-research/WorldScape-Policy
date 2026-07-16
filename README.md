# WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory

WorldScape Policy 2.0 is a controllable World Action Model (WAM) with reasoning-augmented long short-term memory for long-horizon robotic manipulation, fine-grained instruction following, visual-context reasoning, and in-context skill transfer.

[![Project Page](https://img.shields.io/badge/Project-Page-6F35C7?logo=googlechrome&logoColor=white)](https://manifoldai-research.github.io/WorldScape-Policy/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg?logo=adobeacrobatreader)](https://manifoldai-research.github.io/WorldScape-Policy/assets/docs/worldscape-policy-2.pdf)

## News

- **Code is coming soon.** We are preparing the training, inference, and evaluation code for release.
- Project page: https://manifoldai-research.github.io/WorldScape-Policy/

## Overview

World Action Models jointly model future visual state transitions and robot actions, providing a natural interface for robot planning and controllable execution. However, existing WAMs are often limited by short temporal context, coarse episode-level language supervision, and text-only conditioning.

WorldScape Policy 2.0 addresses these limitations with a reasoning-augmented long short-term memory design:

- **Short-term visual memory** supplies recent observations as causal DiT prefill to preserve local interaction dynamics.
- **Long-term event memory** organizes historical VLM outputs into global-history, local-active, and event-boundary representations for progress-aware retrieval.
- **Latent subgoal reasoning** uses retrieved history to augment perception and autoregressively generated planning tokens.
- **Event-grounded pretraining** builds fine-grained multimodal controllability from text prompts, goal images, video demonstrations, and action trajectories.

## Key Features

- Long-horizon autonomous planning from high-level task instructions.
- Fine-grained instruction following from event-level subtask captions.
- Memory-dependent visual reasoning for tasks such as shell-game state tracking.
- In-context skill transfer from goal images and video-context demonstrations.
- Unified video-action modeling with ManipEvent-5M pretraining.

## Citation

```bibtex
@article{worldscape_policy_2026,
  title={WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory},
  author={Manifold AI - WorldScape Team},
  year={2026},
  url={https://manifoldai-research.github.io/WorldScape-Policy/}
}
```
