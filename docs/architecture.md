# WorldScape Policy architecture

WorldScape Policy provides one policy runtime for autonomous planning, direct
instruction following, goal-image conditioning, and demonstration-video
conditioning. The public model boundary is:

```text
ObservationBatch + PromptBatch
          |
    ConditionRouter
      /         \
 AutoConditioner InteractiveConditioner
          |
 VisualPrefillManager
          |
       WAMPlugin
          |
  WorldActionOutput
```

## Conditioning axes

Language interaction and visual prompting are independent axes:

- `WSP_MODE=auto` uses a vision-language model to generate latent planning
  tokens from the task, current observation, and retrieved event history.
- `WSP_MODE=interactive` encodes the current instruction directly with T5.
- `VISUAL_PROMPT=none` uses no persistent visual prompt.
- `VISUAL_PROMPT=goal` encodes one goal image as persistent visual prefill.
- `VISUAL_PROMPT=demo` encodes a uniformly sampled demonstration video as
  persistent visual prefill.

The condition router is the only component that selects Auto or Interactive
language conditioning. Goal images and demonstration videos are represented as
visual prefill and are not appended to language cross-attention tokens.

## Module ownership

Checkpoint keys, freezing rules, builders, and runtime construction share one
registered module tree:

```text
WorldScapePolicy
├── condition_router
│   ├── auto
│   │   ├── vlm
│   │   │   ├── vlm
│   │   │   └── qformer (optional)
│   │   ├── projector
│   │   └── event_memory
│   └── interactive
│       └── t5
├── visual_memory
│   └── codec
│       └── vae
└── wam
    ├── image_encoder
    └── core
```

Each trainable module has exactly one owner. The WAM owns its core and image
encoder, while the shared VAE is registered under visual memory. Runtime event
state is separate from the learned event-memory module.

## Reasoning and memory

Auto mode maintains two complementary forms of memory:

- short-term visual memory keeps recent observations as causal WAM prefill;
- long-term event memory stores bounded reasoning history for progress-aware
  retrieval.

`PolicyRuntime` applies predictions transactionally:

```text
reset(mode)
    |
predict(observation, prompt)
    |
pending action + candidate next memory
    |                         |
commit after execution       discard on failure
    |                         |
advance memory and caches    retain committed state
```

An action advances memory only after `commit()`. Rejected or unexecuted actions
must call `discard()`. Resetting a session or replacing a persistent goal/demo
prompt clears the affected visual, WAM, and event state.

## Auto token flow

The default Auto path uses Qwen3-VL final-layer states for multimodal
perception and autoregressively generated planning tokens. The two sequences
share one projector into the WAM cross-attention width.

With `VLM_TOKEN_MODE=qformer`, selected perception features are compressed by
QFormer before projection. Autoregressive planning-token states remain in the
Qwen hidden width and join the compressed perception sequence before the shared
WAM projector.

## WAM plugin boundary

`worldscape_policy.wam.registry` is the construction boundary for WAM
implementations. Plugins declare a stable name, version, configuration type,
and capability set. Callers can validate required capabilities before loading
checkpoints or allocating devices.

The built-in `wan22` plugin supports native training, sampling, image
conditioning, and causal caches. Text, goal-image, and video-demonstration
pretraining all use the same Wan2.2 parameter tree and differ only in their
conditioning inputs.

## Training and inference invariants

1. Auto samples require planning text; Interactive samples require a direct
   language instruction.
2. Interactive mode does not read or update event memory.
3. Semantic targets are training-only and apply only to Auto-routed samples.
4. Goal images and demonstration videos are mutually exclusive per sample.
5. Persistent prompts, recent observations, WAM caches, and cross-attention
   caches have separate lifecycle state.
6. The input image range is explicit: `uint8`, `zero_one`, or
   `minus_one_one`.
7. A checkpoint's interaction mode must match the requested training or
   evaluation mode.

## Checkpoint contract

Native checkpoint directories contain checksum-validated model shards,
`checkpoint_manifest.json`, and `transform_bundle.json`. Training, evaluation,
and serving construct the same module tree from these artifacts. Evaluation
also reads state/action ordering, normalization, action semantics, and
embodiment metadata from the transform bundle.

Every complete `checkpoint-N/` is resumable and evaluable. Runtime entry points
accept native WorldScape checkpoints only and fail closed when required
artifacts are missing or inconsistent.
