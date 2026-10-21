# Exercise 3

## Installation

We use uv for the installation:

```bash
uv sync
source .venv/bin/activate
balrog-post-install
```

Install and run Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Download and test Qwen interactively:

```bash
ollama pull qwen3:4b-instruct
```

## Run Experiments

Three benchmarks are used here, and each with two subtypes: BabyAI (`goto`, `putnext`), TextWorld (`coin_collector`, `treasure_hunter`),  BabaIsAI (`goto_win`, `make_win`).
- Two models use same tasks, episodes, seed, max tokens.
- Use `eval.max_steps_per_episode=null` to use the default max steps for each environment, avoiding premature truncation. 
- GPT uses `reasoning_effort=low`, Qwen has no corresponding parameters.
- Set `client.generate_kwargs.temperature=0.6` for Qwen as recommended for thinking mode (for complex logical reasoning, math, code).
- Set `eval.num_workers` to a smaller value if OOM or slow response occurs.
- Set `client.generate_kwargs.max_tokens=1024` for GPT to avoid truncation because it also include reasoning tokens. Same value for Qwen for fair comparison.
- 11434 is the default port for Ollama's local service.
- Use `agent.type=reflective` to call reflective agent class.
- `agent.use_reflection=true` to enable reflection.
- `agent.max_reflections=3` to set the maximum number of reflections.
- `eval.num_trials=4` to set the number of trials for each episode, so the latter 3 trials would use reflections.


Run BALROG using Ollama's Qwen API (make sure Ollama is on).

Run model with reflections:

```bash
uv run python eval.py \
    envs.names=babyai-textworld-babaisai \
    'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto","BabyAI-MixedTrainLocal-v0/putnext"]' \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    'tasks.babaisai_tasks=["env/goto_win","env/make_win"]' \
    eval.num_episodes.babyai=10 \
    eval.num_episodes.textworld=10 \
    eval.num_episodes.babaisai=10 \
    eval.num_workers=2 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise3_with_reflection \
    envs.env_kwargs.seed=2026 \
    agent.type=reflective \
    agent.use_reflection=true \
    agent.max_reflections=3 \
    eval.num_trials=4 \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=vllm \
    client.model_id=qwen3:4b-instruct \
    client.base_url=http://127.0.0.1:11434/v1 \
    client.generate_kwargs.temperature=0.6 \
    client.generate_kwargs.max_tokens=1024 \
    client.timeout=300 
```

Run model without reflections:

```bash
uv run python eval.py \
    envs.names=babyai-textworld-babaisai \
    'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto","BabyAI-MixedTrainLocal-v0/putnext"]' \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    'tasks.babaisai_tasks=["env/goto_win","env/make_win"]' \
    eval.num_episodes.babyai=10 \
    eval.num_episodes.textworld=10 \
    eval.num_episodes.babaisai=10 \
    eval.num_workers=2 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise3_without_reflection \
    envs.env_kwargs.seed=2026 \
    agent.type=reflective \
    agent.use_reflection=false \
    agent.max_reflections=0 \
    eval.num_trials=4 \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=vllm \
    client.model_id=qwen3:4b-instruct \
    client.base_url=http://127.0.0.1:11434/v1 \
    client.generate_kwargs.temperature=0.6 \
    client.generate_kwargs.max_tokens=1024 \
    client.timeout=300 
```


## Summarize results

Summarize results from JSON files by trials:

