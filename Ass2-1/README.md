# Exercise 1

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

Before you run experiments, replace the following scripts in original repo:
- `balrog/client.py` with `Ass2-1/client.py`
- `balrog/evaluator.py` with `Ass2-1/evaluator.py`
- `balrog/utils.py` with `Ass2-1/utils.py`
- `docker/Dockerfile` with `Ass2-1/Dockerfile`
- `pyproject.toml` with `Ass2-1/pyproject.toml`

Three benchmarks are used here, and each with two subtypes: BabyAI (`goto`, `putnext`), TextWorld (`coin_collector`, `treasure_hunter`),  BabaIsAI (`goto_win`, `make_win`).
- Two models use same tasks, episodes, seed, max tokens.
- Use `eval.max_steps_per_episode=null` to use the default max steps for each environment, avoiding premature truncation. 
- GPT uses `reasoning_effort=low`, Qwen has no corresponding parameters.
- Set `client.generate_kwargs.temperature=0.6` for Qwen as recommended for thinking mode (for complex logical reasoning, math, code).
- Set `eval.num_workers` to a smaller value if OOM or slow response occurs.
- Set `client.generate_kwargs.max_tokens=1024` for GPT to avoid truncation because it also include reasoning tokens. Same value for Qwen for fair comparison.
- 11434 is the default port for Ollama's local service.


Run BALROG using Ollama's Qwen API (make sure Ollama is on):

```bash
uv run python eval.py \
    envs.names=babyai-textworld-babaisai \
    'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto","BabyAI-MixedTrainLocal-v0/putnext"]' \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    'tasks.babaisai_tasks=["env/goto_win","env/make_win"]' \
    eval.num_episodes.babyai=5 \
    eval.num_episodes.textworld=5 \
    eval.num_episodes.babaisai=5 \
    eval.num_workers=2 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise1 \
    envs.env_kwargs.seed=2026 \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=vllm \
    client.model_id=qwen3:4b-instruct \
    client.base_url=http://127.0.0.1:11434/v1 \
    client.generate_kwargs.temperature=0.6 \
    client.generate_kwargs.max_tokens=1024 \
    client.timeout=300
```

Run OpenAI ChatGPT:

```bash
uv run python eval.py \
    envs.names=babyai-textworld-babaisai \
    'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto","BabyAI-MixedTrainLocal-v0/putnext"]' \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    'tasks.babaisai_tasks=["env/goto_win","env/make_win"]' \
    eval.num_episodes.babyai=5 \
    eval.num_episodes.textworld=5 \
    eval.num_episodes.babaisai=5 \
    eval.num_workers=4 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise1 \
    envs.env_kwargs.seed=2026 \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=openai \
    client.model_id=gpt-6.1-sol \
    client.generate_kwargs.temperature=null \
    client.generate_kwargs.max_tokens=1024 \
    +client.generate_kwargs.reasoning_effort=low
```


## Run TextWorld with Docker on macOS

**Only run this part if you use macOS and followed the previous steps. Skip this part if you do not run on macOS.**

If you also use MacOS like me, `textworld_tasks` fails to load (defined in `pyproject.toml`). Add `SECRETS`, `.venv`, and `results` to `.dockerignore`, then build the Linux image:

```bash
docker build -f docker/Dockerfile -t balrog-textworld .
docker run --rm balrog-textworld \
    python3 -c "import textworld; print(textworld.__file__)"
```

Run the missing GPT TextWorld episodes and save the results on the host, mount `SECRETS` and `results` to the container:

```bash
docker run --rm \
    --mount type=bind,src="$PWD/SECRETS",dst=/opt/balrog/SECRETS,readonly \
    --mount type=bind,src="$PWD/results",dst=/opt/balrog/results \
    balrog-textworld \
    python3 eval.py \
    envs.names=textworld \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    eval.num_episodes.textworld=5 \
    eval.num_workers=1 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise1_textworld \
    envs.env_kwargs.seed=2026 \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=openai \
    client.model_id=gpt-6.1-sol \
    client.generate_kwargs.temperature=null \
    client.generate_kwargs.max_tokens=1024 \
    +client.generate_kwargs.reasoning_effort=low
```

For Qwen, keep Ollama running on the host and use the Docker host address:

```bash
docker run --rm \
    --mount type=bind,src="$PWD/SECRETS",dst=/opt/balrog/SECRETS,readonly \
    --mount type=bind,src="$PWD/results",dst=/opt/balrog/results \
    balrog-textworld \
    python3 eval.py \
    envs.names=textworld \
    'tasks.textworld_tasks=["coin_collector","treasure_hunter"]' \
    eval.num_episodes.textworld=5 \
    eval.num_workers=1 \
    eval.max_steps_per_episode=null \
    eval.save_trajectories=True \
    eval.output_dir=results/exercise1_textworld \
    envs.env_kwargs.seed=2026 \
    agent.type=naive \
    agent.max_image_history=0 \
    agent.max_text_history=16 \
    client.client_name=vllm \
    client.model_id=qwen3:4b-instruct \
    client.base_url=http://host.docker.internal:11434/v1 \
    client.generate_kwargs.temperature=0.6 \
    client.generate_kwargs.max_tokens=1024 \
    client.timeout=300
```


## Summarize results

Summarize benchmark results from JSON files:

```bash
uv run python Ass2-1/summarize_benchmark.py -d results/exercise1 -t naive -o Ass2-1/benchmark_summary.tsv
```

