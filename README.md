<p align="center">
  <a href="https://balrogai.com">
    <img src="docs/imgs/balrog_banner.png" width="50%" alt="BALROG Agent" />
  </a>
</p>

---

# BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games

BALROG is a novel benchmark evaluating agentic LLM and VLM capabilities on long-horizon interactive tasks using reinforcement learning environments. Check out how current models fare on our [leaderboard](https://balrogai.com). You can read more about BALROG in our [paper](https://arxiv.org/abs/2411.13543).

## Installation

We use uv for the installation

```bash
uv sync
source .venv/bin/activate
balrog-post-install
```

On Mac make sure you have `wget` installed for the `balrog-post-install`

## ⚡️ Evaluate using vLLM locally

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
ollama serve
```

Download and test Qwen interactively:

```bash
ollama pull qwen3:4b-instruct
ollama run qwen3:4b-instruct
# use /bye to quit
```

Run BALROG using Ollama's Qwen API:

```bash
uv run python eval.py \
  envs.names=babyai \
  eval.num_episodes.babyai=1 \
  "tasks.babyai_tasks=[BabyAI-MixedTrainLocal-v0/goto]" \
  eval.num_workers=1 \
  eval.max_steps_per_episode=20 \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  envs.env_kwargs.seed=2026 \
  eval.output_dir=results/stage1_pilot \
  client.generate_kwargs.temperature=0.7 \
  client.generate_kwargs.max_tokens=256 \
  client.client_name=vllm \
  client.model_id=qwen3:4b-instruct \
  client.base_url=http://localhost:11434/v1
```


## 🛜 Evaluate using API

We support how of the box clients for OpenAI, Anthropic and Google Gemini APIs. If you want to evaluate an agent using one of these APIs, you first have to set up your API key in one of two ways:

You can either directly export it:

```bash
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
```

Or you can modify the `SECRETS` file, adding your api keys.

You can then run the evaluation with:

```bash
uv run python eval.py \
  envs.names=babyai \
  'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto"]' \
  eval.num_episodes.babyai=1 \
  eval.num_workers=1 \
  eval.max_steps_per_episode=10 \
  agent.type=naive \
  agent.max_image_history=0 \
  agent.max_text_history=16 \
  client.client_name=openai \
  client.model_id=gpt-6.1-sol \
  client.generate_kwargs.temperature=null \
  client.generate_kwargs.max_tokens=512 \
  +client.generate_kwargs.reasoning_effort=low
```

## Documentation

- [Evaluation Guide](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md) - Detailed instructions for various evaluation scenarios
- [Agent Development](https://github.com/balrog-ai/BALROG/blob/main/docs/agents.md) - Tutorial on creating custom agents
- [Few Shot Learning](https://github.com/balrog-ai/BALROG/blob/main/docs/few_shot_learning.md) - Instructions on how to run Few Shot Learning

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/balrog-ai/BALROG/blob/main/docs/contribution.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BALROG in any of your work, please cite:

```
@article{paglieri2024balrog,
  title={Benchmarking Agentic LLM and VLM Reasoning On Games},
  author={Paglieri, Davide and Cupia{\l}, Bart{\l}omiej and Coward, Sam and Piterbarg, Ulyana and Wo{\l}czyk, Maciej and Khan, Akbir and Pignatelli, Eduardo and Kuci{\'n}ski, {\L}ukasz and Pinto, Lerrel and Fergus, Rob and Foerster, Jakob Nicolaus and Parker-Holder, Jack and Rockt{\"a}schel, Tim},
  journal={arXiv preprint arXiv:2411.13543},
  year={2024}
}
```
