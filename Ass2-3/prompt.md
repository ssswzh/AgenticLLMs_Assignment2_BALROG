# Exercise 3 Prompts

The reflective agent extends the custom agent from Exercise 2. The environment-specific instructions, latest 16 textual observations and actions, persistent plan, loop warning, and two-line `ACTION`/`PLAN` response format are unchanged and are documented in [`Ass2-3/prompt.md`](../Ass2-3/prompt.md). The reflective agent additionally generates a verbal reflection after each trial and makes the accumulated reflections available to later trials of the same episode.

## Prompt Assembly

During Trial 1, no reflection exists, so the action prompt is identical to the Exercise 2 custom-agent prompt. During later trials, the reflection memory is appended to the environment-specific instruction before the normal observation history, plan, loop warning, and action instruction are assembled. Source: [`balrog/agents/reflective.py`, lines 27–47](../balrog/agents/reflective.py#L27-L47).

```text
USER:
{environment-specific instruction}

Reflections from earlier trials of this same episode:
1. {reflection from an earlier trial}
2. {reflection from another earlier trial}
...
Use these lessons only when relevant to the current state.

USER:
Observation:
{previous observation}

ASSISTANT:
{previous action}

...

USER:
Current Observation:
{current observation}

{current plan or no-plan message}

{optional agentic-memory warning}

{planning and output instruction}
```

The reflection block is temporary: it is added while constructing the current action prompt and the original environment instruction is restored afterwards. This prevents the same block from being permanently duplicated in the prompt history.

## Reflection Prompt

After a trial finishes, the evaluator supplies the trial outcome, cumulative reward, progression, number of steps, failed action candidates, and recorded trajectory. The agent sends a separate reflection request containing the following user message. Sources: [`balrog/evaluator.py`, lines 371–410](./balrog/evaluator.py#L371-L410) and [`balrog/agents/reflective.py`, lines 58–112](./balrog/agents/reflective.py#L58-L112).

```text
USER:
Task instructions:
{environment-specific instruction}

Existing reflection memory:
{numbered reflections, or None}

Trial result: status={success, failure, or unknown}, reward={episode return}, stats={
  "progression": {progression},
  "num_steps": {number of steps},
  "failed_candidates": {invalid action candidates}
}

Compact full trajectory:
Actions:
{step}:{action} [{plan}] -> {step}:{action} [{plan}] -> ...
Key observations sampled across the trial:
step {step} after {action}: {observation}; reward={reward}; done={done}
...

{success or failure reflection instruction}
Reply in at most 100 words on exactly one line using: REFLECTION: concise actionable strategy
```

## Success Reflection Instruction

When progression reaches `1.0`, the following instruction asks the model to retain the action pattern supported by the successful trajectory rather than continuing to diagnose failure. Source: [`balrog/agents/reflective.py`, lines 83–90](./balrog/agents/reflective.py#L83-L90).

```text
The trial succeeded. Identify the decisive successful action pattern and produce one concise winning strategy for repeating the success more efficiently. Treat the observed success as authoritative, replace speculative failure advice when necessary, and do not describe a failure cause.
```

## Failure Reflection Instruction

For an unsuccessful trial, the following instruction requests one grounded correction for the next retry. It explicitly asks the model to revise conflicting advice without inventing facts that are absent from the trajectory. Source: [`balrog/agents/reflective.py`, lines 91–96](./balrog/agents/reflective.py#L91-L96).

```text
The trial failed. Identify the concrete failure pattern and produce one corrective strategy for the next retry. Revise conflicting older advice, preserve any previously proven successful lesson, and do not invent object locations or events absent from the trajectory.
```

## Compact Trajectory Representation

The reflection prompt uses a bounded representation of the complete trial rather than only the final 16-message action context. At most 160 action entries are included; for a longer trajectory, the first 80 and last 80 are retained and an omission marker replaces the middle. Each action entry contains its step, executed action, and an optional plan shortened to 120 characters.

Up to 16 observations are sampled at approximately even positions across the trajectory. Reward-bearing and terminal transitions are always added to this sample, and each observation is limited to 240 characters. This gives the reflection model evidence from the beginning, middle, and end of the trial while bounding prompt size. Source: [`balrog/agents/reflective.py`, lines 135–197](./balrog/agents/reflective.py#L135-L197).

## Reflection Memory and Response Handling

The parser removes the `REFLECTION:` prefix and optional `<advice>` tags, normalizes whitespace, and limits the stored text to 1,200 characters. A failed trial appends its corrective strategy to the bounded reflection deque. A successful trial first clears potentially conflicting failure advice and then stores the successful strategy. Source: [`balrog/agents/reflective.py`, lines 114–133](./balrog/agents/reflective.py#L114-L133).

With four trials and `max_reflections=3`, Trial 1 starts without reflection memory, while Trials 2–4 receive the memory produced by preceding trials. `reset_trial()` clears the custom agent's short-term plan and loop state but preserves reflection memory; the full `reset()` between episodes clears all reflections. Therefore, experience is transferred only between retries of the same episode and never between different episode tasks. Sources: [`balrog/evaluator.py`, lines 330–369](./balrog/evaluator.py#L330-L369) and [`balrog/agents/reflective.py`, lines 203–214](./balrog/agents/reflective.py#L203-L214).
