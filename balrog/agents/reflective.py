"""A CustomAgent with verbal reflection memory across trials."""

import json
import re
from collections import deque

from balrog.agents.custom import CustomAgent
from balrog.prompt_builder.history import Message


class ReflectiveAgent(CustomAgent):
    """Keep reflections between trials of one episode, but not between episodes."""

    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.use_reflection = bool(config.agent.use_reflection)
        max_reflections = int(config.agent.max_reflections)
        if max_reflections < 0:
            raise ValueError("max_reflections must be non-negative")

        self.reflections = deque(maxlen=max_reflections)
        self.completed_trials = 0
        self.last_reflection_response = None
        self.reflection_input_tokens = 0
        self.reflection_output_tokens = 0

    def act(self, obs, prev_action=None):
        """Add previous reflections to the normal CustomAgent prompt."""
        if not self.use_reflection or not self.reflections:
            return super().act(obs, prev_action)

        original_prompt = self.prompt_builder.system_prompt
        memory = "\n".join(
            f"{index}. {reflection}"
            for index, reflection in enumerate(self.reflections, start=1)
        )
        self.prompt_builder.system_prompt = (
            f"{original_prompt or ''}\n\n"
            "Reflections from earlier trials of this same episode:\n"
            f"{memory}\n"
            "Use these lessons only when relevant to the current state."
        )

        try:
            return super().act(obs, prev_action)
        finally:
            self.prompt_builder.system_prompt = original_prompt

    def reflect(self, success, reward=None, stats=None):
        """Summarize the completed trial and save advice for the next retry."""
        return self.reflect_on_trajectory(
            success=success,
            reward=reward,
            stats=stats,
            trajectory=None,
        )

    def reflect_on_trajectory(self, success, reward=None, stats=None, trajectory=None):
        """Create a grounded reflection from the trial's compact full trajectory."""
        self.completed_trials += 1
        if not self.use_reflection or self.reflections.maxlen == 0:
            return None

        if success is True:
            outcome = "success"
        elif success is False:
            outcome = "failure"
        else:
            outcome = "unknown"

        result = f"status={outcome}, reward={reward}"
        if stats is not None:
            result += ", stats=" + json.dumps(stats, ensure_ascii=False, default=str)

        # Format the existing reflection memory and the trajectory for the prompt
        existing_memory = "\n".join(
            f"{index}. {reflection}"
            for index, reflection in enumerate(self.reflections, start=1)
        ) or "None"
        trajectory_text = self._format_trajectory(trajectory)
        task_context = self.prompt_builder.system_prompt or ""

        # Distinguish between a successful trial and a failed one, and provide instructions for reflection accordingly
        if success is True:
            reflection_instruction = (
                "The trial succeeded. Identify the decisive successful action pattern and produce one "
                "concise winning strategy for repeating the success more efficiently. Treat the observed "
                "success as authoritative, replace speculative failure advice when necessary, and do not "
                "describe a failure cause."
            )
        else:
            reflection_instruction = (
                "The trial failed. Identify the concrete failure pattern and produce one corrective strategy "
                "for the next retry. Revise conflicting older advice, preserve any previously proven successful "
                "lesson, and do not invent object locations or events absent from the trajectory."
            )

        # Send the reflection request to the LLM and process the response
        messages = [
            Message(
                role="user",
                content=(
                    f"Task instructions:\n{task_context}\n\n"
                    f"Existing reflection memory:\n{existing_memory}\n\n"
                    f"Trial result: {result}\n\n"
                    f"Compact full trajectory:\n{trajectory_text}\n\n"
                    f"{reflection_instruction}\n"
                    "Reply in at most 100 words on exactly one line using: "
                    "REFLECTION: concise actionable strategy"
                ),
            )
        ]

        response = self.client.generate(messages)
        self.last_reflection_response = response
        self.reflection_input_tokens += getattr(response, "input_tokens", 0) or 0
        self.reflection_output_tokens += getattr(response, "output_tokens", 0) or 0

        reflection = re.sub(
            r"^\s*REFLECTION\s*:\s*",
            "",
            response.completion,
            flags=re.IGNORECASE,
        )
        reflection = re.sub(r"</?advice>", "", reflection, flags=re.IGNORECASE)
        reflection = re.sub(r"\s+", " ", reflection).strip()[:1_200]
        if reflection:
            # A successful trajectory is stronger evidence than earlier speculative failure diagnoses
            # Keep the distilled winning policy and discard advice that may conflict with it
            if success is True:
                self.reflections.clear()
            self.reflections.append(reflection)
        return reflection or None

    @staticmethod
    def _format_trajectory(trajectory):
        """Return a bounded representation that covers the whole action sequence."""
        if not trajectory:
            return "Trajectory unavailable."

        # Limit the number of action entries to avoid overwhelming the reflection prompt
        max_action_entries = 160
        if len(trajectory) <= max_action_entries:
            action_entries = trajectory
            omitted_actions = 0
        else:
            # Keep the first and last few actions, but omit the middle ones to stay within the limit
            keep_each_side = max_action_entries // 2
            action_entries = trajectory[:keep_each_side] + trajectory[-keep_each_side:]
            omitted_actions = len(trajectory) - len(action_entries)

        action_parts = []
        for entry in action_entries:
            step = entry.get("step", "?")
            action = entry.get("action", "unknown")
            plan = re.sub(r"\s+", " ", str(entry.get("plan") or "")).strip()
            item = f"{step}:{action}"
            if plan:
                item += f" [{plan[:120]}]"
            # item example: 0:turn right [face the red ball]
            action_parts.append(item)
        if omitted_actions:
            action_parts.insert(len(action_parts) // 2, f"... {omitted_actions} actions omitted ...")

        # Sample observations across the complete trial while always retaining reward-bearing and terminal transitions
        sample_count = min(16, len(trajectory))
        sampled_indices = {
            round(index * (len(trajectory) - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        }
        sampled_indices.update(
            index
            for index, entry in enumerate(trajectory)
            if entry.get("reward") or entry.get("done")
        )

        # Format the sampled observations for logging
        observation_parts = []
        for index in sorted(sampled_indices):
            entry = trajectory[index]
            observation = re.sub(
                r"\s+", " ", str(entry.get("observation") or "")
            ).strip()
            if len(observation) > 240:
                observation = observation[:237] + "..."
            observation_parts.append(
                f"step {entry.get('step', index)} after {entry.get('action', 'unknown')}: "
                f"{observation or '(no textual observation)'}; "
                f"reward={entry.get('reward', 0)}; done={bool(entry.get('done', False))}"
            )

        return (
            "Actions:\n"
            + " -> ".join(action_parts)
            + "\nKey observations sampled across the trial:\n"
            + "\n".join(observation_parts)
        )

    def get_reflections(self):
        """Return the reflection memory for logging."""
        return list(self.reflections)

    def reset_trial(self):
        """Clear the current trajectory while retaining reflections."""
        super().reset()

    def reset(self):
        """Start a new episode and clear all reflection memory."""
        super().reset()
        self.reflections.clear()
        self.completed_trials = 0
        self.last_reflection_response = None
        self.reflection_input_tokens = 0
        self.reflection_output_tokens = 0
