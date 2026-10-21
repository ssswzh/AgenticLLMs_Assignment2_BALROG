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

        messages = self.prompt_builder.get_prompt()
        messages.append(
            Message(
                role="user",
                content=(
                    f"Trial result: {result}. "
                    "Identify the main successful choice or failure cause and give "
                    "one concrete strategy for the next trial of the same episode. "
                    "Do not invent facts. Reply in at most 100 words using: "
                    "REFLECTION: <advice>"
                ),
            )
        )

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
        reflection = re.sub(r"\s+", " ", reflection).strip()[:1_200]
        if reflection:
            self.reflections.append(reflection)
        return reflection or None

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
