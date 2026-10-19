import hashlib
import re
from collections import Counter, deque

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """An agent that generates actions and plans based on observations with a customizable approach."""
    # The memory length is set to 12 to cover repeated short cycles of up to three states while remaining within the 16-step textual context window
    # This value is a practical heuristic rather than an optimised hyperparameter
    MEMORY_SIZE = 12
    NO_CHANGE_THRESHOLD = 2
    MAX_CYCLE_PERIOD = 3

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        # Save the long-term plan and the history of actions and states (sha256 of observations) to detect loops or repeated ineffective actions.
        self.plan = None
        self.action_history = deque(maxlen=self.MEMORY_SIZE)
        self.state_history = deque(maxlen=self.MEMORY_SIZE)
        self.state_visit_counts = Counter()
        self.no_progress_steps = 0
        self.last_loop_event = None

    def act(self, obs, prev_action=None):
        """Generate the next action and update the plan based on the current observation.

        Args:
            obs (dict): The current observation in the environment.
            prev_action (str, optional): The previous action taken.

        Returns:
            LLMResponse: The response containing the updated plan and selected action.
        """
        # Update the memory with the current observation and previous action, and check for loops or repeated ineffective actions
        loop_warning = self._update_memory(obs, prev_action)

        # Append previous action to the prompt builder: 
        # HistoryPromptBuilder._events.append({"type": "action", "action": prev_action, "reasoning": self.previous_reasoning})
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        # HistoryPromptBuilder._events.append({"type": "observation", "text": text, "image": self.image})
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        planning_instructions = """
Review the current plan above if present. Decide whether to continue with it or make changes.
If you make changes, provide the updated plan. Then, provide the next action to take.
You must output an action at every step.
Format your answer in the following way:
PLAN: <your updated plan if changed, or "No changes to the plan." if the current plan is good>
ACTION: <your next action>
        """.strip()

        # Only append the plan and instructions to the last user message in the prompt
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            memory_text = ""
            # Add warning to the message, not to _events, to avoid cluttering the prompt with repeated warnings
            if loop_warning:
                memory_text = (
                    "\nAgentic Memory Warning:\n"
                    + loop_warning
                    + "\nThe current plan is not making progress. You must revise it and choose an action that does not continue the detected repetition.\n"
                )
            messages[-1].content += "\n\n" + plan_text + memory_text + "\n" + planning_instructions

        response = self.client.generate(messages)

        # Extract the plan and action from the LLM's response
        plan, action = self._extract_plan_and_action(response.completion)

        # Update the internal plan if it has changed
        if plan and plan.casefold() != "no changes to the plan.":
            self.plan = plan

        # Save the plan in the response.reasoning field and the action in response.completion
        effective_plan = self.plan or plan
        reasoning = effective_plan
        if loop_warning:
            reasoning = f"{loop_warning}\nPlan: {effective_plan}"
        # Use namedtuple._replace to create a new instance with updated fields, since namedtuples are immutable
        response = response._replace(reasoning=reasoning, completion=action)
        return response

    def _extract_plan_and_action(self, response_text):
        """Extract the plan and action from the LLM's response text.

        Args:
            response_text (str): The response text from the LLM.

        Returns:
            tuple: A tuple containing the extracted plan and action.
        """
        # Local and smaller models often wrap the requested labels in Markdown
        # (for example ``**ACTION:**``) or indent them. Accept those harmless
        # variations instead of returning ``None`` and passing it to env.step().
        if response_text is None:
            response_text = ""
        elif not isinstance(response_text, str):
            response_text = str(response_text)

        label_decoration = r"(?:\*\*|__|`{1,3})?"
        plan_match = re.search(
            rf"^\s*{label_decoration}\s*PLAN\s*:\s*{label_decoration}\s*(.*?)"
            rf"(?=^\s*{label_decoration}\s*ACTION\s*:|\Z)",
            response_text,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        action_match = re.search(
            rf"^\s*{label_decoration}\s*ACTION\s*:\s*{label_decoration}\s*([^\r\n]+)",
            response_text,
            re.IGNORECASE | re.MULTILINE,
        )

        plan = plan_match.group(1).strip() if plan_match else "No changes to the plan."
        if action_match:
            action = action_match.group(1).strip().strip("`*_\"' ")
        else:
            # Always return a non-empty string. Most environments will replace
            # an invalid fallback with their default action; TextWorld accepts
            # arbitrary text commands and cannot safely receive ``None``.
            nonempty_lines = [line.strip().strip("`*_\"' ") for line in response_text.splitlines() if line.strip()]
            action = nonempty_lines[-1] if nonempty_lines else "wait"

        return plan, action

    def _state_signature(self, obs):
        """Return a stable signature for the current textual observation."""
        text_observation = obs.get("text", {})
        long_term = text_observation.get("long_term_context", "")
        short_term = text_observation.get("short_term_context", "")
        state_text = f"{long_term}\n{short_term}"

        # Invalid-action feedback is transient metadata, not part of the game state, remove it
        state_text = re.sub(
            r"^\s*Your previous output did not contain a valid action\..*?Observation:\s*",
            "",
            state_text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Normalize whitespace and compute a SHA256 hash of the state text to create a stable signature
        normalized = re.sub(r"\s+", " ", state_text).strip()
        # Same hash = same state, even if the text is long and truncated in the prompt
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _update_memory(self, obs, prev_action):
        """Record the latest transition and return a warning when a loop is detected."""
        # Get the current state SHA256 signature based on the observation
        current_state = self._state_signature(obs)

        if not self.state_history:
            self.state_history.append(current_state)
            self.state_visit_counts[current_state] += 1
            self.last_loop_event = None
            return None

        # Add the previous state to the history and update visit counts
        previous_state = self.state_history[-1]
        if prev_action:
            self.action_history.append(prev_action.strip())
        self.state_history.append(current_state)
        self.state_visit_counts[current_state] += 1

        if current_state == previous_state:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        # Detect loops or repeated ineffective actions and return a warning if detected
        self.last_loop_event = self._detect_loop()
        return self._format_loop_warning(self.last_loop_event)

    def _detect_loop(self):
        """Detect repeated ineffective actions or short repeated state cycles."""
        actions = list(self.action_history)
        states = list(self.state_history)

        # Check for repeated ineffective actions (same action leading to unchanged state)
        if self.no_progress_steps >= self.NO_CHANGE_THRESHOLD:
            recent_actions = actions[-self.NO_CHANGE_THRESHOLD :]
            if recent_actions and len(set(recent_actions)) == 1:
                return {
                    "type": "action_repetition",
                    "action": recent_actions[-1],
                    "steps": self.no_progress_steps,
                }
            return {"type": "unchanged_state", "steps": self.no_progress_steps}

        # Check for short repeated state cycles (same state appearing multiple times)
        # e.g. [A, B, A, B] is a cycle of period 2, [A, B, C, A, B, C] is a cycle of period 3
        max_period = min(self.MAX_CYCLE_PERIOD, len(states) // 2)
        for period in range(2, max_period + 1):
            if states[-2 * period : -period] == states[-period:]:
                return {
                    "type": "state_cycle",
                    "period": period,
                    "actions": actions[-(2 * period) :],
                }

        return None

    @staticmethod
    def _format_loop_warning(loop_event):
        """Convert a detected loop into concise feedback for the language model."""
        if not loop_event:
            return None

        if loop_event["type"] == "action_repetition":
            return (
                f'The action "{loop_event["action"]}" has repeatedly left the observation unchanged '
                f'for {loop_event["steps"]} steps. Do not repeat this action in the current state.'
            )
        if loop_event["type"] == "unchanged_state":
            return (
                f'The observation has remained unchanged for {loop_event["steps"]} consecutive actions. '
                "Try a different action or subgoal."
            )

        recent_actions = ", ".join(loop_event["actions"]) or "unavailable"
        return (
            f'The recent observations form a repeating cycle of {loop_event["period"]} states. '
            f"Recent actions: {recent_actions}. Do not repeat this action sequence."
        )

    def reset(self):
        """Reset the agent's internal state, including the plan and histories."""
        super().reset()
        self.plan = None
        self.action_history.clear()
        self.state_history.clear()
        self.state_visit_counts.clear()
        self.no_progress_steps = 0
        self.last_loop_event = None
