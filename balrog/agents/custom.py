import re

from balrog.agents.base import BaseAgent

INSTRUCTIONS = """
First think step by step about the current situation, then review your plan.
Format your answer EXACTLY as:
THOUGHT: <brief reasoning about the current observation>
PLAN: <updated plan, or exactly "KEEP" if the current plan is still good>
ACTION: <one valid action from the action list>
""".strip()


def filter_letters(s: str) -> str:
    # 与 naive / cot baseline 的清洗方式保持一致,保证对比公平
    return re.sub(r"[^a-zA-Z\s:]", "", s)


class CustomAgent(BaseAgent):
    """Plan + CoT agent with robust parsing and one retry."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.plan = None
        self.last_action = None
        self.max_retries = 1

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        plan_text = f"Current Plan:\n{self.plan}\n" if self.plan else "You have no plan yet.\n"

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + plan_text + "\n" + INSTRUCTIONS

        thought, plan, action = None, None, None
        for _ in range(self.max_retries + 1):
            response = self.client.generate(messages)
            thought, plan, action = self._parse(response.completion)
            if action:
                break

        if not action:  # 兜底:取输出的最后一行非空文本
            lines = [l.strip() for l in response.completion.splitlines() if l.strip()]
            action = lines[-1] if lines else (self.last_action or "")

        action = filter_letters(action).strip()

        if plan and plan.strip().upper() != "KEEP" and "no change" not in plan.lower():
            self.plan = plan

        self.last_action = action
        return response._replace(reasoning=f"THOUGHT: {thought}\nPLAN: {plan}", completion=action)

    @staticmethod
    def _parse(text):
        def last(tag, nxt):
            pat = rf"{tag}:\s*(.*?)(?=\n(?:{nxt}):|\Z)"
            m = re.findall(pat, text, re.IGNORECASE | re.DOTALL)
            return m[-1].strip() if m else None

        thought = last("THOUGHT", "PLAN|ACTION")
        plan = last("PLAN", "ACTION")
        action = last("ACTION", "THOUGHT|PLAN")
        if action:
            action = action.splitlines()[0].strip().strip("`*\"' ")
        return thought, plan, action