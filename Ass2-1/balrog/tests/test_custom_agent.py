from collections import namedtuple

import pytest

from balrog.agents.custom import CustomAgent
from balrog.prompt_builder.history import HistoryPromptBuilder


@pytest.fixture
def agent():
    # The parser does not depend on a model client or prompt builder
    return object.__new__(CustomAgent)


@pytest.mark.parametrize(
    ("response", "expected_plan", "expected_action"),
    [
        ("ACTION: turn left\nPLAN: Go around the wall.", "Go around the wall.", "turn left"),
        ("PLAN: Go around the wall.\nACTION: turn left", "Go around the wall.", "turn left"),
        ("  ACTION: TURN LEFT.\n  PLAN: Go around the wall.", "Go around the wall.", "turn left"),
        ("**ACTION:** `turn left`\n**PLAN:** Go around the wall.", "Go around the wall.", "turn left"),
        ("PLAN: Keep exploring until the action can be chosen", "Keep exploring until the action can be chosen", "look"),
        ("turn left", "No changes to the plan.", "look"),
        ("", "No changes to the plan.", "look"),
    ],
)
def test_extract_plan_and_action_tolerates_common_model_formats(agent, response, expected_plan, expected_action):
    plan, action = agent._extract_plan_and_action(response)

    assert plan == expected_plan
    assert action == expected_action
    assert isinstance(action, str)
    assert action


def test_extract_plan_and_action_handles_none(agent):
    plan, action = agent._extract_plan_and_action(None)

    assert plan == "No changes to the plan."
    assert action == "look"


def test_act_keeps_plan_internal_and_returns_only_compact_loop_reasoning():
    response_type = namedtuple("Response", "completion reasoning input_tokens output_tokens")
    completion = f"ACTION: TURN LEFT.\nPLAN: {'x' * 500}"
    response = response_type(completion, None, 0, 0)

    class Client:
        def generate(self, messages):
            return response

    prompt_builder = HistoryPromptBuilder()
    prompt_builder.update_instruction_prompt("Choose a valid action")
    agent = CustomAgent(lambda: Client(), prompt_builder)
    observation = {
        "text": {"long_term_context": "same state", "short_term_context": ""},
        "image": None,
    }

    first = agent.act(observation)
    second = agent.act(observation, prev_action=first.completion)
    third = agent.act(observation, prev_action=second.completion)

    assert first.completion == "turn left"
    assert first.reasoning == ""
    assert len(agent.plan) == 300
    assert "repeatedly left the observation unchanged" in third.reasoning
    assert agent.plan not in third.reasoning
