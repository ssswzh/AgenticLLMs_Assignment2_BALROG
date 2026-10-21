# Exercise 2 Prompts

Both models use the same custom-agent prompt construction. The environment-specific instruction and the latest 16 textual observations and actions are assembled in the same way as for the naive agent. The custom agent additionally maintains a compact plan, detects repeated states or ineffective actions, and asks the model to return a structured action and updated plan.

## Prompt Assembly

At interaction step `t`, the custom content is appended only to the latest user message. The complete message sequence has the following structure. Sources: [`balrog/prompt_builder/history.py`, lines 79–152](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/prompt_builder/history.py#L79-L152) and [`balrog/agents/custom.py`, lines 27–85](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L27-L85).

```text
USER:
{environment-specific instruction}

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

The BabyAI, TextWorld, and BabaIsAI environment instructions are unchanged from Exercise 1 and are documented in [`Ass2-1/prompts.md`](../Ass2-1/prompts.md).

## Persistent Plan

Before an initial plan exists, the agent appends:

```text
You have no plan yet.
```

On later steps, it appends the most recent plan:

```text
Current Plan:
{current plan}
```

Only the latest plan is retained. It is normalized and limited to 300 characters before being inserted into the next prompt. Source: [`balrog/agents/custom.py`, lines 47–79](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L47-L79).

## Conditional Agentic-Memory Warning

When the observation remains unchanged or a short state cycle is detected, the following block is inserted after the current plan. It is added only to the current prompt and is not stored in the permanent message history. Source: [`balrog/agents/custom.py`, lines 61–69](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L61-L69).

```text
Agentic Memory Warning:
{loop warning}
The current plan is not making progress. You must revise it and choose an action that does not continue the detected repetition.
```

Depending on the detected pattern, `{loop warning}` is one of the following. Source: [`balrog/agents/custom.py`, lines 175–225](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L175-L225).

```text
The action "{action}" has repeatedly left the observation unchanged for {steps} steps. Do not repeat this action in the current state.
```

```text
The observation has remained unchanged for {steps} consecutive actions. Try a different action or subgoal.
```

```text
The recent observations form a repeating cycle of {period} states. Recent actions: {recent actions}. Do not repeat this action sequence.
```

## Planning and Output Instruction

The following instruction is appended at every interaction step. Source: [`balrog/agents/custom.py`, lines 49–56](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L49-L56).

```text
Review the current plan above if present. Decide whether to continue with it or make changes.
Output exactly two lines in this order:
ACTION: <one exact valid action from the action list, with no punctuation or explanation>
PLAN: <one short sentence describing the next subgoal, at most 25 words>
The ACTION line must be first and must never be omitted.
Never output stop, wait, or any command that is not in the action list.
```

## Response Handling

The parser extracts the `ACTION` and `PLAN` lines while tolerating harmless Markdown decoration. Only the parsed action is sent to the environment, while the plan is retained for the next interaction step. The plan and loop memory are cleared between episodes. Sources: [`balrog/agents/custom.py`, lines 87–126](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L87-L126) and [lines 227–235](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/3420822c0449a02a4a272bc3d662979099d04e54/Ass2-2/balrog/agents/custom.py#L227-L235).
