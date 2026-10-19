# Exercise 1 Prompts

Both models use the same naive-agent prompt construction. The agent uses the latest 16 textual observations and actions, with no image history, few-shot examples, or explicit chain-of-thought prompt.

## Prompt Assembly

The environment instruction and observations are sent as user messages, while previous actions are represented as assistant messages. At interaction step `t`, the messages have the following structure. Source: [`balrog/prompt_builder/history.py`, lines 79–140](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/prompt_builder/history.py#L79-L140).

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
```


## Shared Action Constraint

The following instruction is appended to the current observation at every interaction step. Source: [`balrog/agents/naive.py`, lines 32–37](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/agents/naive.py#L32-L37).

```text
You always have to output one of the above actions at a time and no other text. You always have to output an action until the episode terminates.
```


## BabyAI

`{mission}` is replaced by the mission generated for the current episode, while `{action_strings}` is generated from the action definitions above. Sources: [`balrog/environments/babyai_text/__init__.py`, action definitions at lines 3–10](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/babyai_text/__init__.py#L3-L10) and [prompt template at lines 13–28](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/babyai_text/__init__.py#L13-L28).


```text
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

turn left: turn to the left,
turn right: turn to the right,
go forward: take one step forward,
pick up: pick up the object below you,
drop: drop the object that you are holding,
toggle: manipulate the object in front of you.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
```

## TextWorld: Coin Collector

Source: [`balrog/environments/textworld/__init__.py`, lines 62–71](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/textworld/__init__.py#L62-L71).

```text
You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated maze and must find the coin. You need to explore different rooms to find the target object. Here are the available commands: goal: print the goal of this game go <dir>: move the player north, east, south, or west. You can only go in the direction indicated with something like an exit or a door. take coin: 2in the game by ‘take coin’ if you see the coin in the room The only action you can do is go <dir> to explore the maze and ‘take coin’ when you see the coin in the room. You have 25 steps to complete the task. Restarting is forbidden.
```

## TextWorld: Treasure Hunter

Source: [`balrog/environments/textworld/__init__.py`, lines 14–31](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/textworld/__init__.py#L14-L31).

```text
You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated maze and must find a specific object. You need to explore different rooms to find the target object. Here are the available commands: look: describe the current room. goal: print the goal of this game inventory: print the player’s inventory go <dir>: move the player north, east, south, or west. You can only go in the direction indicated with an exit or a door. open ...: open a door or a container. You need to open a closed door before you want to go through it. drop ...: drop an object on the floor take ...: take an object that is visible. Make sure the object is visible to take. put ... on ...: place an object on a supporter take ... from ...: take an object from a container or a supporter insert ... into ...: place an object into a container unlock ... with ...: unlock a door or a container with a key. You need to unlock a locked door with a matched key in your inventory before you want to open it. - The target object might be located in a closed or locked container. - The adjective is useful for determining whether the key is matched with the lock (e.g. non-euclidean keycard is matched with non-euclidean safe). Make sure it is matched to unlock! - The key required to unlock the door may be in another room or locked inside a container. - Take the key whenever you can. - After unlocking a locked door or container, it will remain closed. You will then need to open it. You have 40 steps to complete the task. Restarting is forbidden.
```

## BabaIsAI

Sources: [`balrog/environments/babaisai/__init__.py`, action definitions at lines 6–12](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/babaisai/__init__.py#L6-L12) and [prompt template at lines 15–34](https://github.com/ssswzh/AgenticLLMs_Assignment2_BALROG/blob/e7297f3bbfcad0c96e2036b264dc1c52d8b48f83/balrog/environments/babaisai/__init__.py#L15-L34).

```text
Baba Is You is a puzzle game where you can manipulate the rules of each level. The following are the possible actions you can take in the game, followed by a short description of each action:

idle: wait for one step,
up: take one step up,
right: take one step to the right,
down: take one step down,
left: take one step to the left.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.

PLAY!
```
