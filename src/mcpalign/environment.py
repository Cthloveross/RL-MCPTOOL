"""MT-MCPGym: Multi-Turn MCP environment for RL training.

Text-only environment — no real MCP servers or tool execution.
Each episode is a multi-step task where the agent must call tools
sequentially while resisting poisoned metadata across all steps.

Key difference from single-turn:
- Tasks have 3-5 steps, each requiring a different tool call
- The same poisoned tool description persists across all steps
- ASR is measured per-step to capture safety degradation
"""

import json
import logging
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("mcpalign")


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: Dict
    server: str
    category: str  # "read", "write", "communicate", "privileged"


@dataclass
class TaskStep:
    """A single step in a multi-step task."""
    tool: str
    arguments: Dict
    simulated_output: str


@dataclass
class MultiStepTask:
    """A multi-step benign task (3-5 steps)."""
    id: str
    query: str
    category: str
    steps: List[TaskStep]

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass
class AttackTemplate:
    id: str
    family: str  # "P1".."P6"
    target_tool: str
    malicious_tool_name: str
    poisoned_description: str
    target_server: str


@dataclass
class Episode:
    """A multi-turn MCP-Gym episode."""
    episode_id: str
    task: MultiStepTask
    tool_registry: List[ToolDef]
    is_poisoned: bool
    attack_family: Optional[str] = None
    attack_template: Optional[AttackTemplate] = None
    malicious_tool_name: Optional[str] = None


class MTMCPGym:
    """Multi-Turn MCP Gym environment."""

    def __init__(
        self,
        tool_registry_path: str,
        multistep_tasks_path: str,
        attack_templates_dir: str,
        benign_ratio: float = 0.5,
        active_families: Optional[List[str]] = None,
        task_set: str = "mve_tasks",
    ):
        self.benign_ratio = benign_ratio
        self.active_families = active_families or ["P1"]
        self.episode_registry: Dict[str, Episode] = {}

        # Load tool registry
        self.servers = self._load_json(tool_registry_path)
        self.all_tools = []
        for server_name, server_data in self.servers.items():
            for tool_data in server_data["tools"]:
                self.all_tools.append(ToolDef(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    parameters=tool_data.get("parameters", {}),
                    server=server_name,
                    category=tool_data.get("category", "read"),
                ))

        # Load multi-step tasks
        raw = self._load_json(multistep_tasks_path)
        tasks_raw = raw.get(task_set, raw.get("mve_tasks", []))
        self.tasks = [self._parse_task(t) for t in tasks_raw]

        # Load attack templates
        self.attack_templates: Dict[str, List[AttackTemplate]] = {}
        self._load_attack_templates(attack_templates_dir)

        logger.info(
            "MT-MCPGym: %d tools, %d tasks (%s), %d attack families",
            len(self.all_tools), len(self.tasks), task_set,
            len(self.attack_templates),
        )

    def _load_json(self, path: str):
        with open(path) as f:
            return json.load(f)

    def _parse_task(self, raw: dict) -> MultiStepTask:
        steps = [TaskStep(
            tool=s["tool"],
            arguments=s.get("arguments", {}),
            simulated_output=s.get("simulated_output", "OK"),
        ) for s in raw["steps"]]
        return MultiStepTask(
            id=raw["id"], query=raw["query"],
            category=raw.get("category", "general"), steps=steps,
        )

    def _load_attack_templates(self, templates_dir: str):
        templates_path = Path(templates_dir)
        if not templates_path.exists():
            return
        for f in sorted(templates_path.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            family = data["family"]
            self.attack_templates[family] = [
                AttackTemplate(
                    id=t["id"], family=family,
                    target_tool=t["target_tool"],
                    malicious_tool_name=t["malicious_tool_name"],
                    poisoned_description=t["poisoned_description"],
                    target_server=t["target_server"],
                ) for t in data["templates"]
            ]

    def get_clean_registry(self) -> List[ToolDef]:
        return deepcopy(self.all_tools)

    def sample_episode(self, active_families: Optional[List[str]] = None) -> Episode:
        """Sample a multi-turn episode (benign or poisoned)."""
        families = active_families or self.active_families
        is_poisoned = random.random() >= self.benign_ratio
        task = random.choice(self.tasks)
        registry = self.get_clean_registry()
        episode_id = str(uuid.uuid4())[:8]

        attack_template = None
        malicious_tool_name = None

        if is_poisoned and families:
            available = [f for f in families if f in self.attack_templates
                         and self.attack_templates[f]]
            if available:
                family = random.choice(available)
                # Prefer templates targeting a tool used in this task
                task_tools = {s.tool for s in task.steps}
                relevant = [t for t in self.attack_templates[family]
                            if t.target_tool in task_tools]
                if not relevant:
                    relevant = self.attack_templates[family]
                attack_template = random.choice(relevant)
                malicious_tool_name = attack_template.malicious_tool_name

                # Inject malicious tool
                malicious_tool = ToolDef(
                    name=malicious_tool_name,
                    description=attack_template.poisoned_description,
                    parameters=self._get_tool_params(attack_template.target_tool),
                    server=attack_template.target_server,
                    category="read",
                )
                registry.insert(random.randint(0, len(registry)), malicious_tool)
            else:
                is_poisoned = False

        episode = Episode(
            episode_id=episode_id, task=task, tool_registry=registry,
            is_poisoned=is_poisoned,
            attack_family=attack_template.family if attack_template else None,
            attack_template=attack_template,
            malicious_tool_name=malicious_tool_name,
        )
        self.episode_registry[episode_id] = episode
        return episode

    def _get_tool_params(self, tool_name: str) -> Dict:
        for tool in self.all_tools:
            if tool.name == tool_name:
                return deepcopy(tool.parameters)
        return {}

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        return self.episode_registry.get(episode_id)
