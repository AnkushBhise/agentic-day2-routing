"""
Prompt Manager - Generic multi-agent versioned prompt loader.

Handles ANY agent type by reading structure dynamically from YAML.
No agent-specific logic is hardcoded in this file.

Directory structure:
    prompts/
    ├── determine_issue_type/
    │   ├── v1.0.0.yaml
    │   └── current.yaml  →  symlink
    ├── refund_agent/
    │   ├── v1.0.0.yaml
    │   └── current.yaml
    └── escalation_agent/
        ├── v1.0.0.yaml
        └── current.yaml

Usage:
    manager = PromptManager()

    # Returns [SystemMessage, HumanMessage] ready for LLM
    messages = manager.compile_messages(
        agent_name   = "check_user_tier",
        user_message = "I was charged twice",
    )
    response = llm.invoke(messages)
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("PromptManager")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PromptNotFoundError(Exception):
    """Raised when the requested prompt YAML file does not exist."""


class PromptCompilationError(Exception):
    """Raised when a required section is missing or malformed."""


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """
    Generic prompt loader and compiler for any agent type.

    Assembly order (same for every agent):
    ──────────────────────────────────────
    [TOP SECURITY GUARD]
    Layer 1 — Role
    Layer 1 — Constraints
    Layer 2 — Context          (any keys, rendered dynamically)
    Layer 2 — Examples         (any response shape, rendered dynamically)
    Layer 3 — Output Format    (from output.schema in YAML)
    Layer 3 — Current Task     (live user message)
    [BOTTOM SECURITY GUARD]
    ──────────────────────────────────────

    Two ways to get output:
        compile_prompt()   → str               (full prompt as one string)
        compile_messages() → list[BaseMessage] (SystemMessage + HumanMessage)
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PromptManager ready | dir=%s", self.prompts_dir.resolve())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_prompt(self, agent_name: str, version: str = "current") -> Dict[str, Any]:
        """
        Load a versioned YAML prompt from disk.

        Args:
            agent_name : Subfolder name under prompts_dir (e.g. "refund_agent")
            version    : "current" resolves the symlink, or explicit "v1.2.0"

        Returns:
            Parsed dict with injected _meta key.

        Raises:
            PromptNotFoundError
        """
        filename    = "current.yaml" if version == "current" else f"{version}.yaml"
        prompt_file = self.prompts_dir / agent_name / filename

        if not prompt_file.exists():
            raise PromptNotFoundError(
                f"Prompt not found: {prompt_file}\n"
                f"  -> Create the file or symlink: "
                f"cd prompts/{agent_name} && ln -sf v1.0.0.yaml current.yaml"
            )

        with open(prompt_file, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = yaml.safe_load(fh)

        data["_meta"] = {
            "loaded_at"       : datetime.now(timezone.utc).isoformat(),
            "file_path"       : str(prompt_file.resolve()),
            "agent_name"      : agent_name,
            "resolved_version": data.get("version", "unknown"),
        }

        logger.info(
            "Loaded | agent=%-30s version=%s",
            agent_name,
            data["_meta"]["resolved_version"],
        )
        return data

    def compile_prompt(self, prompt_data: Dict[str, Any], user_message: str) -> str:
        """
        Compile all layers into a single LLM-ready string.

        Use this when you need the raw string (logging, debugging, inspection).
        For actual LLM calls use compile_messages() instead.

        Args:
            prompt_data  : Dict returned by load_prompt()
            user_message : Raw message from the end user.

        Returns:
            Fully assembled prompt string.

        Raises:
            PromptCompilationError
        """
        if not user_message or not user_message.strip():
            raise PromptCompilationError("user_message cannot be empty.")

        security     = prompt_data.get("security", {})
        security_top = security.get("top_guard", "")
        security_bot = security.get("bottom_guard", "")

        sections = [
            security_top,
            self._format_role(prompt_data.get("role", {})),
            self._format_constraints(prompt_data.get("constraints", {})),
            self._format_context(prompt_data.get("context", {})),
            self._format_examples(prompt_data.get("examples", [])),
            self._format_output(prompt_data.get("output", {})),
            self._format_task(user_message, prompt_data.get("output", {})),
            security_bot,
        ]

        compiled = "\n\n".join(s.strip() for s in sections if s and s.strip())
        logger.debug("Compiled | agent=%s chars=%d", prompt_data["_meta"]["agent_name"], len(compiled))
        return compiled

    def compile_messages(
        self,
        agent_name   : str,
        user_message : str,
        version      : str = "current",
    ) -> dict:
        """
        Load + compile into [SystemMessage, HumanMessage] ready for LLM.

        This is the correct method to use inside LangGraph nodes.
        LangChain LLMs expect a list of messages, not a plain string.

        Args:
            agent_name   : Subfolder name under prompts_dir
            user_message : Raw message from the end user
            version      : "current" or explicit version like "v1.0.0"

        Returns:
            [SystemMessage(system_prompt), HumanMessage(user_message)]

        Usage in a node:
            messages = manager.compile_messages("check_user_tier", enriched_message)
            response = llm.invoke(messages)

        Why split into two messages:
            SystemMessage → all the instructions, rules, examples, output format
            HumanMessage  → the actual user input the LLM responds to
            Some LLM providers require the conversation to end on a HumanMessage.
        """
        prompt_data   = self.load_prompt(agent_name, version)
        system_prompt = self._compile_system(prompt_data)

        return {
            "system": system_prompt,
            "user": user_message
        }

    def load_and_compile(
        self,
        agent_name   : str,
        user_message : str,
        version      : str = "current",
    ) -> str:
        """
        Convenience: load + compile into a single string.
        Use for debugging/logging. For LLM calls use compile_messages().
        """
        return self.compile_prompt(self.load_prompt(agent_name, version), user_message)

    def get_version_history(self, agent_name: str) -> List[str]:
        """All available versions for an agent, newest first."""
        agent_dir = self.prompts_dir / agent_name
        if not agent_dir.exists():
            return []
        return sorted([f.stem for f in agent_dir.glob("v*.yaml")], reverse=True)

    def list_agents(self) -> List[str]:
        """Return all agent names that have at least one version file."""
        return sorted([
            d.name for d in self.prompts_dir.iterdir()
            if d.is_dir() and any(d.glob("v*.yaml"))
        ])

    def get_metadata(self, agent_name: str, version: str = "current") -> Dict[str, Any]:
        """Top-level YAML metadata without compiling the prompt."""
        data = self.load_prompt(agent_name, version)
        return {
            "version"    : data.get("version"),
            "model"      : data.get("model"),
            "temperature": data.get("temperature"),
            "created_at" : data.get("created_at"),
            "created_by" : data.get("created_by"),
            **data.get("_meta", {}),
        }

    # ------------------------------------------------------------------
    # Private — system prompt builder (everything except the user message)
    # ------------------------------------------------------------------

    def _compile_system(self, prompt_data: Dict[str, Any]) -> str:
        """
        Build the system prompt — all layers EXCEPT the current task.
        The user message is kept separate as a HumanMessage.
        """
        security     = prompt_data.get("security", {})
        security_top = security.get("top_guard", "")
        security_bot = security.get("bottom_guard", "")

        sections = [
            security_top,
            self._format_role(prompt_data.get("role", {})),
            self._format_constraints(prompt_data.get("constraints", {})),
            self._format_context(prompt_data.get("context", {})),
            self._format_examples(prompt_data.get("examples", [])),
            self._format_output(prompt_data.get("output", {})),
            security_bot,
        ]

        return "\n\n".join(s.strip() for s in sections if s and s.strip())

    # ------------------------------------------------------------------
    # Private formatters — generic, no agent-specific logic
    # ------------------------------------------------------------------

    def _format_role(self, role: Dict) -> str:
        """Layer 1 - Role. Renders ANY keys found under `role:`."""
        if not role:
            return ""
        lines = ["ROLE:"]
        for key, value in role.items():
            lines.append(f"  {key.replace('_', ' ').title():<14}: {value}")
        return "\n".join(lines)

    def _format_constraints(self, constraints: Dict) -> str:
        """
        Layer 1 - Constraints.
        Strings → inline, Lists → bulleted block.
        """
        if not constraints:
            return ""
        lines = ["CONSTRAINTS:"]
        for key, value in constraints.items():
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                lines.append(f"  {label}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  - {label}: {value}")
        return "\n".join(lines)

    def _format_context(self, context: Dict) -> str:
        """Layer 2 - Context. Fully generic nested structure walker."""
        if not context:
            return ""
        lines = ["CONTEXT:"]
        for section_key, section_value in context.items():
            header = section_key.replace("_", " ").title()
            lines.append(f"\n  {header}:")
            lines.extend(self._render_value(section_value, indent=4))
        return "\n".join(lines)

    def _format_examples(self, examples: List[Dict]) -> str:
        """Layer 2 - Few-shot examples. Renders any response shape dynamically."""
        if not examples:
            return ""
        lines = ["EXAMPLES:"]
        for i, example in enumerate(examples, 1):
            scenario = example.get("scenario", f"Example {i}")
            user_msg = example.get("user", "")
            response = example.get("correct_response", {})
            lines.append(f"\n  Example {i}: {scenario}")
            lines.append(f"    User: {user_msg}")
            if response:
                lines.append("    Expected Response:")
                for key, value in response.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"      {key}:")
                        lines.extend(self._render_value(value, indent=8))
                    else:
                        lines.append(f"      {key}: {value}")
        return "\n".join(lines)

    def _format_output(self, output: Dict) -> str:
        """Layer 3 - Output format instruction. No field names hardcoded."""
        if not output:
            return ""
        description = output.get(
            "description",
            "Return ONLY a valid JSON object. No text before or after."
        )
        schema  = output.get("schema", {})
        example = output.get("example_output", "")
        lines   = ["OUTPUT FORMAT (MANDATORY):", f"  {description}"]
        if schema:
            lines.append("\n  Fields:")
            for field, meta in schema.items():
                if not isinstance(meta, dict):
                    continue
                required   = "required" if meta.get("required") else "optional"
                field_type = meta.get("type", "string")
                desc       = meta.get("description", "")
                if "enum" in meta:
                    allowed = ", ".join(f'"{v}"' for v in meta["enum"])
                    lines.append(f"    - {field} ({field_type}, {required})")
                    lines.append(f"        Allowed: [{allowed}]")
                else:
                    lines.append(f"    - {field} ({field_type}, {required}) — {desc}")
        if example:
            lines.append("\n  Example:")
            for line in example.strip().splitlines():
                lines.append(f"    {line}")
        return "\n".join(lines)

    def _format_task(self, user_message: str, output: Dict) -> str:
        """
        Layer 3 - Current task with live user message.
        Only used by compile_prompt() / load_and_compile() (the string methods).
        compile_messages() keeps user message as a separate HumanMessage instead.
        """
        task_instruction = output.get(
            "task_instruction",
            "Analyze this request and respond according to the guidelines and output format above.",
        )
        return (
            "CURRENT REQUEST:\n"
            f"  User: {user_message}\n\n"
            f"  {task_instruction}"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _render_value(self, value: Any, indent: int = 4) -> List[str]:
        """Recursively render any YAML value as indented lines."""
        pad   = " " * indent
        lines: List[str] = []
        if isinstance(value, str):
            lines.append(f"{pad}{value}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"{pad}* {k}: {v}")
                else:
                    lines.append(f"{pad}* {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                label = k.replace("_", " ").title()
                if isinstance(v, list):
                    lines.append(f"{pad}{label}:")
                    for item in v:
                        lines.append(f"{pad}  - {item}")
                elif isinstance(v, dict):
                    lines.append(f"{pad}{label}:")
                    lines.extend(self._render_value(v, indent + 2))
                else:
                    lines.append(f"{pad}{label}: {v}")
        else:
            lines.append(f"{pad}{value}")
        return lines


# ---------------------------------------------------------------------------
# CLI  —  python prompt_manager.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    manager = PromptManager(prompts_dir="prompts")

    agents = manager.list_agents()
    print(f"\nRegistered agents ({len(agents)}): {agents}")

    TEST_MESSAGE = "I was charged twice for my subscription this month"

    for agent in agents:
        print(f"\n{'=' * 60}")
        print(f"Agent    : {agent}")
        print(f"Versions : {manager.get_version_history(agent)}")

        try:
            meta = manager.get_metadata(agent)
            print(f"Model    : {meta.get('model')}  | Temp: {meta.get('temperature')}")

            # Show compiled string (debug)
            compiled = manager.load_and_compile(agent, TEST_MESSAGE)
            print(f"Compiled : {len(compiled)} chars")

            # Show messages (what nodes use)
            messages = manager.compile_messages(agent, TEST_MESSAGE)
            print(f"Messages : {len(messages)} ({[type(m).__name__ for m in messages]})")
            print(f"\nSystem prompt preview (first 300 chars):")
            print(messages[0].content[:300] + "...")

        except PromptNotFoundError as e:
            print(f"[PromptNotFoundError] {e}")
        except PromptCompilationError as e:
            print(f"[PromptCompilationError] {e}")