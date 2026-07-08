"""The one permanent Main Agent.

It never solves the user's task itself. It engineers other agents:
analyze -> plan -> generate code -> save files -> install deps -> execute -> merge.
"""

from pathlib import Path

from executor import execute_all, install_dependencies, save_agent_file
from generator import generate_agent_code
from merger import merge_outputs
from planner import plan_agents


def handle_task(task: str) -> tuple[str, list[Path]]:
    """Run the full lifecycle for one user task.

    Returns the final merged response and the generated agent files
    (so the caller can ask the user to delete or keep them).
    """
    print("\n[1/5] Analyzing task and planning agents...")
    plan = plan_agents(task)
    print(f"  Plan: {plan.reasoning}")
    for spec in plan.agents:
        print(f"  - {spec.name}: {spec.role}")

    print("\n[2/5] Generating agent code...")
    agent_paths = []
    for spec in plan.agents:
        code = generate_agent_code(spec)
        path = save_agent_file(spec, code)
        print(f"  Wrote {path.name}")
        agent_paths.append(path)

    print("\n[3/5] Checking dependencies...")
    install_dependencies(plan.agents)

    print("\n[4/5] Executing agents...")
    outputs = execute_all(agent_paths, task)

    print("\n[5/5] Merging outputs...")
    final_response = merge_outputs(task, outputs)

    return final_response, agent_paths
