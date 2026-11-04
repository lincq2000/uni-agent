import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.tasks.swe_rebench import reward
from uni_agent.tasks.swe_rebench import task as task_module
from uni_agent.tasks.swe_rebench.task import SWEREBenchTask


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_swe_rebench_task_logs_agent_info(monkeypatch, caplog):
    sandbox = object()

    @asynccontextmanager
    async def build_sandbox():
        yield sandbox

    info = {"exit_code": 0, "error_kind": "incomplete", "cli_version": "1.19.3"}
    agent = SimpleNamespace(run=AsyncMock(return_value=AgentResult(finished=False, info=info)))
    task = object.__new__(SWEREBenchTask)
    task.config = SimpleNamespace(
        metadata={"instance_id": "test", "base_commit": "abc123"},
        model_dump=lambda **kwargs: {},
        run_oracle_solution=False,
        prompt=[{"role": "user", "content": "test"}],
        eval_timeout=60,
    )
    task.build_sandbox = build_sandbox
    task.build_agent = lambda: agent
    monkeypatch.setattr(task_module, "_clean_git_history", AsyncMock())
    monkeypatch.setattr(reward, "compute_reward", AsyncMock(return_value={"resolved": False}))

    with caplog.at_level(logging.INFO, logger=task_module.__name__):
        result = await task.run()

    assert result.finished is False
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("agent done")]
    assert lines == [f"agent done: finished=False info={json.dumps(info)}"]
