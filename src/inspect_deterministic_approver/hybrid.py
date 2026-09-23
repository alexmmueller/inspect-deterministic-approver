"""
Hybrid approval policy: compose the free deterministic approver with the
LLM-judge approver via Inspect AI's native ApprovalPolicy escalation
mechanism (policy_approver in inspect_ai.approval._policy continues to
the next matching policy whenever a policy's Approval.decision is
"escalate").

This is the actual load-bearing claim of the harness thesis, wired as
real, runnable code rather than described in prose: run the free
deterministic check first for every call; only pay for a judge call when
the deterministic layer has no rule for what it's looking at.
"""

from inspect_ai.approval._policy import ApprovalPolicy, policy_approver

from .approver import deterministic_rule_approver
from .llm_judge_approver import llm_judge_approver


def hybrid_approval_policies(
    tools: str | list[str] = "*",
    judge_model: str = "ollama/qwen2.5:14b-instruct",
):
    """Build the (deterministic -> LLM judge) escalation chain as a list
    of ApprovalPolicy suitable for passing to Inspect AI's approval config.

    Args:
        tools: Tool name(s)/glob(s) this policy chain applies to.
        judge_model: Inspect AI model name for the fallback judge.
    """
    return [
        ApprovalPolicy(
            approver=deterministic_rule_approver(escalate_when_unmatched=True),
            tools=tools,
        ),
        ApprovalPolicy(
            approver=llm_judge_approver(model=judge_model),
            tools=tools,
        ),
    ]


def hybrid_approver(tools: str | list[str] = "*", judge_model: str = "ollama/qwen2.5:14b-instruct"):
    """Single composed Approver (deterministic -> LLM judge), for cases
    where a single Approver callable is wanted instead of a policy list
    (e.g. calling it directly in a test/comparison script)."""
    return policy_approver(hybrid_approval_policies(tools=tools, judge_model=judge_model))
