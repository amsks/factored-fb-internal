---
name: commit-no-claude-coauthor
description: User does not want the Co-Authored-By Claude trailer in git commits
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ef72e701-92a8-4529-8b66-8bef4943300f
---

In the Factored-FB repo, the user does not want the `Co-Authored-By: Claude ...`
trailer appended to git commit messages.

**Why:** Explicit user preference ("no coauthored by claude"), given while
committing the GCIQL plan. Overrides the harness default that asks for the
trailer.

**How to apply:** Omit the `Co-Authored-By: Claude` line from all `git commit`
messages in this project, including PR bodies' generated-with footer unless
asked. User instructions take precedence over the default harness guidance.

**Now enforced (2026-06-18):** a local `.git/hooks/commit-msg` hook strips any
Claude `Co-Authored-By` / "Generated with Claude Code" / robot-emoji line from
every commit automatically (local-only; re-install on fresh clones). Local-only
branch `jax-ogbench` was history-rewritten clean (11→0). **5 legacy coauthored
commits remain intentionally** in shared `origin` history — `89ba031`,
`32b3127`, and merges `d18ef8b`/`ac902b6`/`d6b922e` — left as-is because the user
declined the destructive force-push (rewriting them = rewriting 49–88 commits of
active shared/merged remote history). Do NOT re-flag these 5 as a problem.
