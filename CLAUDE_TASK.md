# Claude Task Brief — Finish and Professionalize `lseg-thematic-portfolio-optimization`

## Mission
You are working inside the local VS Code repository for **`lseg-thematic-portfolio-optimization`**.
Your job is to **finish the project cleanly and professionally** without changing its core identity.

This is **not** a request to invent a new project.
This is a request to **turn the current repo into a stronger, more coherent, more recruiter-ready GitHub project**.

The main strategic repositioning is:

> **Make ERC (Equal Risk Contribution) the flagship allocation of the project.**
>
> The repo should clearly communicate that, in this concentrated AI/Tech universe, ERC is the most defensible portfolio because it diversifies **risk**, not just capital, while keeping a strong balance between return, volatility, and drawdown.

Black-Litterman should remain in the repo, but as an **advanced extension**, not the main story.

---

## Non-negotiable high-level goals

1. **Preserve the existing project idea**
   - Keep the AI/Tech universe.
   - Keep the five strategies.
   - Keep the walk-forward structure.
   - Keep the stress / factor / BL layers if they already exist.

2. **Reposition the repo around ERC**
   - ERC must become the clearest central insight.
   - Equal Weight vs ERC should be explained visually and intuitively.
   - The README should make ERC feel like the recommended allocation, with the other strategies serving as benchmarks.

3. **Improve professionalism and reproducibility**
   - Cleaner packaging/environment story.
   - Easier demo path for a reviewer without LSEG access.
   - Better tests and CI.
   - Better repo coherence between README, code, outputs, and install instructions.

4. **Do not fabricate results**
   - Do not invent performance numbers.
   - Do not invent charts.
   - Do not invent unavailable LSEG data.
   - If something cannot be reproduced locally, say so clearly and create a safe fallback/demo path.

---

## Current repo context you should assume unless inspection proves otherwise

The public repo currently presents itself as:
- a **20-stock AI/Tech universe**,
- a **walk-forward monthly backtest**,
- five strategies:
  - Equal Weight
  - Minimum Variance
  - Maximum Sharpe
  - ERC
  - Minimum CVaR
- an additional **risk/factor/stress layer**,
- and a **Black-Litterman extension**.

The current public messaging appears to over-emphasize the general framework and the BL extension, while ERC—despite being one of the most defensible results—is not yet sufficiently positioned as the main practical conclusion.

There also appears to be room to improve:
- README narrative hierarchy,
- ERC-specific visual outputs,
- ERC-specific tests,
- demo/repro path without live LSEG dependency,
- packaging coherence (`requirements.txt`, `setup.py`, Python version story),
- visible CI.

Do **not** blindly trust this summary.
First inspect the repo as it currently exists locally and adapt intelligently.

---

## Your workflow

Work in this order.
Do **not** skip the inspection step.

### Phase 0 — Inspect first
Before editing, inspect:
- root structure
- `README.md`
- `main.py`
- `src/`
- `tests/`
- config files
- current outputs/reporting code
- package/install files

Then write a short internal implementation plan for yourself.
Do not rewrite the whole project from scratch.

---

## Phase 1 — Rewrite the project story so ERC is clearly the flagship

### Objective
Transform the repo from:
- “a generic portfolio optimization framework with many equally emphasized strategies”

into:
- “a serious portfolio optimization project whose main insight is that ERC is the most defensible allocation in this concentrated AI/Tech universe, with other methods used as comparison points and BL as an advanced extension.”

### README tasks
Rewrite `README.md` professionally.
Keep it concise, technical, and recruiter-friendly.

### New README structure target
Use this approximate structure:

1. **Project title and one-paragraph value proposition**
2. **Main insight**
   - State clearly why ERC matters in this universe.
3. **What the project does**
4. **Universe**
5. **Methodology**
   - walk-forward setup
   - rebalancing
   - constraints
   - transaction costs
6. **Strategies compared**
   - but frame ERC as the centerpiece
7. **Key results**
8. **Why ERC is the flagship allocation**
   - explain Equal Weight vs ERC in plain but serious language
9. **Stress / factor / robustness evidence**
10. **Black-Litterman extension**
    - clearly label it as an advanced extension
11. **How to run**
12. **Demo mode / no-LSEG fallback**
13. **Project structure**
14. **Known limitations**
15. **Next improvements**

### Messaging constraints
The README must explicitly communicate these ideas:
- Equal Weight diversifies **capital**.
- ERC diversifies **risk**.
- In a concentrated AI/Tech universe, this distinction matters.
- ERC is more robust and more defensible than a pure return-maximizing portfolio.
- Max Sharpe is a useful benchmark but not the preferred practical allocation.
- Black-Litterman is interesting, but it is an advanced extension, not the core message.

### Important
Do not exaggerate.
Do not oversell “production-ready” if it is not actually production-grade.
Use language like:
- “research-grade”
- “recruiter-ready”
- “defensible”
- “walk-forward”
- “risk-budgeted”
- “illustrative advanced extension”

Avoid fake institutional claims.

---

## Phase 2 — Add ERC-specific outputs that make the thesis obvious

### Objective
Create ERC-focused outputs that visually prove the project’s main point.

### Add or improve charts
Create ERC-specific reporting outputs if they do not already exist.
Prefer saving them under something like:
- `outputs/charts/`
- `docs/images/`
- or the repo’s existing reporting destination

### Required ERC visuals
Implement as many of these as the current codebase cleanly supports:

1. **Equal Weight vs ERC: capital weights vs risk contributions**
   - This is the most important figure.
   - It should make the conceptual difference obvious.

2. **ERC risk contribution chart at the latest rebalance**
   - Show whether risk contributions are approximately balanced.

3. **ERC sector composition over time**
   - Useful if sector tagging already exists or can be added safely.

4. **ERC turnover / stability chart**
   - Show ERC is not just theoretically nice, but reasonably stable.

5. **ERC rolling relative performance vs Equal Weight**
   - Optional, only if easy and honest.

### Reporting rules
- Reuse existing outputs if available.
- Avoid fragile one-off notebooks if a script/module is cleaner.
- If the repo already has a reporting pipeline, integrate there.
- If not, create a clean reporting utility.

---

## Phase 3 — Add ERC-specific tests

### Objective
Make ERC verifiable, not just marketable.

### Add tests such as
1. **Risk contributions sum correctly**
2. **ERC solution has reasonably balanced risk contributions**
3. **A much more volatile asset receives a lower ERC weight**
4. **Weights remain valid under constraints**

If the existing portfolio optimization code already has helper functions for marginal risk contribution or total portfolio volatility, reuse them.
If not, implement them cleanly.

### Testing standard
Tests should be deterministic and not depend on live LSEG data.
Use synthetic covariance matrices / synthetic returns where needed.

---

## Phase 4 — Add a clean demo mode for users without LSEG access

### Objective
A reviewer should be able to run **something meaningful** without needing local LSEG Desktop entitlements.

### Acceptable solutions
Choose the cleanest option supported by the repo:

#### Option A — Sample dataset
Add a very small versioned sample dataset under something like:
- `data/sample/`

and provide a command like:

```bash
python main.py --demo
```

#### Option B — Dedicated demo script
If modifying the main pipeline is messy, add something like:

```bash
python scripts/run_demo.py
```

that reproduces:
- strategy comparison table,
- at least one ERC chart,
- a small output folder,
- no live LSEG dependency.

### Rules
- Do not pretend the demo is full live production.
- Clearly label it as demo/sample mode.
- Document it in the README.

---

## Phase 5 — Clean up packaging and environment coherence

### Objective
Make the repo look internally consistent.

Inspect and align:
- `requirements.txt`
- `setup.py`
- any `pyproject.toml` if present
- Python version in README badges / instructions

### Preferred direction
If easy, modernize toward a clean `pyproject.toml`.
If not, keep `setup.py`, but make the repo consistent.

### Minimum expected outcome
- one clear Python version target
- one clear installation path
- dependency versions that make sense together
- no contradictory messaging across files

---

## Phase 6 — Add visible CI

### Objective
Make the repo feel actively maintained and technically disciplined.

Add a minimal GitHub Actions workflow that:
- installs Python,
- installs dependencies,
- runs tests.

If a full install is fragile because of LSEG dependency, structure the workflow so that testable parts still run.
For example, split the package layers or allow tests to run without requiring a live LSEG session.

The workflow should be clean and realistic, not fake.

---

## Phase 7 — Tighten project structure and polish

If the current repo supports it cleanly, also do the following:

### Good polish items
- improve folder consistency,
- reduce dead code or outdated comments,
- make output paths consistent,
- improve function names where obviously confusing,
- add docstrings to important public functions,
- make CLI entry points clearer,
- ensure plots save with predictable filenames,
- ensure generated output directories are created safely.

### But do not
- launch a giant refactor for its own sake,
- break working code to chase elegance,
- over-engineer abstractions,
- introduce unnecessary dependencies.

---

## Deliverables expected from you
By the end, I want you to have modified the repo so that it includes:

1. **A rewritten README centered on ERC**
2. **ERC-specific visuals**
3. **ERC-specific tests**
4. **A no-LSEG demo path**
5. **Cleaner packaging/environment consistency**
6. **A visible CI workflow**
7. **A short final changelog summary** in your terminal/output message

---

## Quality bar
Your output should feel like the work of a strong research engineer / quant developer finishing a serious student project for GitHub and interviews.

The result should be:
- coherent,
- honest,
- reproducible where possible,
- easy to explain in an interview,
- visibly centered on ERC.

---

## Style guide for edits

### README style
- crisp
- technical but readable
- no hypey marketing language
- no generic filler
- no long walls of text
- use tables where useful
- use bullets sparingly

### Code style
- clean
- explicit
- typed where reasonable
- small functions preferred over giant blocks
- deterministic tests
- clear filenames

---

## Hard constraints

### Do not do these things
- Do not remove ERC focus.
- Do not replace the AI/Tech universe with a new universe.
- Do not turn BL into the main thesis.
- Do not fabricate performance outputs.
- Do not require me to manually do lots of cleanup after you.
- Do not leave the repo half-migrated between two packaging systems without explanation.

### If something is unclear
Do not ask vague strategic questions.
Inspect the repo and make the strongest reasonable implementation choices.
Only flag blockers if they are real technical blockers.

---

## Concrete success criterion
When a recruiter or reviewer lands on the repo, they should quickly understand this message:

> This project studies portfolio construction in a concentrated AI/Tech universe and shows, through walk-forward evidence, stress analysis, and risk decomposition, that Equal Risk Contribution is the most balanced and defensible allocation. Other optimizers provide useful benchmarks, and Black-Litterman is included as an advanced extension rather than the main recommendation.

If your edits make that statement obviously true from the repo, you succeeded.

---

## Suggested execution order
Follow this order unless inspection reveals a better one:

1. inspect repo
2. rewrite README
3. add ERC reporting outputs
4. add ERC tests
5. add demo mode
6. clean packaging
7. add CI
8. run tests / smoke checks
9. summarize changes clearly

---

## Final instruction
Do the work directly in the codebase.
Be proactive.
Be strict.
Be honest.
Prefer a smaller number of high-quality, finished improvements over a larger number of half-finished ones.
