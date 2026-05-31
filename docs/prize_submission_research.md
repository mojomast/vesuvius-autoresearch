# Vesuvius Challenge Progress Prize Submission Research

Research timestamp: 2026-05-28. Public web sources were fetched on this date. Facts marked uncertain could not be confirmed from a publicly accessible page during this pass.

## Executive Summary

The Progress Prize is active on the official Open Prizes page, with the next monthly deadline listed as **11:59pm Pacific, May 31st, 2026**. Submission is through the Progress Prize Google Form at <https://forms.gle/LrpQmSAqdwGpTczLA>. Current Progress Prize award tiers are **Gold Aureus: $20,000**, **Denarius: $10,000**, **Sestertius: $2,500**, and **Papyrus: $1,000**. The official criteria favor open-sourced early tools that are actually used by the community and are well documented.

For the autoresearch tool, the strongest framing is: an open-source, reproducible research automation contribution that directly targets ink-detection generalization and accelerates scroll-reading experiments. The submission should include a standalone demo, run instructions, result screenshots/plots, clear standard-format inputs/outputs, and evidence that community members can run or adapt it. The interactive dashboard strengthens this framing by exposing safe diagnostic controls, fold-matrix actions, folded decoded-output review, a filterable scroll-contained experiment ledger, and optional agent chat in the same reviewer-facing interface as the promotion evidence.

## Current Open Prize Tiers and Amounts

Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.

### First Letters and Title Prizes

- **First Letters: $60,000 to the first team that uncovers 10 letters within a single 4cm^2 area of any of Scrolls 2-3.**
- **First Title: $60,000 to the first team to discover the title in any of Scrolls 2-3.**

### Progress Prizes

The Progress Prize section states: "In addition to milestone-based prizes, we offer monthly prizes for open source contributions that help read the scrolls. These prizes are more open-ended, and we have a wishlist to provide some ideas."

Current tiers listed verbatim:

- **Gold Aureus: $20,000 (estimated 4-8 per year) - for major contributions**
- **Denarius: $10,000 (estimated 10-15 per year)**
- **Sestertius: $2,500 (estimated 25 per year)**
- **Papyrus: $1,000 (estimated 50 per year)**

## Deadline

Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.

The official page currently says: **"Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, May 31st, 2026!"**

Conclusion: May 31, 2026 is active according to the official page as of 2026-05-28.

## Submission URL and Method

Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.

Form source rechecked from <https://forms.gle/LrpQmSAqdwGpTczLA> on 2026-05-28. The Google Form rendered field labels without JavaScript but cannot be submitted from this environment because it requires a user email/name and an interactive Google Forms submission.

- Progress Prize submission method: Google Form.
- Progress Prize submission URL: <https://forms.gle/LrpQmSAqdwGpTczLA>
- First Letters / First Title submission URL: <https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform?usp=dialog>
- Required Progress Prize form fields visible from fetched form: email, full name, team description, URL to open-source contribution, short description of how the contribution increases probability of reading complete scrolls, confirmation that a Community Projects / awesome-scroll-tools PR was submitted, and terms acceptance.
- Community Projects PR submitted: <https://github.com/ScrollPrize/villa/pull/991>

Related communication channels:

- Official Discord invite in navigation and Get Started page: <https://discord.com/invite/uTfNwwecCQ> and <https://discord.gg/V4fJhvtaQn>
- Mailing list / Substack: <https://scrollprize.substack.com/>
- GitHub repository: <https://github.com/ScrollPrize/villa>
- FAQ for confidential progress: "Please email us at [email protected]. We will keep it confidential." The email is Cloudflare-obfuscated on the public FAQ, so the exact address is uncertain from the fetched markdown. Source: <https://scrollprize.org/faq>, fetched 2026-05-28.

## Judging Criteria and Requirements

Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.

### What They Favor

The Progress Prize page states:

> We favor submissions that:
>
> - Are **released or open-sourced early**. Tools released earlier have a higher chance of being used for reading the scrolls than those released the last day of the month.
> - Actually **get used**. We’ll look for signals from the community: questions, comments, bug reports, feature requests. Our Annotation Team will publicly provide comments on tools they use.
> - Are **well documented**. It helps a lot if relevant documentation, walkthroughs, images, tutorials or similar are included with the work so that others can use it!

### Core Requirements

The page lists these requirements verbatim:

> **Core Requirements:**
>
> 1. Problem Identification and Solution
>    - Address a specific challenge using Vesuvius Challenge scroll data
>    - Provide clear implementation path and a demonstration of its use
>    - Demonstrate significant advantages over existing solutions
> 2. Documentation
>    - Include comprehensive documentation
>    - Provide usage examples
> 3. Technical Integration
>    - Accept standard community formats (e.g. OME-Zarr or Zarr arrays, quadmeshes, triangular meshes)
>    - Maintain consistent output formats
>    - Designed for modular integration

Dashboard integration notes for the submission package:

- Safe run controls are gated by `VESUVIUS_DASHBOARD_ENABLE_RUNS=1` and token auth, so reviewers can use the dashboard without enabling execution.
- The Agent Chat panel defaults to a Hermes-style local provider for this workspace but is configurable through `VESUVIUS_DASHBOARD_AGENT_BASE_URL`, `VESUVIUS_DASHBOARD_AGENT_MODEL`, and `VESUVIUS_DASHBOARD_AGENT_API_KEY` for non-Hermes users.
- Agent output is advisory and cannot execute commands; artifact-writing research commands remain copy-only.
- Optional agent settings action mode is limited to allowlisted non-secret dashboard preferences, requires `VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE=1`, validates every patch server-side, and requires a user click before apply.
- Settings versioning creates immutable local recovery snapshots before every apply and before rollback, making broken dashboard settings rewindable without relying on the agent.
- Optional decoded-output visual analysis is opt-in through `VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_ENABLED=1` or the versioned dashboard setting, sends only guarded preview images/metrics, and cannot mutate settings or run commands.
- Long dashboard surfaces are reviewer-safe by default: the usefulness leaderboard and decoded-output gallery start folded, the validation matrix includes explicit filter/best-run and safe-command actions, and the experiment ledger is scroll-contained with text, status, segment, and F1 filters. The hard `20230530172803` fold is profiled as AP/separability evidence, so weak hard-fold ranking triggers audit/sampling review instead of relaxed threshold claims.

### Terms Relevant to Submission

The Terms and Conditions state:

> Prizes are awarded at the sole discretion of Curious Cases, Inc. and are subject to review by our Technical Team, Annotation Team, and Papyrological Team. We may issue more or fewer awards based on the spirit of the prize and the received submissions. You agree to make your method open source if you win a prize. It does not have to be open source at the time of submission, but you have to make it open source under a permissive license to accept the prize.

## Wishlist / Tools They Need

Source: <https://scrollprize.org/prizes>, <https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22>, fetched 2026-05-28.

The official Progress Prize page says:

> We maintain a public wishlist of ideas that would make excellent progress prize submissions. Improvements to VC3D can be also considered for progress prizes! Some are additionally labeled as good first issues for newcomers!

The linked public wishlist is GitHub issues labeled `help wanted`, with the label text shown by GitHub as "Good candidate for a Progress Prize." Open examples visible in the fetched page include:

- Review: Explore whole volume deformation by exploiting newly available information (vertical fibers, large meshes), issue #203, opened Apr 18, 2025.
- Scroll specific 3d augmentations for model training, issue #201, opened Apr 18, 2025.
- Identify and update out-of-date documentation and other information on the scrollprize.org website, issue #199, opened Apr 18, 2025.
- Methods for generating surface, fiber, or ink labels, issue #193, opened Apr 18, 2025.
- Accurate 3d ink labels, issue #192, opened Apr 18, 2025.
- Surface and Fiber Predictions in Compressed or Highly Curved areas, issue #191, opened Apr 18, 2025.

The Get Started page also frames the current need as: "We need help in automating the virtual unwrapping of entire scrolls" and points open-source contributors to "Awesome Scroll Tools" and the "Progress Prizes Wish List." Source: <https://scrollprize.org/get_started>, fetched 2026-05-28.

## Community and Reviewer Discovery

Requested source note: <https://scrollprize.org/community> returned 404 when fetched on 2026-05-28. The current public equivalent appears to be <https://scrollprize.org/community_projects>, titled "Awesome Scroll Tools."

### How Reviewers Discover Submissions

Confirmed sources:

- Official submissions are discovered through the Progress Prize submission form: <https://forms.gle/LrpQmSAqdwGpTczLA>. Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.
- Reviewers also look for community usage signals. The official language is: "We’ll look for signals from the community: questions, comments, bug reports, feature requests. Our Annotation Team will publicly provide comments on tools they use." Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.
- The Terms say awards are subject to review by the Technical Team, Annotation Team, and Papyrological Team. Source: <https://scrollprize.org/prizes>, fetched 2026-05-28.

Uncertain:

- The exact internal workflow by which reviewers monitor Discord threads or map Discord posts to form submissions is not publicly specified in the fetched sources. Source uncertainty: <https://scrollprize.org/prizes> and <https://scrollprize.org/faq>, fetched 2026-05-28.

### Where to Announce Open-Source Tools

Confirmed sources:

- The FAQ encourages public sharing and says: "Be sure to also post in our Discord, to get feedback from the community." Source: <https://scrollprize.org/faq>, fetched 2026-05-28.
- The FAQ says: "If you're open to sharing your improvements publicly (and be eligible for progress prizes), you can post in Discord." Source: <https://scrollprize.org/faq>, fetched 2026-05-28.
- The Community Projects page says: "For state-of-the-art updates join our Discord server" and "If you want to contribute and add any resource please submit a PR!" Source: <https://scrollprize.org/community_projects>, fetched 2026-05-28.

Recommended announcement path:

- Submit the form before the deadline.
- Open-source the repo early under a permissive license.
- Post a concise demo and documentation link in the Discord.
- If accepted or useful, submit a PR to add the tool to the Community Projects / Awesome Scroll Tools page.

## Recent 2025-2026 Progress Prize Examples

Sources: <https://scrollprize.org/winners>, <https://scrollprize.substack.com/archive?sort=new>, and individual Substack posts fetched 2026-05-28.

### 1. Scroll Slab Viewer, Paul Geiger, $1,000 Papyrus, March 2026

Sources: <https://scrollprize.org/winners>, <https://scrollprize.substack.com/p/we-are-cooking>, <https://github.com/Paul-G2/ScrollSlabViewer>.

What won:

- A "very user-friendly 3D viewer tailored for the Kaggle Surface Detection challenge."
- The post explicitly cited community usage: it "was upvoted by 70+ Kaggle competitors, and used by several competitors to post screenshots in Kaggle discussions."

Documentation/demo expectation inferred:

- Standalone repository.
- Tool was easy enough for Kaggle competitors to use and produce screenshots.
- Community-use evidence mattered, not just technical novelty.

Contribution framing lesson:

- Frame tools around immediate workflow fit and visible adoption signals.
- Include screenshots, a quickstart, and a demo dataset/path so reviewers can reproduce the user experience.

### 2. VC3D Stability Improvements, Philip Allgaier, $5,000, May 2025

Sources: <https://scrollprize.org/winners>, <https://scrollprize.substack.com/p/may-progress-prizes-and-updates-to>, <https://github.com/spacegaier/volume-cartographer/blob/progress-prizes/changes-2025-05.md>.

What won:

- Significant improvements to VC3D stability and ease of use.
- The post emphasized prebuilt Docker packages, fixes for memory leaks, GitHub CI, Ubuntu/Qt compatibility, and clearer console output.
- The post linked to a full change list.

Documentation/demo expectation inferred:

- Reviewers value a concrete change log tied to known issues.
- Packaging and easy installation are prize-worthy when they unblock community usage.
- Fixing existing tool pain points can beat a flashy new algorithm if it improves the pipeline.

Contribution framing lesson:

- For autoresearch, emphasize reduced setup friction, reproducible runs, clear logs/artifacts, and any CI or containerized execution.
- Map the contribution to specific Vesuvius bottlenecks: running model research repeatedly, comparing experiments, and transferring results across scrolls.

### 3. Ball-and-Spring / Stress Metrics, Will Stevens, $2,500, April-July 2025

Sources: <https://scrollprize.org/winners>, <https://scrollprize.substack.com/p/april-progress-prizes-updates>, <https://scrollprize.substack.com/p/summer-haze-comes-with-ink>, <https://github.com/WillStevens/scrollreading/blob/main/report6.pdf>, <https://github.com/WillStevens/scrollreading/blob/main/report7.pdf>.

What won:

- April 2025: extended a ball-and-spring surface-growing algorithm. The writeup says the per-vertex stress map acts as "an intrinsic quality check, clearly exposing layer 'sheet switches' and other mistakes."
- June/July 2025: continued particle-based simulation and stress metrics work. The post says high-stress areas often correspond to errors in surface predictions and could mitigate growth-process errors.

Documentation/demo expectation inferred:

- A technical report with visual outputs was sufficient to communicate the contribution, even where the work was research-heavy.
- Reviewers rewarded iterative progress over multiple months when each installment produced clearer diagnostics or better tooling.

Contribution framing lesson:

- A report-backed research contribution should include experimental evidence, before/after visuals, failure-mode analysis, and exact reproduction steps.
- Autoresearch should be framed as an iterative accelerator that can keep producing monthly measurable improvements.

## Documentation Depth and Standalone Demo Expectations

Across recent Progress Prize posts, winning submissions commonly include:

- A public repository or report link.
- Screenshots, GIFs, plots, or rendered outputs.
- A narrow problem statement tied to the current reading pipeline.
- Evidence of usefulness: community usage, issue fixes, speedups, install simplification, or new diagnostic capability.
- Enough documentation for another person to run or inspect the contribution.

The official criteria explicitly ask for "comprehensive documentation" and "usage examples," plus "a demonstration of its use." For an autoresearch submission, a minimal acceptable package should include:

- `README.md` quickstart with environment setup.
- One command to run a small demo.
- A documented example using Vesuvius data or a small included/mock fixture where raw data cannot be bundled.
- Sample outputs: experiment database rows, artifacts, plots, generated hypotheses, model cards, and logs.
- A dashboard walkthrough or screenshot set showing the reviewer-facing state: current candidate, promotion gate, LOO/full-tile evidence, quality verdicts, collapsed usefulness leaderboard controls, validation fold-matrix actions, folded decoded-output gallery, filterable run ledger, and next recommended command.
- Reproducibility notes: exact configs, seeds where relevant, hardware assumptions, expected runtime, and dependencies.
- A short technical report explaining what specific scroll-reading bottleneck it solves and what measurable improvement it produced.

## Recommended Framing for the Autoresearch Tool Submission

### Positioning

Frame the tool as: **an open-source research automation system that helps the community systematically discover, run, compare, and document ink-detection experiments on Vesuvius Challenge data.**

Tie it directly to the official requirements:

- Problem identification: ink-detection generalization and experiment throughput are bottlenecks for reading multiple scrolls.
- Demonstration: include a reproducible run that starts from a config, launches an agent/experiment loop, and produces a ranked result summary.
- Advantages: fewer manual experiment cycles, auditable logs, standardized output artifacts, easier comparison across scrolls and model variants, and a dashboard that makes promotion safety review legible to humans.
- Documentation: include a quickstart, architecture overview, config guide, and annotated example run.
- Technical integration: document supported Vesuvius data paths/formats, especially Zarr/OME-Zarr if supported; otherwise mark current format support honestly and list next steps.

### Evidence to Include

- A concise benchmark table showing baseline vs autoresearch-discovered configuration or architecture.
- If available, a cross-scroll metric such as validation Dice improvement on one scroll when training on another.
- Screenshots of the dashboard or artifact viewer. For this submission, the dashboard should be treated as first-class evidence because it turns scattered experiment artifacts into an actionable review surface; include the compact default view and expanded examples of the leaderboard, fold matrix actions, decoded gallery, and filtered ledger.
- Links to logs/artifacts that show the full chain from hypothesis to run to result.
- A short failure analysis section explaining hallucination/overfitting risks, especially if model outputs could be interpreted as ink.

### Dashboard Framing

The dashboard is a strong Progress Prize asset, not a minor convenience. It gives reviewers and future contributors a fast, read-only way to understand the state of the research loop: which candidate is best, what evidence supports it, which folds failed, whether full-tile provenance is eligible, whether positive-rate caps are binding, and what action should happen next. It now keeps the page usable for large histories by collapsing long leaderboards/gallery panels by default, adding actions to the validation fold matrix, and containing the experiment run ledger in a searchable scroll window. This is exactly the kind of practical tooling that reduces wasted scroll-research cycles.

The dashboard also helps with hallucination control. It makes unsafe-looking wins visible by showing positive-rate ratios, AP/prevalence lift, fixed-threshold diagnostics, full-tile checks, and promotion warnings together. That makes it harder to accidentally promote a pretty but inflated patch result, and easier for community reviewers to reproduce the reasoning behind a rejection.

### Community Discovery Plan

- Submit the Progress Prize Google Form before 11:59pm Pacific on May 31, 2026.
- Post the GitHub repo, demo screenshots, and a short "how to try this" message in Discord.
- Invite bug reports and feature requests to create the community signals the prize page says reviewers consider.
- Open issues labeled `good first issue` or similar so community members can contribute agent prompts, experiment templates, and data adapters.
- Consider a PR to the Vesuvius Community Projects page after the tool is usable.

### Submission Narrative Draft

The submission should say, in substance:

> This contribution helps read the scrolls by making ink-detection research reproducible and scalable. It automates the loop of proposing model/training changes, running controlled experiments, recording artifacts, and ranking outcomes. The included demo shows a complete run on Vesuvius-style data and produces standardized logs, metrics, and visual reports. This is designed for modular integration with the community pipeline and to help researchers quickly validate which ideas improve cross-scroll generalization.

Post-submission update: the repository now also includes a no-download synthetic demo generator, a cleaner README landing page, residual 2.5D CPU-safe search configs, committed DB/runner audit logs, sampled/full-tile/LOO residual evidence, and updated research-status docs. The latest residual work found strong sampled and eligible full-tile signal but correctly blocked promotion after LOO exposed zero-fold failures on `20230530172803`, demonstrating the value of the dashboard and promotion gates.

## Source Index

- Official Open Prizes: <https://scrollprize.org/prizes>, fetched 2026-05-28.
- Get Started: <https://scrollprize.org/get_started>, fetched 2026-05-28.
- FAQ: <https://scrollprize.org/faq>, fetched 2026-05-28.
- Community Projects / Awesome Scroll Tools: <https://scrollprize.org/community_projects>, fetched 2026-05-28.
- Requested Community URL: <https://scrollprize.org/community>, returned 404 on 2026-05-28.
- Winners: <https://scrollprize.org/winners>, fetched 2026-05-28.
- Substack archive: <https://scrollprize.substack.com/archive?sort=new>, fetched 2026-05-28.
- March 2026 update / Scroll Slab Viewer prize: <https://scrollprize.substack.com/p/we-are-cooking>, fetched 2026-05-28.
- May 2025 Progress Prizes: <https://scrollprize.substack.com/p/may-progress-prizes-and-updates-to>, fetched 2026-05-28.
- April 2025 Progress Prizes: <https://scrollprize.substack.com/p/april-progress-prizes-updates>, fetched 2026-05-28.
- July 2025 Progress Prize: <https://scrollprize.substack.com/p/summer-haze-comes-with-ink>, fetched 2026-05-28.
- Public wishlist GitHub issues: <https://github.com/ScrollPrize/villa/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22>, fetched 2026-05-28.
