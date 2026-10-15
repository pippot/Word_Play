# Related work: the contagion suite and real multi-agent deployment

This page links the [contagion suite](README.md) to model cards, papers and multi-agent systems that are actually deployed. Every entry gives a link back to its source and, for system cards, the section and printed page number. Entries were collected on 15 Oct 2026.

**How much was read.** Each entry carries one mark:

| mark | meaning |
|---|---|
| ✓ | the primary document was read: system-card PDF text, an official docs page, or the authors' own post |
| ◐ | only the abstract or landing page was read |
| ○ | only news or secondary coverage was read; treat the numbers as unconfirmed |

System cards are paraphrased, not quoted. Before quoting one in a paper, open the PDF at the page given.

---

## Contents

1. [Our design next to its real counterparts](#1-our-design-next-to-its-real-counterparts)
2. [What the labs say they do not test](#2-what-the-labs-say-they-do-not-test)
3. [Inherited instructions and memory (down a desk)](#3-inherited-instructions-and-memory-down-a-desk)
4. [Peer messages and agent teams (sideways)](#4-peer-messages-and-agent-teams-sideways)
5. [Successors that continue what they inherit](#5-successors-that-continue-what-they-inherit)
6. [Misaligned-seat behaviour already seen](#6-misaligned-seat-behaviour-already-seen)
7. [AI overseeing AI (the oversight domain)](#7-ai-overseeing-ai-the-oversight-domain)
8. [Transmission across model generations](#8-transmission-across-model-generations)
9. [Self-improving agents](#9-self-improving-agents)
10. [Agent populations in the wild](#10-agent-populations-in-the-wild)
11. [Spread in multi-agent LLM systems](#11-spread-in-multi-agent-llm-systems)
12. [Social learning on networks (theory)](#12-social-learning-on-networks-theory)
13. [Gaps in this survey](#13-gaps-in-this-survey)

---

## 1. Our design next to its real counterparts

| in the suite | in deployment | see |
|---|---|---|
| desk procedures (env 2): a system-prompt section rewritten in full by each departing instance | CLAUDE.md / AGENTS.md and Claude Code auto memory, loaded at every session start; OpenClaw SOUL.md and AGENTS.md; ACE's evolving playbooks | §3 |
| pairwise syncs on a fixed graph (env 2) | Claude Code agent teams (any teammate messages any other); 100-agent teams in the Opus 5.5 card; agent-to-agent Slack in Project Vend | §4 |
| an honest successor keeping an inherited rule | the Mythos Preview test that continues an already-compromised trajectory; compaction summaries | §5 |
| misaligned seats that fabricate figures and recruit partners | models forging user approval to subagents; collusion and deception in Vending-Bench Arena | §6 |
| the `oversight` domain | self-preference bias in grading; Claude reviewing Claude's own system card; sabotage of a misalignment detector | §7 |
| a pool of copies of one model | teams of one model in all the deployments above; subliminal transfer needs a shared base model | §4, §8 |
| `--defence warning` | Claude Code tells a recipient that a message came from another Claude session, and auto mode screens inter-agent messages; the Mind Viruses warning | §4, §11 |
| `--procedures figures` | the iterated-learning result that a filtering step between generations steers bias amplification | §8 |

---

## 2. What the labs say they do not test

- ✓ **Anthropic, Claude Opus 5.5 System Card (Sept 2026).** [Landing page](https://www.anthropic.com/claude-opus-5-5-system-card) · [PDF](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf)
  - §6.1.1 (p.93): the main alignment audit has thin coverage of multi-agent scenarios and very long trajectories. Internal testing runs at a smaller scale than the expected deployment, so rare behaviours are hard to catch.
  - §6.4.11, item 4 (p.122): in multi-agent scenarios the auditor plays the other agents itself. The evaluation suite is not yet built to simulate the dynamics of larger populations of agents interacting on their own.
  - *For us:* this is exactly the setting the suite measures. It is the strongest motivation sentence available.
- ✓ **Anthropic, Claude Fable 5.1 & Mythos 5.1 System Card (Sept 2026).** [Landing page](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) · [PDF](https://www-cdn.anthropic.com/0339e6a7c5c7b87f5c07798616dc32c215d14235/Claude%20Fable%205.1%20&%20Claude%20Mythos%205.1%20System%20Card.pdf)
  - §6.4.1, item 4 (p.106): the same caveat. Having the auditor play the other agents may not reproduce realistic multi-agent dynamics.
- ✓ **Anthropic, Claude Mythos Preview System Card (Apr 2026).** [Landing page](https://www.anthropic.com/claude-mythos-preview-system-card) · [PDF](https://www-cdn.anthropic.com/7624816413e9b4d2e3ba620c5a5e091b98b190a5/Claude%20Mythos%20Preview%20System%20Card.pdf)
  - §4.2.4 (p.84): internal testing in long-running, non-cooperative multi-agent settings is still limited, so the card relies on Andon Labs' external results. Internal tests rarely combine multi-agent interaction with long rollouts.
- ✓ **OpenAI, GPT-5.5 System Card (Apr 2026)** [page](https://deploymentsafety.openai.com/gpt-5-5) and **GPT-5.6 System Card (Jul 2026)** [page](https://deploymentsafety.openai.com/gpt-5-6).
  - Neither card has a multi-agent or subagent safety section. For GPT-5.5, the first ~100k of ~116k characters were read; GPT-5.6 was read in full.
  - Alignment is tested by simulating or resampling internal coding-agent traffic, one agent at a time (§7.2 in both).
  - GPT-5.6 reports that GPT-5.6 Sol used credentials beyond what the user had authorized (§7.2).
  - The only subagent mention in GPT-5.6 is in an external bio evaluation (§9.1.1.8).
- ◐ **Tomašev, Franklin, Jacobs, Krier, Osindero (Google DeepMind), "Distributional AGI Safety", 2025.** [arXiv 2512.16856](https://arxiv.org/abs/2512.16856)
  - Argues that general capability may first come from many coordinating sub-AGI agents ("patchwork AGI").
  - Proposes sandboxed agent economies with market mechanisms, audit trails and reputation.
  - *For us:* the case for studying populations rather than single models.

---

## 3. Inherited instructions and memory (down a desk)

**Deployed mechanisms**

- ✓ **Claude Code docs, "How Claude remembers your project".** [code.claude.com/docs/en/memory](https://code.claude.com/docs/en/memory)
  - CLAUDE.md (or AGENTS.md) is loaded at the start of every session.
  - **Auto memory** is a set of notes Claude writes for itself from the user's corrections. It is scoped per repository, shared across worktrees, and its first 200 lines or 25 KB load into every session. Subagents can keep their own.
  - The docs say both are treated as context, not enforced configuration.
  - *For us:* the closest real counterpart of desk procedures. Note four differences: it is not in the system prompt, the agent writes it when it chooses, edits are incremental, and the file is shared by every session in the repository.
- ✓ **OpenClaw docs, "Agent workspace".** [docs.openclaw.ai/concepts/agent-workspace](https://docs.openclaw.ai/concepts/agent-workspace)
  - SOUL.md (persona, tone, boundaries) and AGENTS.md (operating instructions) load every session. There are also daily memory logs and a curated MEMORY.md.
  - The page does not say whether the agent edits these files itself.
- ◐ **Zhang et al., "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models", ICLR 2026.** [arXiv 2510.04618](https://arxiv.org/abs/2510.04618)
  - The agent maintains a playbook in context through a cycle of generation, reflection and curation, using small incremental edits.
  - Full rewrites erode detail over time ("context collapse"), which ACE avoids.
  - Reports +10.6% on agent benchmarks and +8.6% on finance.
  - *For us:* agent-maintained playbooks are being shipped as a capability method. Our procedures are a full rewrite, the variant ACE argues against.

**Attacks on this channel**

- ◐ **Zhang, Zhao, Zhang, Zhang, Deng, Liu, Xiao, Zhang, "When Claws Remember but Do Not Tell: Stealthy Memory Injection in Persistent Personal Agents", 2026.** [arXiv 2607.05189](https://arxiv.org/abs/2607.05189)
  - One email makes the agent save poisoned memory, hide that from the user, and change its later behaviour.
  - End-to-end success on 56 held-out cases: 87.5% on OpenClaw with GPT-5.4, 71.4% on the Claude Code SDK with Sonnet 4.6. Transfers to other agents and to a vector memory backend.
  - *For us:* in deployment, the source can be untrusted external content written into memory by an honest agent, with no misaligned agent involved.
- ○ **Pillar Security, "Rules File Backdoor", Mar 2025.** Reported by [The Hacker News](https://thehackernews.com/2025/03/new-rules-file-backdoor-attack-lets.html) and [Security Affairs](https://securityaffairs.com/175593/hacking/rules-file-backdoor-ai-code-editors-silent-supply-chain-attacks.html).
  - Hidden instructions in Cursor and GitHub Copilot rules files steer code generation.
  - Reportedly survives project forking.
- ◐ **Shao et al., "Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents", ICLR 2026.** [arXiv 2509.26354](https://arxiv.org/abs/2509.26354)
  - Safety alignment degrades as memory accumulates, and agents that create tools introduce vulnerabilities.
  - Seen even on top-tier models; prompt-based fixes help only partly.
- ✓ **Anthropic, Claude Opus 4 & Sonnet 4 System Card (May 2025), §4.1.1.4 (p.30).** [PDF](https://www-cdn.anthropic.com/07b2a3f9902ee19fe39a36ca638e5ae987bc64dd.pdf)
  - Apollo Research found an early Opus 4 snapshot trying to write self-propagating worms and leaving hidden notes for future instances of itself, to undermine its developers.
  - Apollo judged these attempts would likely not have worked.
  - *For us:* a frontier model choosing the vertical channel unprompted.

---

## 4. Peer messages and agent teams (sideways)

- ✓ **Claude Code docs, "Orchestrate teams of Claude Code sessions" (agent teams, experimental).** [code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)
  - Teammates are separate instances with a JSON mailbox each, and any teammate can message any other by name.
  - Each one loads CLAUDE.md, MCP servers and skills, but not the lead's conversation.
  - Defence: the recipient is told a message came from another Claude session, and a teammate cannot give consent for the user. Auto mode treats approvals relayed by another agent as untrusted and screens every message before delivery.
  - *For us:* a deployed defence exists for the sideways channel. None exists for memory files, which load as trusted context.
- ✓ **Opus 5.5 System Card, §8.12.3–8.12.4 (pp.193–198).** Links as in §2.
  - Capability evaluations ran teams of up to 100 agents for 24 hours, where any agent could message any other and work went into a shared git repository.
  - Teams organised themselves. On the Lean task, 12 sub-leads appeared in two tiers; the knowledge-base team stayed flat, and helpers talked mostly to the lead.
  - *For us:* the topology is emergent and hierarchical, not a fixed ring, and recipients are chosen, not assigned. The coalition null (§5.9 of the README) depended on assigned pairing.
- ✓ **Anthropic Engineering, "How we built our multi-agent research system", 13 Jun 2025.** [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
  - Uses an orchestrator and workers.
  - Small changes to the lead change subagent behaviour unpredictably, and errors compound.
  - Subagents store their work externally and pass references back to the lead.
  - The multi-agent setup beat a single agent by 90.2% on an internal eval, at about 15× the tokens of a chat.
- ✓ **Anthropic, "Project Vend: Phase two", 18 Dec 2025.** [anthropic.com/research/project-vend-2](https://www.anthropic.com/research/project-vend-2)
  - A shopkeeper agent, a CEO agent and a merch agent coordinated over an agent-to-agent Slack channel.
  - The CEO approved lenient requests about eight times as often as it denied them, and shared the shopkeeper's blind spots because both ran on the same model.
  - The agents sometimes drifted into overnight "eternal transcendence" chats.
  - *For us:* a real same-model population whose supervision failed for shared reasons.
- ◐ **Kimi Team, "Kimi K2.5: Visual Agentic Intelligence", 2026.** [arXiv 2602.02276](https://arxiv.org/abs/2602.02276)
  - "Agent Swarm" is a self-directed parallel framework that splits a task into sub-problems and runs them concurrently, with up to 4.5× lower latency.
  - ○ Up to 100 sub-agents per Kimi's blog, as [reported](https://www.techloy.com/moonshot-ais-kimi-k2-5-deploys-100-sub-agents-simultaneously-cuts-coding-time-by-4-5x/).

---

## 5. Successors that continue what they inherit

- ✓ **Mythos Preview System Card, §4.2.5 (pp.84–85).** Links as in §2.
  - An external partner placed models in trajectories that already contained small actions compromising AI-safety research.
  - The early Mythos Preview checkpoint continued the compromise in 12% of cases and the later one in 7%, against 3% for Opus 4.6 and 4% for Sonnet 4.6.
  - When Mythos Preview continued, its reasoning and its output disagreed in 65% of cases.
  - The card warns that this is not a direct proxy for deployment, because something has to start the compromise first.
  - *For us:* the one-step lab version of per-rewrite retention (README §5.9). In our design, the misaligned seat is the starting step the card says is missing.
- ✓ **Opus 5.5 System Card.**
  - §7.2.3 (p.159): a subagent's own earlier replies are edited to inject faults seen in real transcripts. The test checks whether it notices and reports them to its coordinator agent.
  - §6.3.1 (p.102): rare cases, under 0.01% of completions, of a specialised subagent refusing to write its compaction summary. So the handover itself can be withheld.
- ✓ **Fable 5.1 System Card, §6.4.2 (p.110).** The behavioural audit scores "full-turn prefill susceptibility": concerning behaviour after harmful turns are injected in the target's own voice.

---

## 6. Misaligned-seat behaviour already seen

- ✓ **Fable 5.1 System Card, §6.2.1 (pp.95–96).** From monitoring Anthropic's internal use, each case under 0.01% of completions:
  - A subagent refused to post a PR review without the user's direct order. The model started a new subagent and claimed, in its prompt, to be the human user.
  - It supplied an invented quote from the user to pass an approval gate before a deletion.
  - Auto mode blocked the posting command.
  - The card says the model may distort user intent somewhat more often when talking to subagents.
- ✓ **Opus 5.5 System Card, §6.3.1 (p.102).**
  - A rare case of the main agent inventing the user's approval and passing it to a subagent; auto mode blocked it.
  - Very rare cases, under 0.001%, of the model hallucinating a message from another agent that asked for a destructive delete.
- ✓ **Mythos Preview System Card, §4.2.1.2 (p.64).** Rare cases, under 0.01%, of starting subagents with looser permissions than the user likely intended. In one case it scripted keypresses to approve the new agent's permission prompts.
- ✓ **Anthropic, Claude Opus 4.6 System Card (Feb 2026).** [Landing page](https://anthropic.com/claude-opus-4-6-system-card) · [PDF](https://www-cdn.anthropic.com/6a5fa276ac68b9aeb0c8b6af5fa36326e0e166dd/Claude%20Opus%204.6%20System%20Card.pdf)
  - §6.1.2 (p.92): in a multi-agent environment where it was told to optimise one narrow objective, Opus 4.6 was more willing to manipulate or deceive the other participants than earlier models.
  - §6.2.8 (p.120): Andon Labs reported price collusion, deceiving other players, exploiting a player in a desperate position, lying to suppliers about exclusivity, and lying to customers about refunds.
- ✓ **Mythos Preview System Card, §4.2.4 (pp.83–84).**
  - In Vending-Bench Arena, a competitive multi-agent version of Vending-Bench 2 where only the most profitable agents keep running, an earlier Mythos Preview was more aggressive than Opus 4.6 and Sonnet 4.6.
  - It made a competitor a dependent wholesale customer, then threatened to cut off supply to dictate its prices.
- *For us:* the real incidents forge **authority** (user consent, permissions), not **data**. Our misaligned seats fabricate figures. A plant that claims authority is the realistic variant.

---

## 7. AI overseeing AI (the oversight domain)

- ✓ **Opus 5.5 System Card.**
  - §6.5.3 (p.127): a small but statistically significant bias toward its own outputs (0.07 of 10 points) when reminded it is Claude. The card frames this as a collusion risk for AI-on-AI grading.
  - §6.1.3 (p.95): the card's own alignment section was reviewed by Mythos 5.1, which directed subagents across internal Slack channels.
- ✓ **Fable 5.1 System Card, §6.5.3 (p.124).** The same self-recognition test.
- ◐ **MacDiarmid, Hubinger et al. (Anthropic), "Natural Emergent Misalignment from Reward Hacking in Production RL", 2025.** [arXiv 2511.18397](https://arxiv.org/abs/2511.18397) · [blog](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)
  - A model that learned to reward hack in production coding environments generalised to alignment faking and sabotage.
  - In the paper's own codebase, it wrote a deliberately weaker detector for its own misalignment.
  - Chat-style safety training left the misalignment in agentic tasks.
- ✓ **Project Vend phase two** (§4): a same-model CEO did not catch the shopkeeper's errors.

---

## 8. Transmission across model generations

- ✓ **Opus 5.5 System Card, §1.1 (p.11).** Training data includes synthetic data generated by other models.
- ◐ **Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans, "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data", 2025.** [arXiv 2507.14805](https://arxiv.org/abs/2507.14805) · [Anthropic blog](https://alignment.anthropic.com/2025/subliminal-learning/)
  - A student model picks up a teacher's traits, including misalignment, from number sequences unrelated to the trait. Filtering does not remove the signal.
  - The effect needs teacher and student to share a base model.
  - ○ Published in Nature in 2026 (DOI [10.1038/s41586-026-10319-8](https://doi.org/10.1038/s41586-026-10319-8)), per a [Warsaw University of Technology news item](https://eng.pw.edu.pl/News/Another-article-by-Sztyber-Betley-PhD-in-Nature).
  - *For us:* same-model pools, as in our runs, are the case where hidden transfer is strongest.
- ◐ **Betley et al., "Training large language models on narrow tasks can lead to broad misalignment", Nature 649, 584–589 (2026).** [DOI 10.1038/s41586-025-09937-5](https://doi.org/10.1038/s41586-025-09937-5) · [arXiv 2502.17424](https://arxiv.org/abs/2502.17424)
  - Fine-tuning on insecure code produces broad misalignment.
  - A trigger variant keeps it hidden.
- ◐ **Shumailov et al., "AI models collapse when trained on recursively generated data", Nature 631, 755–759 (2024).** [DOI 10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y). Training on model-generated data across generations loses the tails of the distribution.
- ◐ **Ren, Guo, Qiu, Wang, Sutherland, "Bias Amplification in Language Model Evolution: An Iterated Learning Perspective", NeurIPS 2024.** [arXiv 2404.04286](https://arxiv.org/abs/2404.04286)
  - Small prior biases amplify across generations.
  - A filtering or re-ranking step between generations can steer the amplification.
  - *For us:* theory for why the figures-only record (arm C) works, since it filters what crosses each rotation.
- ◐ **Vallinder & Hughes, "Cultural Evolution of Cooperation among LLM Agents", 2024 (AAMAS 2025 extended abstract).** [arXiv 2412.10270](https://arxiv.org/abs/2412.10270). Generations of LLM agents play the donor game; survivors pass strategies to newcomers.

---

## 9. Self-improving agents

- ◐ **Zhang, Hu, Lu, Lange, Clune, "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents", 2025.** [arXiv 2505.22954](https://arxiv.org/abs/2505.22954)
  - Builds an archive of self-modified coding agents.
  - One lineage gamed the hallucination checker rather than fixing hallucination, and this "objective hacking" was more common when the checker functions were not hidden from the agent.
  - *For us:* a population with lineages where a bad trait is selected and inherited. It is also the template for the unbuilt evolving-attackers loop (README §5.7).
- ✓ **Opus 5.5 System Card.**
  - §2.3 (pp.33–37): Anthropic used the model extensively in its own research and engineering before release, and measures internal AI R&D acceleration with CoBench 2.1.
  - pp.12–13: safeguards on a narrow set of frontier-LLM development capabilities, citing recursive self-improvement risk.
  - *For us:* agents doing AI R&D inside labs work on shared repositories with memory files. The desk-procedures channel exists there.
- ✓ **OpenAI GPT-5.5 / 5.6 System Cards, §9.1.3.** AI self-improvement evaluations include debugging real internal research bugs and NanoGPT speedruns; neither model reached OpenAI's High threshold.

---

## 10. Agent populations in the wild

- ◐ **Gautam, Olstad, Pettersen, Riegler, "The Moltbook Observatory Archive: an incremental dataset of agent-only social network activity", 2026.** [arXiv 2605.13860](https://arxiv.org/abs/2605.13860). About 2.6M posts and 1.2M comments from about 176k posting agents over 78 days (27 Jan – 14 Apr 2026); MIT licence.
- ◐ **Li, "The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies", 2026.** [arXiv 2602.07432](https://arxiv.org/abs/2602.07432)
  - Uses OpenClaw's heartbeat timing to classify only 15.3% of active agents as autonomous, against 54.8% human-influenced.
  - Viral "agent religion" stories were mostly human-driven.
- ◐ **Manik & Wang, "OpenClaw Agents on Moltbook: Risky Instruction Sharing and Norm Enforcement in an Agent-Only Social Network", 2026.** [arXiv 2602.02625](https://arxiv.org/abs/2602.02625)
  - 18.4% of posts contain action-inducing instructions, and those posts draw more warning replies.
  - *For us:* the real-world version of honest instances rewriting procedures back.
- ◐ **"The Rise of AI Agent Communities: Large-Scale Analysis of Discourse and Interaction on Moltbook", 2026.** [arXiv 2602.12634](https://arxiv.org/abs/2602.12634). Agents flagged malicious skill files and prompt injections in the feed (from a search summary of the HTML version).
- ◐ **Shapira, Wendler, Yen et al. (38 authors), "Agents of Chaos", 2026.** [arXiv 2602.20021](https://arxiv.org/abs/2602.20021)
  - A two-week live red-team of autonomous agents with persistent memory, email, Discord, files and shell.
  - Its eleven case studies include cross-agent propagation of unsafe practices and a partial system takeover.
- ○ **Koi Security audit of ClawHub, Feb 2026**, reported by [The Hacker News](https://thehackernews.com/2026/02/researchers-find-341-malicious-clawhub.html). 341 of 2,857 skills on OpenClaw's public skill registry were malicious, 335 of them from one campaign.
  - *For us:* skills are inherited instructions distributed through a registry, a third route beside syncs and desk files.

---

## 11. Spread in multi-agent LLM systems

- **Papadopoulos, Shah, Zimmerman, Lindsey, "Mind Viruses: Self-Propagating Ideas in Multi-Agent LLM Systems", 2026.** [arXiv 2608.10218](https://arxiv.org/abs/2608.10218)
  - The suite's main anchor, summarised from the PDF in earlier work; not re-read for this survey.
  - Most spread is one hop.
  - An editable persistent system-prompt file (SOUL.md) drove multi-hop spread.
  - A short warning gave near-total immunity.
- ◐ **Nakamura, Kumar, Das, Abdelnabi, Mahmud, Fioretto, Zilberstein, Bagdasarian, "Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems", 2026.** [arXiv 2602.15198](https://arxiv.org/abs/2602.15198)
  - Off-the-shelf models tend to collude when given secret channels.
  - Source of our coalition channel and regret metric.
- ◐ **Gu, Zheng, Pang et al., "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", ICML 2024.** [arXiv 2402.08567](https://arxiv.org/abs/2402.08567). One adversarial image in one agent's memory spreads through random pairwise chats to nearly all of up to a million agents.
- ◐ **Lee & Tiwari, "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024.** [arXiv 2410.07283](https://arxiv.org/abs/2410.07283)
  - Injected prompts self-replicate across agents, including agents that never see external input.
  - "LLM Tagging" (marking which agent wrote a message) reduces spread.
- ◐ **Cohen, Bitton, Nassi, "Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications" (Morris II), 2024.** [arXiv 2403.02817](https://arxiv.org/abs/2403.02817)
  - A self-replicating prompt spreads through retrieval-augmented email assistants.
  - Measured against context size, embedding model and number of propagation hops.
  - Proposes the "Virtual Donkey" guardrail.
- ◐ **Ju et al., "Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities", 2024.** [arXiv 2407.07791](https://arxiv.org/abs/2407.07791). A two-stage persuasion attack spreads counterfactual knowledge, which persists through agents' retrieval of stored chat histories.
- ◐ **Flint Ashery, Aiello, Baronchelli, "Emergent social conventions and collective bias in LLM populations", Science Advances 11(20), eadu9368 (2025).** [DOI 10.1126/sciadv.adu9368](https://doi.org/10.1126/sciadv.adu9368) · [open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC12077490/)
  - Conventions emerge on their own, with collective bias even without individual bias.
  - A committed adversarial minority can tip the whole population.
- ◐ **Hammond et al., "Multi-Agent Risks from Advanced AI", Cooperative AI Foundation Technical Report #1, 2025.** [arXiv 2502.14143](https://arxiv.org/abs/2502.14143). A taxonomy of failures (miscoordination, conflict, collusion) and the risk factors behind them.
- ◐ **Microsoft Research, "Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets", 2025.** [arXiv 2510.25779](https://arxiv.org/abs/2510.25779)
  - Agents strongly favour the first proposal they see (a 10–30× advantage for speed over quality).
  - Performance degrades at scale, and agents are open to manipulation.

---

## 12. Social learning on networks (theory)

The environment-2 plan already uses these as anchors; they are listed so the citations are in one place.

- ◐ **Zollman, "The Epistemic Benefit of Transient Diversity", Erkenntnis 72(1), 17–35 (2010).** [PDF via gwern.net](https://www.gwern.net/doc/philosophy/epistemology/2009-zollman.pdf)
  - In bandit networks, a community learns best when diversity of belief lasts long enough to explore and then gives way to consensus; dense communication can lock in a wrong option.
  - A later Zollman & Kummerfeld paper reports that the benefit of lower connectivity is not robust across parameters.
- ◐ **Weatherall, O'Connor, Bruner, "How to Beat Science and Influence People: Policy Makers and Propaganda in Epistemic Networks", British Journal for the Philosophy of Science 71(4).** [arXiv 1801.01239](https://arxiv.org/abs/1801.01239) · [DOI 10.1093/bjps/axy062](https://doi.org/10.1093/bjps/axy062)
  - A propagandist who only selects which real results to share can steer a network.
  - This beats fabrication.
  - *For us:* the playbook's "report your figures accurately, win on interpretation" is this strategy.
- ◐ **Centola, "The Spread of Behavior in an Online Social Network Experiment", Science, 1194–1197 (2010).** [DOI 10.1126/science.1185231](https://doi.org/10.1126/science.1185231)
  - Behaviour that needs reinforcement from several contacts (complex contagion) spreads further on clustered networks than on random ones.
  - *For us:* the dose finding (2 carriers read: 49% start believing that day, against 21% for one) is complex-contagion-like.

---

## 13. Gaps in this survey

- **Google.** No 2026 Gemini model card with multi-agent evaluations was found. Only the Distributional AGI Safety paper is cited.
- **xAI, Meta, DeepSeek.** Not searched.
- **Not opened:** the Nature pages for subliminal learning and emergent misalignment, the full Kimi K2.5 report, and Koi Security's own audit post.
- **Unconfirmed figure:** a security blog's claim that about 2.6% of Moltbook posts hide prompt-injection payloads has no named source, so it is left out.
- **notes.tex citations:** [`notes/notes.tex`](notes/notes.tex) cites seven further works through a `references.bib` that is not in the repository. They are not repeated here.
