# Dollar

### The First Autonomous AI Filmmaker That Ships Nothing Until It's Proud

![Dollar Banner](https://via.placeholder.com/1200x300/0f0f0f/ffffff?text=Dollar+AI+-+Autonomous+Filmmaking+2026) <!-- Replace with a cool generated image later -->

Dollar doesn't just generate videos.

It **tries**.  
It **watches**.  
It **hates what it made**.  
It **rewrites the prompt**.  
It **switches from Kling to Runway**.  
It **tries again**.

And it keeps going—until the video is good.

All while respecting your budget. No $400 surprise Kling bills. No broken characters. Just cinema.

```bash
# Coming soon: pip install dollar-agent
git clone https://github.com/aaross1223-source/Dollar-AI-dollar
cd Dollar-AI-dollar
python dollar.py
Watch Dollar battle itself: Attempt 1 fails critique → Refines prompt → Switches models → Succeeds on Attempt 2. Under $6 total.
This is 2026.
Dollar just shipped it early.
Why Dollar?

Self-Critiquing Loop: Analyzes frame consistency + script adherence. Fails? Retry with smarter prompts.
LLM-Powered Replanning: Claude/GPT-4o reads failures → "Add 'stable camera' to prompt" → Suggests Runway for better consistency.
Budget-Aware Routing: Kling ($5) if you splurge; Pika ($0.30) fallback. Hard caps prevent bankruptcy.
Multi-Model Fallback Chain: kling → sora → runwayml/gen-2 → pika-labs → local_animatediff.
Prompt Evolution: Starts vague ("make it cool") → Evolves to cinematic masterpieces.
Post-Production Magic: Auto-upscale (Topaz), audio gen (ElevenLabs), full audit trail.
Dummy Mode Ready: No APIs needed—simulates everything. Plug in real LiteLLM keys for production.

Built for the 2025 video AI boom (Sora/Kling era). Fork it. Build your $99/month SaaS. Make movies.
Quickstart

Run the Demos:Bashpython dollar.py
Scenario 1: Cyberpunk samurai (refinement needed—watch the retry loop!).
Scenario 2: Serene forest with script (passes or one retry).

Customize:
Edit DEFAULT_CONFIG for your prefs (e.g., max_attempts: 5, default_style: "anime").
Add real API keys: Replace DummyLitellm with litellm.
CLI Coming: dollar "neon cat chase" --4k --cost 10 --audio.

Extend:
New Tools: Add to TOOL_CONFIG (e.g., IP-Adapter for style lock-in).
Local Mode: Set local_fallback_enabled: true—runs AnimateDiff offline.


Architecture Highlights

Agentic Core: AgenticDollar class orchestrates planning → execution → critique → replan.
Tools: Unified via LiteLLM—treats Kling like Claude.
Memory: Deque-based working memory for self-improvement.
Async Everything: Parallel retries, no blocking.

See dollar.py for the full blueprint (1,200+ lines of executable poetry).
Roadmap

 Real LiteLLM Integration + API Keys
 CLI (Typer/Click): dollar generate "prompt" --budget 5
 SaaS Boilerplate: FastAPI wrapper for hosted runs
 Community: Video style adapters, more critique metrics
 Self-Critique Loop (Live!)
 Budget Caps & Fallbacks (Battle-Tested)

License
MIT — Free to fork, sell, conquer. Built by aaross1223-source. From a phone. In 2025.
Star & Contribute
⭐ If Dollar sparks joy (or films).
🤝 PRs welcome: Fix a tool, add a model, demo a video.
💬 Issues: "How do I hook up real Sora?" → Ask away.
The future of filmmaking isn't a tool. It's a director.
And its name is Dollar.
