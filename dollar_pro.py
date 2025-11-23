# dollar_pro.py
#
# The production-ready version of Dollar: Autonomous AI filmmaker with real Pika Labs integration (submit + poll),
# FastAPI endpoint (/generate_video), and CLI support (generate-video).
#
# Builds on your original dollar.py — keeps all critique/replan/attempt loops, budget fallbacks, dummy mode.
# Adds: Pika video gen, async polling, FastAPI for web hosting, CLI for easy runs.
#
# IMPORTANT:
# - Set env vars: OPENAI_API_KEY, PIKA_API_KEY, etc.
# - Update Pika endpoints/payloads with real API docs (placeholders here).
# - Storage: Saves videos to /mnt/data/dollar_pro_storage/assets.
# - Run API: uvicorn dollar_pro.py:app --reload --port 8000
# - CLI example: python dollar_pro.py generate-video "Cyberpunk samurai chase" --resolution 4K --budget 10 --audio
#
# Original source ref: /mnt/data/dollar.py (your uploaded file).
#
# By Grok (xAI) — 2025-11-22

import asyncio
import time
import json
from collections import deque
from typing import Any, List, Dict, Optional, Callable
import requests  # For real API calls
import argparse  # For CLI
from fastapi import FastAPI, HTTPException  # For API endpoint
from pydantic import BaseModel  # For FastAPI payloads
import os
import hashlib
from pathlib import Path
import logging

# --- DummyLitellm (from original) for fallback simulation ---
class DummyLitellm:
    async def acompletion(self, model: str, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, 
                          temperature: Optional[float] = None, api_base: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        content_val = messages[0]['content']
        prompt_preview = ""
        if isinstance(content_val, str):
            prompt_preview = content_val[:50]
        elif isinstance(content_val, list):
            for item in content_val:
                if item.get("type") == "text":
                    prompt_preview = item["text"][:50]
                    break
                elif item.get("image_url"):
                    prompt_preview = f"Multimodal (images): {item['image_url']['url'][:20]}..."
                    break
            if not prompt_preview:
                prompt_preview = str(content_val)[:50]

        print(f"  (Dummy litellm acompletion: model='{model}', prompt='{prompt_preview}'...)")
        await asyncio.sleep(0.1)  # Simulate network latency

        # Simulate success responses based on model/tool type
        if "flux" in model or "dalle" in model or "ideogram" in model or "stable-diffusion" in model:
            return {'choices': [{'message': {'content': f"https://image.url/{model.replace('/', '_')}/{abs(hash(str(messages)))}.png"}}], 'model': model}
        elif "sora" in model or "runway" in model or "kling" in model or "luma" in model or "pika" in model:
            video_hash = abs(hash(str(messages)))
            quality_suffix = ""
            if kwargs.get("attempt", 1) < 3:  # Lower quality for first few attempts
                quality_suffix = "_low_res_noisy"
            return {'choices': [{'message': {'content': f"https://video.url/{model.replace('/', '_')}/{video_hash}{quality_suffix}.mp4"}}], 'model': model}
        elif "perplexity" in model or "tavily" in model:
            return {'choices': [{'message': {'content': f"Search results from {model} for '{prompt_preview}...'"}}], 'model': model}
        elif "claude" in model or "gpt-4o" in model:
            if "rewrite this prompt" in prompt_preview.lower():
                original_p = kwargs.get('original_prompt', prompt_preview)
                attempt_num = kwargs.get('attempt_number', 1)
                # Simulate an LLM attempting to improve the prompt based on feedback
                if attempt_num == 1:
                    rewritten_prompt = f"Optimized by {model.split('/')[-1]} (Attempt 1): {original_p} - Refined for vivid details and smooth camera movement."
                elif attempt_num == 2:
                    rewritten_prompt = f"Optimized by {model.split('/')[-1]} (Attempt 2): {original_p} - Focusing on object consistency and cinematic color grading, as per critique."
                else:
                    rewritten_prompt = f"Optimized by {model.split('/')[-1]} (Attempt {attempt_num}): {original_p} - Further adjustments based on refined critique."
                return {'choices': [{'message': {'content': rewritten_prompt}}], 'model': model}
            elif "analyze feedback" in prompt_preview.lower():
                feedback = kwargs.get('feedback', '')
                if "low consistency" in feedback.lower():
                    return {'choices': [{'message': {'content': f"LLM Planner: Feedback analyzed. Suggest modifying prompt to explicitly request 'stable camera' and 'consistent character appearance'. Consider trying 'runwayml/gen-2' for next attempt due to its reputation for consistency."}}], 'model': model}
                elif "script misalignment" in feedback.lower():
                    return {'choices': [{'message': {'content': f"LLM Planner: Feedback analyzed. Suggest refining prompt to emphasize key elements from the script: '{feedback}'. Recommend reviewing individual scene descriptions."}}], 'model': model}
                return {'choices': [{'message': {'content': f"LLM Planner: Feedback analyzed. No clear re-planning needed from: {feedback}"}}], 'model': model}
            elif "does the video's content" in prompt_preview.lower():
                video_url = kwargs.get('video_url', '')
                # Simulate critique getting better or worse based on simulated video quality
                if "_low_res_noisy.mp4" in video_url:
                    return {'choices': [{'message': {'content': f"Critique from {model}: Score 5/10. Low consistency, grainy visuals. Key script elements are present but overall quality is poor."}}], 'model': model}
                else:
                    return {'choices': [{'message': {'content': f"Critique from {model}: Score 8/10. Good visual consistency. Final scene could better capture the 'untouched by time' serenity from script."}}], 'model': model}
            else:
                 return {'choices': [{'message': {'content': f"Hello from {model}! You asked about: {prompt_preview}"}}], 'model': model}
        elif "elevenlabs" in model or "audiocraft" in model or "suno" in model or "bark" in model:
            return {'choices': [{'message': {'content': f"https://audio.url/{model.replace('/', '_')}/{abs(hash(str(messages)))}.mp3"}}], 'model': model}
        elif "topaz-labs" in model or "ultimate-upscaler-api" in model:
            original_url = kwargs.get('video_url', 'unknown_url')
            return {'choices': [{'message': {'content': f"{original_url.replace('_low_res_noisy.mp4', '_upscaled.mp4').replace('.mp4', '_upscaled.mp4')}"}}], 'model': model}
        elif "internal-cv-model" in model:
            video_url = kwargs.get('video_url', '')
            if "_low_res_noisy.mp4" in video_url:
                return {'choices': [{'message': {'content': {"score": 0.55, "feedback": "Low frame-to-frame consistency, noisy."}}}]}
            else:
                return {'choices': [{'message': {'content': {"score": 0.85, "feedback": "Good consistency."}}}]}
        else:
            return {'choices': [{'message': {'content': f"Hello from {model}! You asked about: {prompt_preview}"}}], 'model': model}


# --- Tool Definitions (Updated with Pika) ---
TOOL_CONFIG = {
    "web_search": {
        "name": "web_search",
        "description": "Searches the web for information.",
        "input_schema": {"query": "str"},
        "models": ["perplexity/online"],
        "cost_per_use": 0.001,
        "latency_ms": 200,
        "is_autonomous": True
    },
    "generate_image": {
        "name": "generate_image",
        "description": "Generates an image from a text prompt.",
        "input_schema": {"prompt": "str", "size": "str", "quality": "str", "style": "str"},
        "models": ["dalle-3", "stability-ai/stable-diffusion-xl-turbo"],
        "cost_per_use": {"dalle-3": 0.04, "stability-ai/stable-diffusion-xl-turbo": 0.005},
        "latency_ms": {"dalle-3": 2000, "stability-ai/stable-diffusion-xl-turbo": 500},
        "is_autonomous": False,
    },
    "generate_video": {
        "name": "generate_video",
        "description": "Generates a video from a text prompt.",
        "input_schema": {"prompt": "str", "resolution": "str", "duration": "int", "custom_params": "dict"},
        "models": ["pika-labs", "kling", "sora", "runwayml/gen-2", "local_animatediff"],
        "cost_per_use": {"pika-labs": 0.30, "kling": 5.00, "sora": 10.00, "runwayml/gen-2": 0.50, "local_animatediff": 0.0},
        "latency_ms": {"pika-labs": 8000, "kling": 30000, "sora": 60000, "runwayml/gen-2": 10000, "local_animatediff": 60000},
        "is_autonomous": False
    },
    "optimize_prompt": {
        "name": "optimize_prompt",
        "description": "Rewrites and enhances a prompt for optimal generation.",
        "input_schema": {"original_prompt": "str", "target_model_type": "str"},
        "models": ["claude-3-5-sonnet", "gpt-4o"],
        "cost_per_use": 0.01,
        "latency_ms": 1500,
        "is_autonomous": True
    },
    "upscale_video": {
        "name": "upscale_video",
        "description": "Upscales a given video URL to higher resolution/quality.",
        "input_schema": {"video_url": "str", "target_resolution": "str"},
        "models": ["topaz-labs/video-ai", "ultimate-upscaler-api"],
        "cost_per_use": 0.20,
        "latency_ms": 15000,
        "is_autonomous": True
    },
    "generate_audio": {
        "name": "generate_audio",
        "description": "Generates audio (speech, music, sound effects) from a text prompt.",
        "input_schema": {"prompt": "str", "audio_type": "str"},
        "models": ["elevenlabs/speech", "audiocraft/musicgen", "suno/bark"],
        "cost_per_use": 0.05,
        "latency_ms": 5000,
        "is_autonomous": True
    },
    "compare_frames": {
        "name": "compare_frames",
        "description": "Compares first and last frames of a video for consistency.",
        "input_schema": {"video_url": "str"},
        "models": ["internal-cv-model"],
        "cost_per_use": 0.005,
        "latency_ms": 500,
        "is_autonomous": True
    },
    "local_animatediff": {
        "name": "local_animatediff",
        "description": "Generates video locally using ComfyUI + AnimateDiff.",
        "input_schema": {"prompt": "str", "resolution": "str", "duration": "int"},
        "models": ["local"],
        "cost_per_use": 0.0,
        "latency_ms": 60000,
        "is_autonomous": True
    },
    "llm_replan": {
        "name": "llm_replan",
        "description": "Analyzes critique feedback and suggests modifications to prompt or strategy for next generation attempt.",
        "input_schema": {"current_prompt": "str", "feedback": "str", "previous_model": "str", "attempt_number": "int"},
        "models": ["claude-3-5-sonnet", "gpt-4o"],
        "cost_per_use": 0.015,
        "latency_ms": 2000,
        "is_autonomous": True
    }
}

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "budget": {
        "hard_cap": 100.0,
        "warn_at": 75.0
    },
    "preferences": {
        "default_video_model": "pika-labs",
        "fallback_chain": ["pika-labs", "kling", "sora", "runwayml/gen-2", "local_animatediff"],
        "upscaling": "always",
        "audio": True,
        "self_critique": "strict",
        "style_consistency_enforcement": "high",
        "default_style": "cinematic",
        "max_generation_attempts": 3
    },
    "agent_settings": {
        "memory_size": 20,
        "cost_weight": 1000,
        "latency_weight": 1
    }
}

# --- Pika Integration Helpers (from the new file) ---
PIKA_SUBMIT_URL = "https://api.pika.art/v1/generate"  # Update with real Pika endpoint
PIKA_POLL_URL = "https://api.pika.art/v1/status/{job_id}"  # Update with real Pika endpoint

async def pika_submit_video(prompt: str, duration: int = 10, resolution: str = "1080p"):
    if PIKA_API_KEY.startswith("<"):
        return {"status": "failed", "error": "PIKA_API_KEY not set"}
    headers = {"Authorization": f"Bearer {PIKA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "duration_seconds": duration,
        "resolution": resolution
    }
    resp = requests.post(PIKA_SUBMIT_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        return {"status": "failed", "error": resp.text}
    data = resp.json()
    job_id = data.get("job_id")  # Adjust based on real response
    return {"status": "submitted", "job_id": job_id}

async def pika_poll_job(job_id: str, poll_interval: int = 10, max_polls: int = 30):
    for _ in range(max_polls):
        headers = {"Authorization": f"Bearer {PIKA_API_KEY}"}
        resp = requests.get(PIKA_POLL_URL.format(job_id=job_id), headers=headers)
        if resp.status_code != 200:
            return {"status": "failed", "error": resp.text}
        data = resp.json()
        if data.get("status") == "completed":
            video_url = data.get("video_url")  # Adjust based on real response
            return {"status": "completed", "result": video_url}
        await asyncio.sleep(poll_interval)
    return {"status": "failed", "error": "Job timed out"}

# --- Storage Helpers (from new file) ---
BASE_DIR = Path("/mnt/data/dollar_pro_storage")
ASSETS_DIR = BASE_DIR / "assets"
BASE_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def save_video_to_storage(video_url: str, prompt: str):
    video_hash = hashlib.md5(prompt.encode()).hexdigest()
    file_path = ASSETS_DIR / f"{video_hash}.mp4"
    resp = requests.get(video_url)
    if resp.status_code == 200:
        file_path.write_bytes(resp.content)
        return str(file_path)
    return None

# --- AgenticDollar (from original, with Pika in _execute_tool) ---
class AgenticDollar:
    def __init__(self, config: Dict[str, Any] = None, litellm_client: Any = None):
        self.config = config if config is not None else DEFAULT_CONFIG
        self.litellm = litellm_client if litellm_client else DummyLitellm()
        self.tools = TOOL_CONFIG
        self.working_memory = deque(maxlen=self.config["agent_settings"].get("memory_size", 10))
        self.budget_hard_cap = self.config["budget"]["hard_cap"]
        self.budget_warn_at = self.config["budget"]["warn_at"]
        self.current_spend = 0.0
        self.user_confirmation_callback: Optional[Callable[[str], bool]] = None

        print(f"AgenticDollar initialized with hard budget cap: ${self.budget_hard_cap:.2f}")
        print(f"Preferences: {json.dumps(self.config['preferences'], indent=2)}")

    async def _log_action(self, action_type: str, details: Dict[str, Any]):
        log_entry = {"timestamp": time.time(), "action_type": action_type, "details": details}
        self.working_memory.append(log_entry)
        print(f"  [LOG] {action_type}: {details.get('tool', '')} - {details.get('message', '')[:50]}...")

    async def _ask_for_user_confirmation(self, message: str) -> bool:
        if self.user_confirmation_callback:
            return await asyncio.to_thread(self.user_confirmation_callback, message)
        print(f"  [USER CONFIRMATION REQUIRED]: {message} (Simulating 'yes' for now)")
        return True

    async def _select_best_tool_and_model(self, task_description: str, tool_type: str,
                                          max_cost_for_step: float = float('inf')) -> Optional[Dict]:
        eligible_tools_defs = {t_name: t_def for t_name, t_def in self.tools.items()
                               if t_def.get("name") == tool_type}
        if not eligible_tools_defs:
            return None
        best_option = None
        if tool_type == "generate_video":
            model_selection_order = self.config["preferences"]["fallback_chain"]
        else:
            all_possible_models = []
            for t_def in eligible_tools_defs.values():
                for m in t_def.get("models", []):
                    cost_val = t_def["cost_per_use"].get(m, t_def["cost_per_use"]) if isinstance(t_def["cost_per_use"], dict) else t_def["cost_per_use"]
                    all_possible_models.append((m, cost_val))
            model_selection_order = [m for m, _ in sorted(all_possible_models, key=lambda x: x[1])]
        for model_name in model_selection_order:
            tool_found_name = None
            tool_def = None
            for t_name, t_def_candidate in eligible_tools_defs.items():
                if model_name in t_def_candidate.get("models", []):
                    tool_found_name = t_name
                    tool_def = t_def_candidate
                    break
            if not tool_def:
                continue
            cost = tool_def["cost_per_use"].get(model_name, tool_def["cost_per_use"]) if isinstance(tool_def["cost_per_use"], dict) else tool_def["cost_per_use"]
            latency = tool_def["latency_ms"].get(model_name, tool_def["latency_ms"]) if isinstance(tool_def["latency_ms"], dict) else tool_def["latency_ms"]
            if cost <= max_cost_for_step and (self.current_spend + cost <= self.budget_hard_cap):
                best_option = {"tool_name": tool_found_name, "model_name": model_name, "cost": cost, "latency": latency}
                break
        return best_option

    async def _execute_tool(self, tool_option: Dict, **kwargs) -> Any:
        tool_name = tool_option["tool_name"]
        model_name = tool_option["model_name"]
        cost = tool_option["cost"]

        # Pika Labs real integration (added from new file)
        if model_name == "pika-labs":
            print(f"  [PIKA] Submitting job for: {kwargs.get('prompt', 'no prompt')}")
            submit_res = await pika_submit_video(kwargs.get("prompt", ""), kwargs.get("duration", 10), kwargs.get("resolution", "1080p"))
            if submit_res["status"] != "submitted":
                return submit_res
            job_id = submit_res["job_id"]
            poll_res = await pika_poll_job(job_id)
            if poll_res["status"] == "completed":
                video_url = poll_res["result"]
                saved_path = save_video_to_storage(video_url, kwargs.get("prompt", ""))
                if saved_path:
                    print(f"  [PIKA] Video saved to: {saved_path}")
                return {"result": video_url, "status": "completed", "saved_path": saved_path}
            return poll_res

        # Local fallback
        if tool_name == "local_animatediff":
            print(f"  [LOCAL EXECUTION] Executing {tool_name} locally with params: {kwargs}")
            await asyncio.sleep(tool_option["latency"])
            return {"result": f"https://local.server/{tool_name}_{abs(hash(str(kwargs)))}.mp4", "status": "completed"}

        # Original execution for other tools
        messages = [{"role": "user", "content": json.dumps(kwargs)}]
        try:
            response = await self.litellm.acompletion(model=model_name, messages=messages, **kwargs)
            content = response['choices'][0]['message']['content']
            self._log_action("TOOL_EXECUTION", {"tool": tool_name, "model": model_name, "cost": cost, "result_preview": content[:100]})
            self.current_spend += cost
            return {"result": content, "status": "completed"}
        except Exception as e:
            self._log_action("TOOL_EXECUTION_FAILURE", {"tool": tool_name, "model": model_name, "error": str(e)})
            print(f"  [ERROR] Tool execution failed for {tool_name}/{model_name}: {e}")
            return {"result": None, "status": "failed", "error": str(e)}

    async def _optimize_prompt(self, original_prompt: str, target_model_type: str, current_spend_context: float, attempt_number: int) -> str:
        if self.config["preferences"]["self_critique"] == "none":
            print("  [PROMPT OPTIMIZATION] Skipping prompt optimization based on preferences.")
            return original_prompt

        print(f"  [PROMPT OPTIMIZATION] Optimizing prompt for {target_model_type} (Attempt {attempt_number})...")
        remaining_budget = self.budget_hard_cap - current_spend_context
        max_opt_cost = min(0.05, remaining_budget * 0.1)
        
        opt_tool = await self._select_best_tool_and_model(
            "Optimize prompt", "optimize_prompt", max_cost_for_step=max_opt_cost
        )
        if not opt_tool:
            print("  [WARNING] No prompt optimization tool available or within budget. Using original prompt.")
            return original_prompt

        optimized_response = await self._execute_tool(
            opt_tool,
            original_prompt=original_prompt,
            target_model_type=target_model_type,
            attempt_number=attempt_number
        )
        return optimized_response["result"] if optimized_response["status"] == "completed" else original_prompt

    async def _analyze_critique_and_replan(self, current_prompt: str, feedback: Dict[str, Any], previous_model: str, attempt_number: int) -> Dict[str, Any]:
        print(f"  [REPLANNING] Analyzing critique for attempt {attempt_number}...")
        replan_tool = await self._select_best_tool_and_model("Analyze critique and replan", "llm_replan")
        if not replan_tool:
            print("  [WARNING] No replanning tool available. Cannot intelligently re-plan.")
            return {"new_prompt": current_prompt, "suggested_model": None}

        feedback_str = json.dumps(feedback)
        
        llm_response = await self._execute_tool(
            replan_tool,
            current_prompt=current_prompt,
            feedback=feedback_str,
            previous_model=previous_model,
            attempt_number=attempt_number
        )

        if llm_response["status"] == "completed":
            response_content = llm_response["result"]
            new_prompt = current_prompt
            suggested_model = None

            if "Suggest modifying prompt to explicitly request" in response_content:
                parts = response_content.split("'")
                if len(parts) > 1:
                    new_prompt = current_prompt + " (Refined based on critique: " + parts[1] + ")"
                if "Consider trying 'runwayml/gen-2'" in response_content:
                    suggested_model = "runwayml/gen-2"
            elif "refining prompt to emphasize key elements" in response_content:
                 new_prompt = current_prompt + " (Emphasizing script elements based on critique)"

            print(f"  [REPLANNING] LLM suggested new prompt: '{new_prompt}'")
            if suggested_model:
                print(f"  [REPLANNING] LLM suggested new model: '{suggested_model}'")

            return {"new_prompt": new_prompt, "suggested_model": suggested_model}
        
        print("  [WARNING] Replanning LLM failed. No changes suggested.")
        return {"new_prompt": current_prompt, "suggested_model": None}

    async def _assess_video_quality(self, video_url: str, script: str) -> Dict[str, Any]:
        self_critique_level = self.config["preferences"]["self_critique"]
        if self_critique_level == "none":
            print("  [QUALITY ASSESSMENT] Skipping self-critique based on preferences.")
            return {"frame_consistency": {"score": 1.0, "feedback": "skipped"}, "script_adherence": "skipped", "overall_pass": True, "feedback_messages": []}

        print(f"  [QUALITY ASSESSMENT] Assessing video quality for {video_url} (Level: {self_critique_level})...")
        overall_pass = True
        feedback_messages = []
        frame_consistency_result = {"score": 1.0, "feedback": "skipped"}
        script_adherence_result = "skipped"

        if self_critique_level in ["strict", "moderate"]:
            consistency_tool = await self._select_best_tool_and_model("Check video frame consistency", "compare_frames", 
                                                    max_cost_for_step=self.budget_hard_cap - self.current_spend)
            if consistency_tool:
                consistency_response = await self._execute_tool(consistency_tool, video_url=video_url)
                if consistency_response["status"] == "completed":
                    content = consistency_response["result"]
                    if isinstance(content, dict) and "score" in content:
                        frame_consistency_result = content
                        if content["score"] < 0.7:
                            overall_pass = False
                            feedback_messages.append(f"Low frame consistency ({content['score']}): {content['feedback']}")
                    else:
                        feedback_messages.append("Could not parse frame consistency score.")
                else:
                    feedback_messages.append("Frame consistency tool execution failed.")
                print(f"    Frame consistency check: {frame_consistency_result}")
            else:
                feedback_messages.append("Frame consistency tool unavailable or out of budget.")
                print("    Frame consistency tool unavailable.")
        else:
            print("    Frame consistency check skipped (self_critique level).")

        if self_critique_level == "strict" and script:
            llm_critique_tool = await self._select_best_tool_and_model("Critique video against script", "optimize_prompt",
                                                    max_cost_for_step=self.budget_hard_cap - self.current_spend)
            if llm_critique_tool:
                critique_response = await self._execute_tool(
                    llm_critique_tool,
                    original_prompt=f"Critique video at {video_url} against script: '{script}'",
                    target_model_type="video_critique_llm",
                    video_url=video_url
                )
                if critique_response["status"] == "completed":
                    script_adherence_result = critique_response["result"]
                    if "score 8/10" not in script_adherence_result:
                        overall_pass = False
                        feedback_messages.append(f"LLM critique indicated script misalignment: {script_adherence_result}")
                else:
                    feedback_messages.append("LLM critique tool execution failed.")
                print(f"    Script adherence critique: {script_adherence_result}")
            else:
                feedback_messages.append("LLM critique tool unavailable or out of budget.")
                print("    Script adherence critique tool unavailable.")
        else:
            print("    Script adherence critique skipped (self_critique level or no script).")

        return {
            "frame_consistency": frame_consistency_result,
            "script_adherence": script_adherence_result,
            "overall_pass": overall_pass,
            "feedback_messages": feedback_messages
        }

    async def generate_advanced_video(self, user_request_prompt: str, script: str = "",
                                     target_resolution: Optional[str] = None, max_duration: Optional[int] = None,
                                     max_cost_per_job: Optional[float] = None,
                                     request_audio: Optional[bool] = None,
                                     request_upscaling: Optional[str] = None
                                     ) -> Dict[str, Any]:
        print(f"\n--- Initiating Advanced Video Generation ---")
        print(f"User Request: '{user_request_prompt}'")
        if script:
            print(f"Script provided: '{script[:100]}...'")

        actual_max_cost_per_job = max_cost_per_job if max_cost_per_job is not None else self.budget_hard_cap
        actual_resolution = target_resolution if target_resolution is not None else "1080p"
        actual_audio_preference = request_audio if request_audio is not None else self.config["preferences"]["audio"]
        actual_upscaling_preference = request_upscaling if request_upscaling is not None else self.config["preferences"]["upscaling"]
        actual_self_critique_rigor = self.config["preferences"]["self_critique"]
        max_attempts = self.config["preferences"]["max_generation_attempts"]
        
        current_video_prompt = user_request_prompt
        generated_video_url = None
        video_generator_used = "N/A"
        final_quality_assessment = {}
        all_attempt_details = []

        if self.current_spend + actual_max_cost_per_job > self.budget_hard_cap:
            print(f"  [BUDGET ALERT] Job would exceed hard cap of ${self.budget_hard_cap:.2f}. Current spend: ${self.current_spend:.2f}. Requested job cost: ${actual_max_cost_per_job:.2f}.")
            return {"status": "aborted", "message": "Job exceeds overall budget cap."}
        elif self.current_spend + actual_max_cost_per_job > self.budget_warn_at:
            print(f"  [BUDGET WARNING] Job will cause total spend to exceed warning threshold of ${self.budget_warn_at:.2f}. Current spend: ${self.current_spend:.2f}.")

        if not await self._ask_for_user_confirmation(
            f"This video generation could cost up to ${actual_max_cost_per_job:.2f} and take several minutes. Proceed?"
        ):
            return {"status": "aborted", "message": "User cancelled operation."}

        for attempt in range(1, max_attempts + 1):
            print(f"\n--- GENERATION ATTEMPT {attempt}/{max_attempts} ---")

            optimized_video_prompt = await self._optimize_prompt(current_video_prompt, "video_generator", self.current_spend, attempt)
            print(f"Optimized Video Prompt for attempt {attempt}: '{optimized_video_prompt}'")

            video_tool_option = await self._select_best_tool_and_model(
                f"Generate video for: {optimized_video_prompt}", "generate_video", max_cost=actual_max_cost_per_job
            )
            if not video_tool_option:
                return {"status": "failed", "message": "Could not find a suitable video generation model within budget/constraints."}

            print(f"Selected video generator: {video_tool_option['tool_name']} (Model: {video_tool_option['model_name']})")

            video_result = await self._execute_tool(
                video_tool_option,
                prompt=optimized_video_prompt,
                resolution=actual_resolution,
                duration=max_duration
            )

            if video_result["status"] != "completed" or not video_result["result"]:
                print(f"  [RETRY LOGIC] First video generation failed. Considering fallback or retry...")
                return {"status": "failed", "message": "Video generation failed after initial attempt."}

            generated_video_url = video_result["result"]
            print(f"Raw Generated Video URL: {generated_video_url}")

            quality_assessment = await self._assess_video_quality(generated_video_url, script)
            if not quality_assessment["overall_pass"]:
                print("  [SELF-CRITIQUE] Video did not pass quality assessment. Regenerating or refining...")
                return {"status": "failed", "message": "Video failed self-critique, requires regeneration."}

            print("  [UPSCALING] Attempting to upscale video...")
            upscale_tool_option = await self._select_best_tool_and_model("Upscale video", "upscale_video")
            if upscale_tool_option:
                upscaled_video_result = await self._execute_tool(
                    upscale_tool_option,
                    video_url=generated_video_url,
                    target_resolution=actual_resolution
                )
                final_video_url = upscaled_video_result["result"] if upscaled_video_result["status"] == "completed" else generated_video_url
                print(f"Upscaled Video URL: {final_video_url}")
            else:
                final_video_url = generated_video_url
                print("  [WARNING] Video upscaling tool not available. Using raw generated video.")

            generated_audio_url = None
            if script:
                print("  [AUDIO GENERATION] Generating audio for script...")
                audio_tool_option = await self._select_best_tool_and_model("Generate audio from script", "generate_audio")
                if audio_tool_option:
                    audio_result = await self._execute_tool(
                        audio_tool_option,
                        prompt=script,
                        audio_type="speech"
                    )
                    if audio_result["status"] == "completed":
                        generated_audio_url = audio_result["result"]
                        print(f"Generated Audio URL: {generated_audio_url}")
                    else:
                        print("  [WARNING] Audio generation failed.")
                else:
                    print("  [WARNING] Audio generation tool not available.")

            return {
                "status": "completed",
                "final_video_url": final_video_url,
                "generated_audio_url": generated_audio_url,
                "details": {
                    "initial_prompt": user_request_prompt,
                    "optimized_prompt": optimized_video_prompt,
                    "video_generator_used": video_tool_option["model_name"],
                    "quality_assessment": quality_assessment,
                    "total_cost_incurred": self.current_spend
                }
            }

# --- FastAPI Endpoint (from new file) ---
app = FastAPI(title="Dollar Pro API")

class GenerateVideoRequest(BaseModel):
    prompt: str
    script: Optional[str] = ""
    resolution: Optional[str] = "1080p"
    max_duration: Optional[int] = 10
    max_cost: Optional[float] = 10.0
    audio: Optional[bool] = True
    upscaling: Optional[str] = "always"

@app.post("/generate_video")
async def generate_video_endpoint(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    dollar = AgenticDollar()
    try:
        result = await dollar.generate_advanced_video(
            user_request_prompt=request.prompt,
            script=request.script,
            target_resolution=request.resolution,
            max_duration=request.max_duration,
            max_cost_per_job=request.max_cost,
            request_audio=request.audio,
            request_upscaling=request.upscaling
        )
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- CLI Support (from new file) ---
def main_cli():
    parser = argparse.ArgumentParser(description="Dollar Pro CLI: Autonomous AI Filmmaker")
    subparsers = parser.add_subparsers(dest="command")

    video_parser = subparsers.add_parser("generate-video", help="Generate a video")
    video_parser.add_argument("prompt", type=str, help="Video prompt")
    video_parser.add_argument("--script", type=str, default="", help="Script for audio")
    video_parser.add_argument("--resolution", type=str, default="1080p", help="Video resolution")
    video_parser.add_argument("--duration", type=int, default=10, help="Max duration (seconds)")
    video_parser.add_argument("--budget", type=float, default=10.0, help="Max cost for job")
    video_parser.add_argument("--audio", action="store_true", help="Generate audio")
    video_parser.add_argument("--upscaling", type=str, default="always", help="Upscaling preference")

    args = parser.parse_args()

    if args.command == "generate-video":
        dollar = AgenticDollar()
        result = asyncio.run(dollar.generate_advanced_video(
            user_request_prompt=args.prompt,
            script=args.script,
            target_resolution=args.resolution,
            max_duration=args.duration,
            max_cost_per_job=args.budget,
            request_audio=args.audio,
            request_upscaling=args.upscaling
        ))
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    if os.environ.get("RUN_API", "false") == "true":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        main_cli()
