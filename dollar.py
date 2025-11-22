# dollar_agent_system_updated.py


import asyncio
import time
import json
from collections import deque
from typing import Any, List, Dict, Optional, Callable


# --- Dummy Litellm for Local Execution Simulation ---
# (Remains the same as in the previous complete code block)
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
        await asyncio.sleep(0.1) # Simulate network latency


        # Simulate success responses based on model/tool type
        if "flux" in model or "dalle" in model or "ideogram" in model or "stable-diffusion" in model:
            return {'choices': [{'message': {'content': f"https://image.url/{model.replace('/', '_')}/{abs(hash(str(messages)))}.png"}}], 'model': model}
        elif "sora" in model or "runway" in model or "kling" in model or "luma" in model or "pika" in model:
            # Simulate slight variation in video quality for critique loop
            video_hash = abs(hash(str(messages)))
            quality_suffix = ""
            if kwargs.get("attempt", 1) < 3: # Lower quality for first few attempts
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




# --- Tool Definitions (Remains the same as in the previous complete code block) ---
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
        "models": ["kling", "sora", "runwayml/gen-2", "pika-labs"],
        "cost_per_use": {"kling": 5.00, "sora": 10.00, "runwayml/gen-2": 0.50, "pika-labs": 0.30},
        "latency_ms": {"kling": 30000, "sora": 60000, "runwayml/gen-2": 10000, "pika-labs": 8000},
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
    "llm_replan": { # New internal tool for re-planning based on critique
        "name": "llm_replan",
        "description": "Analyzes critique feedback and suggests modifications to prompt or strategy for next generation attempt.",
        "input_schema": {"current_prompt": "str", "feedback": "str", "previous_model": "str", "attempt_number": "int"},
        "models": ["claude-3-5-sonnet", "gpt-4o"],
        "cost_per_use": 0.015,
        "latency_ms": 2000,
        "is_autonomous": True
    }
}


# --- Default Configuration from dollar.yaml (Conceptual) ---
DEFAULT_CONFIG = {
    "budget": {
        "hard_cap": 100.0,
        "warn_at": 75.0
    },
    "preferences": {
        "default_video_model": "kling",
        "fallback_chain": ["kling", "sora", "runwayml/gen-2", "pika-labs", "local_animatediff"],
        "upscaling": "always",
        "audio": True,
        "self_critique": "strict", # "strict", "moderate", "none"
        "style_consistency_enforcement": "high",
        "default_style": "cinematic",
        "max_generation_attempts": 3 # New config item
    },
    "agent_settings": {
        "memory_size": 20,
        "cost_weight": 1000,
        "latency_weight": 1
    }
}




# --- AgenticDollar Class ---
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
        """Logs actions for transparency and future self-improvement."""
        log_entry = {"timestamp": time.time(), "action_type": action_type, "details": details}
        self.working_memory.append(log_entry)
        print(f"  [LOG] {action_type}: {details.get('tool', '')} - {details.get('message', '')[:50]}...")


    async def _ask_for_user_confirmation(self, message: str) -> bool:
        """Asks the user for confirmation for high-cost or critical actions."""
        if self.user_confirmation_callback:
            return await asyncio.to_thread(self.user_confirmation_callback, message)
        print(f"  [USER CONFIRMATION REQUIRED]: {message} (Simulating 'yes' for now)")
        return True


    async def _select_best_tool_and_model(self, task_description: str, tool_type: str,
                                          max_cost_for_step: float = float('inf')) -> Optional[Dict]:
        """
        Dynamically selects the most appropriate tool and model based on task,
        cost, and available options, respecting preference for `fallback_chain`.
        Latency is now handled more implicitly by the fallback order.
        """
        
        eligible_tools_defs = {t_name: t_def for t_name, t_def in self.tools.items()
                               if t_def.get("name") == tool_type}


        if not eligible_tools_defs:
            return None


        best_option = None
        
        # Determine the order of models to try based on preferences or default
        # For video generation, use the explicit fallback chain
        if tool_type == "generate_video":
            model_selection_order = self.config["preferences"]["fallback_chain"]
        else:
            # For other tools, find all models and sort by a simple heuristic (e.g., lowest cost first)
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


            # Check if this model exceeds budget for this step or overall hard cap
            if cost <= max_cost_for_step and (self.current_spend + cost <= self.budget_hard_cap):
                best_option = {"tool_name": tool_found_name, "model_name": model_name, "cost": cost, "latency": latency}
                # Since the fallback chain implies preference/order, take the first one that fits
                break # Found the best available model in the preferred order within budget
        
        # We only update current_spend upon successful _execution_ of the tool
        return best_option


    async def _execute_tool(self, tool_option: Dict, **kwargs) -> Any:
        """Executes the chosen tool and handles potential retries/fallbacks."""
        tool_name = tool_option["tool_name"]
        model_name = tool_option["model_name"]
        cost = tool_option["cost"]


        # Special handling for local fallback tool
        if tool_name == "local_animatediff":
            print(f"  [LOCAL EXECUTION] Executing {tool_name} locally with params: {kwargs}")
            await asyncio.sleep(tool_option["latency"])
            return {"result_url": f"https://local.server/{tool_name}_{abs(hash(str(kwargs)))}.mp4", "status": "completed"}


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
        """Uses an LLM to rewrite and enhance a prompt for optimal generation."""
        if self.config["preferences"]["self_critique"] == "none":
            print("  [PROMPT OPTIMIZATION] Skipping prompt optimization based on preferences.")
            return original_prompt


        print(f"  [PROMPT OPTIMIZATION] Optimizing prompt for {target_model_type} (Attempt {attempt_number})...")
        remaining_budget = self.budget_hard_cap - current_spend_context
        # Cap optimization cost to a small fraction of remaining budget or a fixed small amount
        max_opt_cost = min(0.05, remaining_budget * 0.1) # Max 5 cents or 10% of remaining budget
        
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
            attempt_number=attempt_number # Pass attempt number for dummy to vary output
        )
        return optimized_response["result"] if optimized_response["status"] == "completed" else original_prompt


    async def _analyze_critique_and_replan(self, current_prompt: str, feedback: Dict[str, Any], previous_model: str, attempt_number: int) -> Dict[str, Any]:
        """
        Uses an LLM to analyze critique feedback and suggest modifications
        to the prompt or strategy for the next generation attempt.
        Returns a dict with 'new_prompt' and optionally 'suggested_model'.
        """
        print(f"  [REPLANNING] Analyzing critique for attempt {attempt_number}...")
        replan_tool = await self._select_best_tool_and_model("Analyze critique and replan", "llm_replan")
        if not replan_tool:
            print("  [WARNING] No replanning tool available. Cannot intelligently re-plan.")
            return {"new_prompt": current_prompt, "suggested_model": None}


        # Structure the feedback for the LLM
        feedback_str = json.dumps(feedback)
        
        llm_response = await self._execute_tool(
            replan_tool,
            current_prompt=current_prompt,
            feedback=feedback_str,
            previous_model=previous_model,
            attempt_number=attempt_number
        )


        if llm_response["status"] == "completed":
            # Simulate parsing LLM's suggested new prompt and model
            response_content = llm_response["result"]
            new_prompt = current_prompt # Default to no change
            suggested_model = None


            if "Suggest modifying prompt to explicitly request" in response_content:
                # Simple parsing of dummy LLM response
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
        """Performs self-critique: frame consistency and script adherence."""
        self_critique_level = self.config["preferences"]["self_critique"]
        if self_critique_level == "none":
            print("  [QUALITY ASSESSMENT] Skipping self-critique based on preferences.")
            return {"frame_consistency": {"score": 1.0, "feedback": "skipped"}, "script_adherence": "skipped", "overall_pass": True, "feedback_messages": []}


        print(f"  [QUALITY ASSESSMENT] Assessing video quality for {video_url} (Level: {self_critique_level})...")
        overall_pass = True
        feedback_messages = []
        frame_consistency_result = {"score": 1.0, "feedback": "skipped"}
        script_adherence_result = "skipped"


        # 1. Frame Consistency Scoring
        if self_critique_level in ["strict", "moderate"]:
            consistency_tool = await self._select_best_tool_and_model("Check video frame consistency", "compare_frames", 
                                                    max_cost_for_step=self.budget_hard_cap - self.current_spend)
            if consistency_tool:
                consistency_response = await self._execute_tool(consistency_tool, video_url=video_url)
                if consistency_response["status"] == "completed":
                    content = consistency_response["result"] # This is the dict from DummyLitellm
                    if isinstance(content, dict) and "score" in content:
                        frame_consistency_result = content
                        if content["score"] < 0.7: # Example threshold for failure
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


        # 2. Script Adherence (using an LLM to compare video description to script)
        if self_critique_level == "strict" and script:
            llm_critique_tool = await self._select_best_tool_and_model("Critique video against script", "optimize_prompt",
                                                    max_cost_for_step=self.budget_hard_cap - self.current_spend)
            if llm_critique_tool:
                critique_response = await self._execute_tool(
                    llm_critique_tool,
                    original_prompt=f"Critique video at {video_url} against script: '{script}'",
                    target_model_type="video_critique_llm",
                    video_url=video_url # Pass for DummyLitellm's quality simulation
                )
                if critique_response["status"] == "completed":
                    script_adherence_result = critique_response["result"]
                    if "score 8/10" not in script_adherence_result: # Example parsing of failure for strict mode
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
        """
        Orchestrates the generation of a high-quality video with advanced features,
        respecting config and command-line overrides, including an iterative self-critique loop.
        """
        print(f"\n--- Initiating Advanced Video Generation ---")
        print(f"User Request: '{user_request_prompt}'")
        if script:
            print(f"Script provided: '{script[:100]}...'")


        # --- Apply Config Defaults and Command-Line Overrides ---
        actual_max_cost_per_job = max_cost_per_job if max_cost_per_job is not None else self.budget_hard_cap
        actual_resolution = target_resolution if target_resolution is not None else "1080p"
        actual_audio_preference = request_audio if request_audio is not None else self.config["preferences"]["audio"]
        actual_upscaling_preference = request_upscaling if request_upscaling is not None else self.config["preferences"]["upscaling"]
        actual_self_critique_rigor = self.config["preferences"]["self_critique"]
        max_attempts = self.config["preferences"]["max_generation_attempts"]
        
        current_video_prompt = user_request_prompt # This prompt will be refined in the loop
        generated_video_url = None
        video_generator_used = "N/A"
        final_quality_assessment = {}
        all_attempt_details = []


        # --- Budget Checks ---
        if self.current_spend + actual_max_cost_per_job > self.budget_hard_cap:
            print(f"  [BUDGET ALERT] Job would exceed hard cap of ${self.budget_hard_cap:.2f}. Current spend: ${self.current_spend:.2f}. Requested job cost: ${actual_max_cost_per_job:.2f}.")
            return {"status": "aborted", "message": "Job exceeds overall budget cap."}
        elif self.current_spend + actual_max_cost_per_job > self.budget_warn_at:
             print(f"  [BUDGET WARNING] Job will cause total spend to exceed warning threshold of ${self.budget_warn_at:.2f}. Current spend: ${self.current_spend:.2f}.")


        # --- User Confirmation (for high-cost/critical tasks) ---
        if not await self._ask_for_user_confirmation(
            f"This video generation could cost up to ${actual_max_cost_per_job:.2f} and take several minutes. Proceed?"
        ):
            return {"status": "aborted", "message": "User cancelled operation."}


        # --- Iterative Self-Critique & Refinement Loop ---
        for attempt in range(1, max_attempts + 1):
            print(f"\n--- GENERATION ATTEMPT {attempt}/{max_attempts} ---")


            # 1. Auto-Prompt Optimization (or Re-optimization based on critique)
            optimized_prompt_for_attempt = await self._optimize_prompt(current_video_prompt, "video_generator", self.current_spend, attempt)
            print(f"Prompt for this attempt: '{optimized_prompt_for_attempt}'")
            
            # 2. Dynamic Tool/Model Selection for Video Generation
            # If the replanning LLM suggested a model, prioritize it for this attempt
            suggested_model_for_attempt = None
            if attempt > 1 and all_attempt_details[-1].get("replan_suggestions"):
                suggested_model_for_attempt = all_attempt_details[-1]["replan_suggestions"].get("suggested_model")
                if suggested_model_for_attempt:
                    print(f"  [MODEL OVERRIDE] Prioritizing LLM-suggested model: {suggested_model_for_attempt}")


            # Calculate remaining budget for this generation attempt
            remaining_budget_for_generation = actual_max_cost_per_job - (self.current_spend - (self.current_spend if self.current_spend < actual_max_cost_per_job else 0)) # simplified


            video_tool_option = await self._select_best_tool_and_model(
                f"Generate video for: {optimized_prompt_for_attempt}", 
                "generate_video", 
                max_cost_for_step=remaining_budget_for_generation
            )
            # If a model was explicitly suggested by the replanning LLM, try to use it if available
            if suggested_model_for_attempt and video_tool_option and video_tool_option["model_name"] != suggested_model_for_attempt:
                 # Re-evaluate, forcing the suggested model if possible
                 forced_tool_option = None
                 for t_name, t_def in self.tools.items():
                     if t_def.get("name") == "generate_video" and suggested_model_for_attempt in t_def.get("models", []):
                         cost = t_def["cost_per_use"].get(suggested_model_for_attempt, t_def["cost_per_use"]) if isinstance(t_def["cost_per_use"], dict) else t_def["cost_per_use"]
                         latency = t_def["latency_ms"].get(suggested_model_for_attempt, t_def["latency_ms"]) if isinstance(t_def["latency_ms"], dict) else t_def["latency_ms"]
                         if cost <= remaining_budget_for_generation and (self.current_spend + cost <= self.budget_hard_cap):
                             forced_tool_option = {"tool_name": t_name, "model_name": suggested_model_for_attempt, "cost": cost, "latency": latency}
                             break
                 if forced_tool_option:
                     video_tool_option = forced_tool_option
                     print(f"  [MODEL OVERRIDE] Successfully switched to LLM-suggested model: {suggested_model_for_attempt}")




            if not video_tool_option:
                all_attempt_details.append({"attempt": attempt, "status": "failed_no_model", "message": "Could not find a suitable video generation model within budget/constraints."})
                print(f"  [FAILURE] Attempt {attempt} failed: No suitable video generation model.")
                break # Cannot proceed without a model


            video_generator_used = video_tool_option["model_name"]
            print(f"Selected video generator for attempt {attempt}: {video_generator_used}")


            # 3. Video Generation Attempt
            video_result = await self._execute_tool(
                video_tool_option,
                prompt=optimized_prompt_for_attempt,
                resolution=actual_resolution,
                duration=max_duration,
                attempt=attempt # Pass attempt number for dummy to vary output
            )


            if video_result["status"] != "completed" or not video_result["result"]:
                all_attempt_details.append({"attempt": attempt, "status": "failed_generation", "message": video_result.get("error", "Generation failed.")})
                print(f"  [FAILURE] Attempt {attempt} failed: {video_result.get('error', 'Generation failed.')}")
                # Decide if we can retry with a different model or if it's a hard failure
                if attempt == max_attempts:
                    return {"status": "failed", "message": "Video generation failed after all attempts."}
                else:
                    # For a real system, here we'd analyze the error and decide to replan or try next model in fallback
                    print("  [RETRY] Attempting next iteration without explicit replanning for now.")
                    continue # Try next iteration


            generated_video_url = video_result["result"]
            print(f"Generated Video URL (Attempt {attempt}): {generated_video_url}")


            # 4. Self-Critique Loop
            final_quality_assessment = await self._assess_video_quality(generated_video_url, script)
            all_attempt_details.append({
                "attempt": attempt,
                "status": "completed",
                "video_url": generated_video_url,
                "assessment": final_quality_assessment,
                "optimized_prompt": optimized_prompt_for_attempt,
                "video_generator": video_generator_used
            })


            if final_quality_assessment["overall_pass"]:
                print(f"  [SUCCESS] Video passed quality assessment on attempt {attempt}!")
                break # Exit loop, quality met
            else:
                print(f"  [FAILURE] Video did NOT pass quality assessment on attempt {attempt}: {final_quality_assessment['feedback_messages']}")
                if attempt < max_attempts:
                    # 5. Analyze Critique and Re-plan for next attempt
                    replan_result = await self._analyze_critique_and_replan(
                        current_prompt=current_video_prompt, # Use the user's initial prompt as base for replanning
                        feedback=final_quality_assessment,
                        previous_model=video_generator_used,
                        attempt_number=attempt
                    )
                    current_video_prompt = replan_result["new_prompt"] # Update prompt for next attempt
                    all_attempt_details[-1]["replan_suggestions"] = replan_result # Store replan suggestions
                    print(f"  [RETRYING] Retrying with refined prompt: '{current_video_prompt}'")
                    # The suggested model from replan will be prioritized in the next loop iteration
                else:
                    print(f"  [FAILURE] Max attempts ({max_attempts}) reached. Video did not meet quality standards.")
                    return {"status": "failed", "message": "Video did not meet quality standards after max attempts."}


        if not generated_video_url:
            return {"status": "failed", "message": "No video could be successfully generated."}




        # --- Post-Processing Steps (executed only after a quality-approved video is generated) ---
        
        # 6. Video Upscaling
        final_video_url = generated_video_url
        if actual_upscaling_preference == "always" or \
           (actual_upscaling_preference == "if_needed" and actual_resolution not in final_video_url): # Simplified heuristic
            
            print("  [UPSCALING] Attempting to upscale video...")
            upscale_tool_option = await self._select_best_tool_and_model("Upscale video", "upscale_video", 
                                                            max_cost_for_step=self.budget_hard_cap - self.current_spend)
            if upscale_tool_option:
                upscaled_video_result = await self._execute_tool(
                    upscale_tool_option,
                    video_url=generated_video_url,
                    target_resolution=actual_resolution
                )
                if upscaled_video_result["status"] == "completed":
                    final_video_url = upscaled_video_result["result"]
                    print(f"Upscaled Video URL: {final_video_url}")
                else:
                    print("  [WARNING] Video upscaling failed. Using raw generated video.")
            else:
                print("  [WARNING] Video upscaling tool not available or out of budget. Using raw generated video.")
        else:
            print(f"  [UPSCALING] Upscaling preference set to '{actual_upscaling_preference}', skipping.")


        # 7. Audio Generation
        generated_audio_url = None
        if script and actual_audio_preference:
            print("  [AUDIO GENERATION] Generating audio for script...")
            audio_tool_option = await self._select_best_tool_and_model("Generate audio from script", "generate_audio",
                                                            max_cost_for_step=self.budget_hard_cap - self.current_spend)
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
                print("  [WARNING] Audio generation tool not available or out of budget.")
        else:
            print("  [AUDIO GENERATION] Audio generation skipped based on preferences or no script.")


        return {
            "status": "completed",
            "final_video_url": final_video_url,
            "generated_audio_url": generated_audio_url,
            "details": {
                "initial_prompt": user_request_prompt,
                "final_optimized_prompt": current_video_prompt,
                "video_generator_used": video_generator_used,
                "quality_assessment_final": final_quality_assessment,
                "total_cost_incurred_for_job": self.current_spend,
                "generation_attempts": all_attempt_details
            }
        }


# --- Example Usage (How you would interact with Dollar via CLI or API) ---
async def main():
    dollar_instance = AgenticDollar()


    async def my_user_confirm_func(message: str) -> bool:
        response = await asyncio.to_thread(input, f"Confirm (y/n): {message} ")
        return response.lower() == 'y'
    dollar_instance.user_confirmation_callback = my_user_confirm_func




    # --- Scenario 1: Simulate CLI command with a prompt that might need refinement
    print("\n" + "="*80)
    print("SCENARIO 1: Cyberpunk Samurai Video (requiring refinement)")
    print("="*80)


    cli_result = await dollar_instance.generate_advanced_video(
        user_request_prompt="A cyberpunk samurai walking through neon rain, slow motion. Make it look cool.",
        target_resolution="4K",
        max_cost_per_job=20.0, # Increased budget to allow for retries
        request_audio=True
    )
    print(f"\n--- SCENARIO 1 Result ---")
    print(json.dumps(cli_result, indent=2))
    print(f"Current Total Spend: ${dollar_instance.current_spend:.2f}")


    # --- Scenario 2: Simple video with script that might pass on first try or one retry
    print("\n" + "="*80)
    print("SCENARIO 2: Serene Forest Scene with Script")
    print("="*80)


    dollar_instance_2 = AgenticDollar() # New instance for a clean budget
    dollar_instance_2.user_confirmation_callback = my_user_confirm_func


    api_result = await dollar_instance_2.generate_advanced_video(
        user_request_prompt="A serene forest scene with a hidden waterfall, gentle sunlight filtering through dense canopy, ancient trees.",
        script="The ancient trees whispered secrets as crystal waters tumbled into a hidden pool, untouched by time, a haven of tranquility.",
        target_resolution="1080p",
        max_cost_per_job=15.0,
        request_audio=True
    )
    print(f"\n--- SCENARIO 2 Result ---")
    print(json.dumps(api_result, indent=2))
    print(f"Current Total Spend (for this second job): ${dollar_instance_2.current_spend:.2f}")




if __name__ == "__main__":
    asyncio.run(main())