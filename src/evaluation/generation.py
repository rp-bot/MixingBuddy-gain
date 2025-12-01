import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


def extract_first_sentence(text: str) -> Optional[str]:
    """Extract the first sentence from text.
    
    Args:
        text: Input text
        
    Returns:
        First sentence (with period) or None if no sentence found
    """
    if not text:
        return None
    
    # Split by common sentence endings
    # Look for period, exclamation, or question mark followed by space or end of string
    sentence_endings = re.compile(r'[.!?](?:\s+|$)')
    match = sentence_endings.search(text)
    
    if match:
        # Return first sentence including the punctuation
        end_pos = match.end()
        first_sentence = text[:end_pos].strip()
        return first_sentence if first_sentence else None
    
    # If no sentence ending found, return the whole text
    return text.strip() if text.strip() else None


def generate_and_compare(
    model: Any,
    dataset: Any,
    num_samples: int,
    max_new_tokens: int,
    use_instruction: bool,
    system_message: str,
    output_dir: Path,
    use_partial_ground_truth: bool = False,
    num_prefix_tokens: int = 5,
    checkpoint_number: Optional[int] = None,
    add_first_sentence_to_instruction: bool = False,
    use_tqdm: bool = True,
    log_every_n_samples: Optional[int] = None,
):
    """Generate responses for samples and save to JSONL for later analysis.
    
    Args:
        model: The model to use for generation
        dataset: The dataset to generate predictions for
        num_samples: Number of samples to generate
        max_new_tokens: Maximum number of new tokens to generate
        use_instruction: Whether to use instructions in generation
        system_message: System message for the model
        output_dir: Directory to save predictions
        use_partial_ground_truth: If True, feed first num_prefix_tokens of ground truth as prefix
        num_prefix_tokens: Number of tokens from ground truth to use as prefix
        checkpoint_number: Checkpoint number to include in filename (e.g., 500)
        add_first_sentence_to_instruction: If True, append first sentence of ground truth to instruction
        use_tqdm: If True, show tqdm progress bar during generation (default: True)
        log_every_n_samples: Log progress every N samples (None to disable periodic logging, default: None)
    """
    logger.info("Generating samples for qualitative evaluation")
    if use_partial_ground_truth:
        logger.info("Using partial ground truth: first %d tokens", num_prefix_tokens)
    if add_first_sentence_to_instruction:
        logger.info("Adding first sentence of ground truth to instruction")

    predictions = []

    def get_expected_magnitude_range(error_category):
        if error_category == "no_error":
            return None, None
        if error_category in ["quiet", "loud"]:
            return 3.0, 6.0
        if error_category in ["very_quiet", "very_loud"]:
            return 6.0, 12.0
        return None, None

    total_samples = num_samples if num_samples is not None else len(dataset)

    # Use tqdm if enabled, otherwise use regular range
    iterator = tqdm(range(total_samples), desc="Generating predictions") if use_tqdm else range(total_samples)
    for i in iterator:
        sample = dataset[i]
        instruction = sample["instruction"]
        audio = sample["audio"]
        ground_truth = sample["response"]
        global_uid = sample["global_uid"]
        target_stem = sample["target_stem"]
        error_category = sample["error_category"]

        item_data = dataset.data[i]
        intended_gain_db = item_data["meta"].get("intended_gain_db", 0.0)
        start_sec = item_data["meta"].get("time_ref", {}).get("start_sec", 0.0)
        end_sec = item_data["meta"].get("time_ref", {}).get("end_sec", 0.0)

        expected_min_db, expected_max_db = get_expected_magnitude_range(error_category)

        # Extract prefix tokens from ground truth if requested (do this first)
        prefix_tokens = None
        prefix_text = None
        if use_partial_ground_truth and ground_truth:
            # Tokenize ground truth and take first num_prefix_tokens
            tokenizer = model.tokenizer
            ground_truth_tokens = tokenizer(
                ground_truth, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            if ground_truth_tokens.shape[1] >= num_prefix_tokens:
                prefix_tokens = ground_truth_tokens[:, :num_prefix_tokens]
            elif ground_truth_tokens.shape[1] > 0:
                # Use all available tokens if less than num_prefix_tokens
                prefix_tokens = ground_truth_tokens
            
            # Decode prefix tokens to text for storage in predictions
            if prefix_tokens is not None:
                prefix_text = tokenizer.decode(prefix_tokens[0], skip_special_tokens=False)

        # Modify instruction if requested
        original_instruction = instruction
        if add_first_sentence_to_instruction and ground_truth:
            first_sentence = extract_first_sentence(ground_truth)
            if first_sentence:
                # Append first sentence to instruction
                instruction = f"{instruction} {first_sentence}"

        text_for_generation = instruction if use_instruction else ""

        # Pass prefix_tokens to model.generate() so it appears after the assistant tag
        # (not in the user message)
        generated_text = model.generate(
            text_input=text_for_generation,
            audio=audio,
            max_new_tokens=max_new_tokens,
            system_message=system_message,
            prefix_tokens=prefix_tokens,  # This will be added after <|im_start|>assistant
        )

        # Create combined generated text with prefix for visual completeness
        if prefix_text and use_partial_ground_truth:
            # Combine prefix and generated text for a complete-looking response
            combined_generated = f"{prefix_text}{generated_text}"
        else:
            combined_generated = generated_text
        
        prediction = {
            "global_uid": global_uid,
            "instruction": original_instruction,  # Store original instruction
            "modified_instruction": instruction if add_first_sentence_to_instruction else original_instruction,  # Store modified instruction if first sentence was added
            "ground_truth": ground_truth,
            "generated": combined_generated,  # Combined prefix + generated for visual completeness
            "generated_only": generated_text,  # Just the model's continuation (without prefix)
            "target_stem": target_stem,
            "error_category": error_category,
            "intended_gain_db": intended_gain_db,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "expected_magnitude_min_db": expected_min_db,
            "expected_magnitude_max_db": expected_max_db,
            "used_partial_ground_truth": use_partial_ground_truth,
            "prefix_text": prefix_text,  # The prefix text (appears after assistant tag, kept separately for reference)
            "added_first_sentence_to_instruction": add_first_sentence_to_instruction,
        }
        predictions.append(prediction)

        # Log first 3 samples with full details, then periodic logging if configured
        should_log = i < 3 or (log_every_n_samples is not None and (i + 1) % log_every_n_samples == 0)
        
        if should_log:
            if i < 3:
                # Full detailed logging for first 3 samples
                logger.debug(
                    "Sample %d/%d: uid=%s stem=%s err=%s",
                    i + 1,
                    total_samples,
                    global_uid,
                    target_stem,
                    error_category,
                )
                
                # Log full input/output for verification
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text_for_generation},
                ]
                # Note: This recreates the prompt string. The actual input embeddings 
                # will include the prefix tokens appended after this.
                prompt_str = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                log_msg = f"\n{'='*40}\nSample {i+1} Visualization:"
                log_msg += f"\n--- INPUT PROMPT (Chat Template) ---\n{prompt_str}"
                
                if prefix_text is not None:
                    log_msg += f"\n--- PREFIX (Appended to Input) ---\n{prefix_text}"
                    
                log_msg += f"\n--- GENERATED OUTPUT ---\n{generated_text}\n{'='*40}"
                logger.info(log_msg)
            else:
                # Periodic progress logging
                logger.info(
                    "Progress: %d/%d samples (%.1f%%) - uid=%s stem=%s err=%s",
                    i + 1,
                    total_samples,
                    100.0 * (i + 1) / total_samples,
                    global_uid,
                    target_stem,
                    error_category,
                )

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename with checkpoint number and increment
    if checkpoint_number is not None:
        base_name = f"predictions-checkpoint-{checkpoint_number}"
        # Find existing files with this pattern
        pattern = re.compile(rf"^{re.escape(base_name)}(?:-(\d+))?\.jsonl$")
        existing_files = [f for f in predictions_dir.glob(f"{base_name}*.jsonl")]
        
        # Check if base file exists (no increment)
        base_file = predictions_dir / f"{base_name}.jsonl"
        has_base_file = base_file.exists()
        
        # Find the highest increment number
        max_increment = -1
        for file in existing_files:
            match = pattern.match(file.name)
            if match:
                increment_str = match.group(1)
                if increment_str:
                    max_increment = max(max_increment, int(increment_str))
        
        # Set the next increment number
        # If base file exists or any incremented files exist, use increment
        if has_base_file or max_increment >= 0:
            next_increment = max_increment + 1
            predictions_file = predictions_dir / f"{base_name}-{next_increment}.jsonl"
        else:
            # No existing files, use base name
            predictions_file = predictions_dir / f"{base_name}.jsonl"
    else:
        predictions_file = predictions_dir / "predictions.jsonl"

    logger.info("Saving predictions to %s", predictions_file)
    with open(predictions_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info("Saved %d predictions", len(predictions))
    return predictions_file
