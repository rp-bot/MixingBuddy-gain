import json
import os
import random
import copy
import numpy as np
import soundfile as sf
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def apply_gain(input_path: str, output_path: str, gain_db: float) -> None:
    """
    Applies a gain to an audio file and saves it.
    """
    data, samplerate = sf.read(input_path)
    gain_linear = 10 ** (gain_db / 20)
    data = data * gain_linear
    # Clipping audio to [-1, 1] range
    data = np.clip(data, -1, 1)
    sf.write(output_path, data, samplerate)

def augment_no_error_samples(input_file: str, output_file: str) -> Tuple[int, int, int]:
    """Augments the 'no_error' samples by applying a random gain.

    Returns:
        Tuple[int, int, int]: Processed samples, total samples, augmented examples.
    """
    logger.info("Counting total samples in %s", input_file)
    with open(input_file, 'r') as count_file:
        total_samples = sum(1 for _ in count_file)
    logger.info("Found %d total samples", total_samples)

    processed = 0
    augmented_created = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for processed, line in enumerate(infile, start=1):
            sample = json.loads(line)
            outfile.write(line) # Write original sample

            if sample['meta']['error_category'] == 'no_error':
                for i in range(3): # Create 3 augmented versions
                    # Generate a random gain between -3 and +3 dB
                    gain_db = random.uniform(-3, 3)

                    original_mix_path = sample['flawed_mix_path']

                    # Create a new unique filename for the augmented audio
                    mix_dirname, mix_basename = os.path.split(original_mix_path)
                    base_name, ext = os.path.splitext(mix_basename)
                    new_filename = f"{base_name}_aug_{i+1}{ext}"
                    new_mix_path = os.path.join(mix_dirname, new_filename)
                    os.makedirs(os.path.dirname(new_mix_path), exist_ok=True)

                    # Apply gain and save the new audio file
                    apply_gain(original_mix_path, new_mix_path, gain_db)

                    # Create a new sample dictionary for the augmented data
                    new_sample = copy.deepcopy(sample)
                    new_sample['global_uid'] = f"{sample['global_uid']}_aug_{i+1}"
                    new_sample['flawed_mix_path'] = new_mix_path
                    new_sample.setdefault('meta', {})['augmentation'] = {
                        'type': 'gain_shift',
                        'gain_db': gain_db
                    }

                    outfile.write(json.dumps(new_sample) + '\n')
                    augmented_created += 1

            if processed % 1000 == 0:
                logger.info(
                    "Processed %d/%d samples; generated %d augmented entries so far",
                    processed,
                    total_samples,
                    augmented_created,
                )

    return processed, total_samples, augmented_created

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    input_jsonl = 'data/musdb18hq_processed/train/training_samples.jsonl'
    output_jsonl = 'data/musdb18hq_processed/train/training_samples_augmented_full.jsonl'
    logger.info("Starting augmentation: input=%s output=%s", input_jsonl, output_jsonl)
    total_processed, total_samples, total_augmented = augment_no_error_samples(input_jsonl, output_jsonl)
    logger.info(
        "Augmentation complete. Processed %d/%d samples and generated %d augmented entries.",
        total_processed,
        total_samples,
        total_augmented,
    )
    logger.info("Augmented data saved to %s", output_jsonl)
