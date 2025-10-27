import os
import json
import argparse
import random

def create_dataset(input_dir, train_file, test_file, split_ratio):
    all_data = []
    for instrument in os.listdir(input_dir):
        instrument_path = os.path.join(input_dir, instrument)
        if os.path.isdir(instrument_path):
            for audio_file in os.listdir(instrument_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(instrument_path, audio_file)
                    data = {
                        "path": audio_path,
                        "instruction": "listen to the 3 second audio and classify which instrument it is.",
                        "response": instrument
                    }
                    all_data.append(data)

    random.shuffle(all_data)
    split_index = int(len(all_data) * split_ratio)
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]

    if train_file:
        with open(train_file, 'w') as f:
            for entry in train_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Generated training dataset with {len(train_data)} samples: {train_file}")

    if test_file:
        with open(test_file, 'w') as f:
            for entry in test_data:
                f.write(json.dumps(entry) + '\n')
        print(f"Generated testing dataset with {len(test_data)} samples: {test_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a JSONL dataset for instrument classification with train/test split.')
    parser.add_argument('input_dir', type=str, help='The directory containing the instrument audio files.')
    parser.add_argument('--train_file', type=str, default=None, help='The path to the output JSONL file for the training set.')
    parser.add_argument('--test_file', type=str, default=None, help='The path to the output JSONL file for the testing set.')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='The ratio of data to use for the training set (e.g., 0.8 for 80%% train, 20%% test).')
    args = parser.parse_args()

    if not args.train_file and not args.test_file:
        parser.error("At least one of --train_file or --test_file must be provided.")

    create_dataset(args.input_dir, args.train_file, args.test_file, args.split_ratio)
