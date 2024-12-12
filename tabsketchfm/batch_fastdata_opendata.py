from tabsketchfm.data_processing.data_prep import prep_data
import os
import argparse

def preprocess_pretrain_data(input_dir, metadata_dir, output_path):
    for folder, subs, files in os.walk(input_dir):
        for index, filename in enumerate(files):
            lower_filename = filename.lower()
            if not lower_filename.endswith(".csv"):
                continue

            print(f" folder name {folder} parent_dir is {os.path.basename(folder)}")
            metadata_file = f"{metadata_dir}/{os.path.basename(folder)}/{filename}.meta"

            if not os.path.exists(metadata_file):
                print(f"file {metadata_file} does not exist")
                continue

            print(filename, metadata_file)
            prep_data(f"{folder}/{filename}", output_path, metadata_file, None)

            if index > 20:
                break



parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True, help='path to the directory with tables')
parser.add_argument('--metadata_dir', required=True, help='path to the directory with metadata')
parser.add_argument('--output_dir', required=True, help='path to processed data dir')
args = parser.parse_args()
preprocess_pretrain_data(args.input_dir, args.metadata_dir, args.output_dir)
