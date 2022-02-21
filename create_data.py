import argparse
import csv
import subprocess
from pathlib import Path

import yaml


def run_process(arguments, *args, **kwargs):
    arguments = [str(arg) for arg in arguments]
    print(" ".join(arguments))
    return subprocess.run(arguments, *args, **kwargs)


def spm_train(input_path, vocab_size, model_prefix):
    run_process(
        [
            "spm_train",
            f"--input={input_path}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            "--character_coverage=0.9995",
            "--input_sentence_size=1000000",
            "--shuffle_input_sentence=true",
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("actions", choices=("split", "spm"))

    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--valid_size", default=2000, type=int)
    parser.add_argument("--corpus", default="data/ilia-parallel-corpus_cleaned.csv")
    parser.add_argument("--no_header", action='store_false', default=True)

    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.load(config_path.open('r', encoding='utf-8'), yaml.FullLoader)

    from_code = config.get('from_code', 'ka')
    to_code = config.get('to_code', 'en')
    from_name = config.get('from_name', 'Georgian')
    to_name = config.get('to_name', 'English')
    package_version = config.get('package_version', '1.0')
    argos_version = config.get('argos_version', '1.5')

    open_nmt_path = Path(args.open_nmt_path)

    data_path = Path(config['data_dir'])
    data_path.mkdir(exist_ok=True)

    split_data_path = data_path / "split_data/"
    split_data_path.mkdir(exist_ok=True)

    if 'split' in args.actions:
        source_data, target_data, size = [], [], 0
        with Path(args.corpus).open('r', encoding='utf-8', newline='') as fp:
            iterator = iter(csv.reader(fp, delimiter='\t'))

            # Skip header
            if not args.no_header:
                header = next(iterator)
                from_idx = 0 if header[0] == from_code else 1
                to_idx = 1 if from_idx == 0 else 0
            else:
                from_idx = 0
                to_idx = 1

            for idx, row in enumerate(iterator, start=1):
                if len(row) < 2:
                    print(row)
                    raise ValueError("")
                source, target = row[from_idx], row[to_idx]
                source_data.append(source)
                target_data.append(target)
                size = idx

        with (split_data_path / f"{from_code}-val.txt").open("w") as source_valid_file:
            source_valid_file.writelines(source_data[:args.valid_size])
        with (split_data_path / f"{from_code}-train.txt").open("w") as source_train_file:
            source_train_file.writelines(source_data[args.valid_size:])
        with (split_data_path / f"{to_code}-val.txt").open("w") as target_valid_file:
            target_valid_file.writelines(target_data[:args.valid_size])
        with (split_data_path / f"{to_code}-train.txt").open("w") as target_train_file:
            target_train_file.writelines(target_data[args.valid_size:])

    if 'spm' in args.actions:
        if config['share_vocab']:
            assert config['src_subword_model'] == config['tgt_subword_model'], "Misconfigured subword model paths"
            assert config['src_vocab_size'] == config['tgt_vocab_size'], "Misconfigured subword model vocabulary sizes"

            combined_data = (data_path / "split_data/all.txt")
            with combined_data.open("w") as combined:
                with (split_data_path / f"{from_code}-train.txt").open("r") as source_train_file:
                    combined.write(source_train_file.read())
                with (split_data_path / f"{to_code}-train.txt").open("r") as target_train_file:
                    combined.write(target_train_file.read())

            spm_train(str(combined_data.absolute()), config['src_vocab_size'], config['src_subword_model'])

            combined_data.unlink()
        else:
            spm_train(
                str((split_data_path / f"{from_code}-val.txt")),
                config['src_vocab_size'],
                config['src_subword_model']
            )
            spm_train(
                str((split_data_path / f"{to_code}-val.txt")),
                config['tgt_vocab_size'],
                config['tgt_subword_model']
            )

    # TODO: Don't hardcode vocab_size and set user_defined_symbols

    run_process(["onmt_build_vocab", "-config", config_path, "-n_sample", "-1"])
