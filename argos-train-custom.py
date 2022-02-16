import argparse
import shutil
import subprocess

from argostrain import opennmtutils
from argostrain.custom_dataset import CustomDataset
from argostrain.data import prepare_data
from argostrain.dataset import *


def run_process(arguments, *args, **kwargs):
    print(" ".join(arguments))
    return subprocess.run(arguments, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("action", nargs="+", type=str, choices=('corpus', 'train'))

    parser.add_argument("--from_code", default='ka')
    parser.add_argument("--to_code", default='en')
    parser.add_argument("--from_name", default='Georgian')
    parser.add_argument("--to_name", default='English')
    parser.add_argument("--package_version", "--version", default='1.0')
    parser.add_argument("--argos_version", default="1.5")

    parser.add_argument('--stanza_lang_code', type=str, required=False)
    parser.add_argument('--run_dir', type=str, default='run/', required=False)
    parser.add_argument('--open_nmt_path', type=str, default='./../OpenNMT-py/', required=False)

    args = parser.parse_args()

    open_nmt_path = Path(args.open_nmt_path)

    run_path = Path(args.run_dir)
    run_path.mkdir(exist_ok=True)

    # Generate metadata.json
    metadata = {
        "package_version": args.package_version,
        "argos_version": args.argos_version,
        "from_code": args.from_code,
        "from_name": args.from_name,
        "to_code": args.to_code,
        "to_name": args.to_name,
    }

    if 'corpus' in args.action:
        metadata_json = json.dumps(metadata, indent=4)
        with open(run_path / "metadata.json", "w") as metadata_file:
            metadata_file.write(metadata_json)

        raw_data = run_path / "raw_data/"
        raw_data.mkdir(exist_ok=True)

        dataset = CustomDataset("data/ilia-parallel-corpus_cleaned.csv")

        raw_source_path = raw_data / "source.txt"
        with raw_source_path.open('w', encoding='utf-8') as fp:
            fp.writelines((line + "\n" for line in dataset.source))

        raw_target_path = raw_data / "target.txt"
        with raw_target_path.open('w', encoding='utf-8') as fp:
            fp.writelines((line + "\n" for line in dataset.target))

        prepare_data(raw_source_path, raw_target_path, run_path)

        with (run_path / "split_data/all.txt").open("w") as combined:
            with (run_path / "split_data/src-train.txt").open('r') as src:
                for line in src:
                    combined.write(line)
            with (run_path / "split_data/tgt-train.txt").open('r') as tgt:
                for line in tgt:
                    combined.write(line)

        # TODO: Don't hardcode vocab_size and set user_defined_symbols
        run_process(
            [
                "spm_train",
                f"--input={run_path}/split_data/all.txt",
                f"--model_prefix={run_path}/sentencepiece",
                "--vocab_size=50000",
                "--character_coverage=0.9995",
                "--input_sentence_size=1000000",
                "--shuffle_input_sentence=true",
            ]
        )

        run_process(["rm", run_path / "split_data/all.txt"])

        run_process(["onmt_build_vocab", "-config", "config.yml", "-n_sample", "-1"])

    package_version_code = args.package_version.replace(".", "_")
    model_dir = f"translate-{args.from_code}_{args.to_code}-{package_version_code}"
    model_path = run_path / model_dir

    if 'train' in args.action:
        run_process(["onmt_train", "-config", "config.yml"])

        # Package
        run_process(
            [
                f"{open_nmt_path}/tools/average_models.py",
                "-m",
                f"{run_path}/openmt.model_step_25000.pt",
                f"{run_path}/openmt.model_step_30000.pt",
                "-o",
                f"{run_path}/averaged.pt",
            ]
        )

        run_process(
            [
                "ct2-opennmt-py-converter",
                "--model_path",
                f"{run_path}/averaged.pt",
                "--output_dir",
                f"{run_path}/model",
                "--quantization",
                "int8",
            ]
        )

        run_process(["mkdir", model_path])

        run_process(["cp", "-r", f"{run_path}/model", model_path])

        run_process(["cp", f"{run_path}/sentencepiece.model", model_path])

    if stanza_lang_code := args.get('stanza_lang_code', False):
        import stanza

        # Include a Stanza sentence boundary detection model
        stanza_model_located = False
        while not stanza_model_located:
            try:
                stanza.download(stanza_lang_code, model_dir=str(run_path / "stanza/"), processors="tokenize")
                stanza_model_located = True
            except:
                print(f"Could not locate stanza model for lang {stanza_lang_code}")
                print(
                    "Enter the code of a different language to attempt to use its stanza model."
                )
                print(
                    "This will work best for with a similar language to the one you are attempting to translate."
                )
                print(
                    "This will require manually editing the Stanza package in the finished model to change its code"
                )
                stanza_lang_code = input("Stanza language code (ISO 639): ")

        run_process(["cp", "-r", f"{run_path}/stanza", model_path])

    run_process(["cp", f"{run_path}/metadata.json", model_path])
    run_process(["cp", f"{run_path}/README.md", model_path])

    package_path = (
            Path("run") / f"translate-{args.from_code}_{args.to_code}-{package_version_code}.argosmodel"
    )

    shutil.make_archive(model_dir, "zip", root_dir="run", base_dir=model_dir)
    run_process(["mv", f'{model_dir}.zip', package_path])

    # Make .argoscheckpoint zip

    latest_checkpoint = opennmtutils.get_checkpoints()[-1]
    print(latest_checkpoint)
    print(latest_checkpoint.name)
    print(latest_checkpoint.num)

    print(f'Package saved to {package_path.resolve()}')
