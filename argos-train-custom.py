import argparse
import glob
import hashlib
import shutil
import subprocess
import tempfile
from typing import List

import yaml

from argostrain import opennmtutils
from argostrain.custom_dataset import CustomDataset
from argostrain.data import prepare_data
from argostrain.dataset import *


def run_process(arguments, *args, **kwargs):
    arguments = [str(arg) for arg in arguments]
    print(" ".join(arguments))
    return subprocess.run(arguments, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("action", nargs="+", type=str, choices=('corpus', 'train', 'pack'))

    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--corpus", default="data/ilia-parallel-corpus_cleaned.csv")

    parser.add_argument('--custom_stanza_tokenizer', type=str, default=None, required=False)
    parser.add_argument('--stanza_lang_code', type=str, required=False)
    parser.add_argument('--open_nmt_path', type=str, default='./../OpenNMT-py/', required=False)

    parser.add_argument('--no_average', action='store_true', default=False)

    # OpenNMT options
    parser.add_argument('--train_from', type=str)

    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.load(config_path.open('r', encoding='utf-8'), yaml.FullLoader)

    open_nmt_path = Path(args.open_nmt_path)

    run_path = Path(config['run_dir'])
    run_path.mkdir(exist_ok=True)

    from_code = config.get('from_code', 'ka')
    to_code = config.get('to_code', 'en')
    from_name = config.get('from_name', 'Georgian')
    to_name = config.get('to_name', 'English')
    package_version = config.get('package_version', '1.0')
    argos_version = config.get('argos_version', '1.5')

    try:
        if 'train' in args.action:
            omnt_options = []
            if (train_from := args.train_from) is not None:
                omnt_options.extend([
                    '--train_from',
                    str(Path(train_from).absolute())
                ])
            if (batch_size := args.batch_size) is not None:
                omnt_options.extend([
                    '--batch_size',
                    str(Path(batch_size).absolute())
                ])

            run_process(["onmt_train", "-config", config_path] + omnt_options)

            saved_models = glob.glob(config.get('save_model', f"{run_path}/openmt.model") + "_step_*")
            saved_models: List[Path] = list(map(lambda name: Path(name).absolute(), saved_models))
            saved_models.sort(
                key=lambda path: int(path.name.split("_")[-1].split(".")[0]),
                reverse=True
            )
            if not args.no_average and (save_checkpoint_steps := config.get('save_checkpoint_steps', False)):
                last_two_models = saved_models[:2]

                run_process(
                    [
                        f"{open_nmt_path}/tools/average_models.py",
                        "-m",
                        f"{last_two_models[1]}",
                        f"{last_two_models[0]}",
                        "-o",
                        f"{run_path}/averaged.pt",
                    ]
                )
            else:
                run_process(
                    [
                        "ln",
                        f"{saved_models[0]}"
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

        if 'pack' in args.action:
            metadata = {
                "package_version": args.package_version,
                "argos_version": args.argos_version,
                "from_code": args.from_code,
                "from_name": args.from_name,
                "to_code": args.to_code,
                "to_name": args.to_name,
            }
            metadata_json = json.dumps(metadata, indent=4)
            with open(run_path / "metadata.json", "w") as metadata_file:
                metadata_file.write(metadata_json)

            package_version_code = args.package_version.replace(".", "_")
            from_to = f"{args.from_code}_{args.to_code}"
            model_dir = f"translate-{from_to}-{package_version_code}/{from_to}"
            package_path = (run_path / model_dir).absolute()

            package_path.mkdir(parents=True, exist_ok=True)
            model_path = package_path / "model"
            model_path.mkdir(exist_ok=True)

            run_process(["ln", *glob.glob(f"{run_path}/model/*"), model_path])

            run_process(["ln", f"{run_path}/sentencepiece.model", package_path])

            if args.custom_stanza_tokenizer is not None:
                stanza_lang_code = args.stanza_lang_code if args.stanza_lang_code is not None else args.from_code

                tokenize_model = Path(args.custom_stanza_tokenizer).absolute()

                tokenize_path = (package_path / f"stanza/{stanza_lang_code}/tokenize/")
                tokenize_path.mkdir(parents=True, exist_ok=True)

                run_process(
                    [
                        "ln",
                        f"{tokenize_model}",
                        f"{tokenize_path}/"
                    ]
                )

                with tokenize_model.open('rb') as fp:
                    md5_sum = hashlib.md5(fp.read()).hexdigest().lower()

                tokenize_name = tokenize_model.name
                if tokenize_name.endswith(".pt"):
                    tokenize_name = tokenize_name[:-3]

                with Path(f"{package_path}/stanza/{stanza_lang_code}/resoures.json").open('w', encoding='utf-8') as fp:
                    resources_json = {
                        stanza_lang_code: {
                            "tokenize": {
                                tokenize_name: {
                                    "md5": md5_sum
                                }
                            },
                            "default_processors": {
                                "tokenize": tokenize_name
                            },
                            "default_dependencies": {

                            },
                            "default_md5": md5_sum,
                            "lang_name": args.from_name
                        },
                        "url": "https://www.texttechnologylab.org/"
                    }
                    json.dump(resources_json, fp, indent=4)

            elif (stanza_lang_code := args.stanza_lang_code) is not None:
                import stanza

                # Include a Stanza sentence boundary detection model
                stanza_model_located = False
                while not stanza_model_located:
                    try:
                        stanza.download(stanza_lang_code, dir=str(run_path / "stanza/"), processors="tokenize")
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

                run_process(["cp", "-r", f"{run_path}/stanza", package_path])

            run_process(["cp", f"{run_path}/metadata.json", package_path])
            run_process(["cp", f"{run_path}/README.md", package_path])
            run_process(["cp", f"{run_path}/metadata.json", package_path])
            package_name = f"translate-{args.from_code}_{args.to_code}-{package_version_code}"
            package_destination = Path(f"{package_name}.argosmodel")

            shutil.make_archive(package_name, "zip", root_dir=str(package_path.parent), base_dir=str(package_path.name))
            run_process(["mv", f'{package_name}.zip', package_destination])

            print(f'Package saved to {package_destination.resolve()}')
    finally:
        config_path.unlink(missing_ok=True)
