import random
from argparse import ArgumentParser
from pathlib import Path

from conllu.models import Token, TokenList
from stanza.utils.datasets import common as datasets_common
from stanza.utils.datasets.prepare_tokenizer_treebank import process_treebank, add_specific_args
from stanza.utils.training import common as training_common
from stanza.utils.training.run_tokenizer import run_treebank
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tqdm import tqdm

from argostrain.custom_dataset import CustomDataset


def new_token(idx: int, form: str):
    return Token({
        'id': idx,
        'form': form,
        'lemma': "_",
        'upos': "_",
        'xpos': "_",
        'feats': "_",
        'head': "_",
        'deprel': "_",
        'deps': "_",
        'misc': "_",
    })


def create_dataset():
    pre_tokenizer = BertPreTokenizer()
    normalizer = BertNormalizer(lowercase=False)

    georgian = CustomDataset("data/ilia-parallel-corpus_cleaned.csv").data()[0]

    plain_text = []
    dataset = []
    for line in tqdm(georgian, desc="Preprocessing Dataset"):
        line = line.strip()
        if "\n" in line:
            continue

        plain_text.append(line)

        tokens = pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(line))

        dataset.append(TokenList([new_token(i, t) for i, (t, _) in enumerate(tokens, start=1)]))

    random.shuffle(plain_text)
    random.shuffle(dataset)

    size = len(plain_text) // 10
    train_size = size * 8
    dev_size = size * 9

    train_text, train_dataset = plain_text[:train_size], dataset[:train_size]
    dev_text, dev_dataset = plain_text[train_size:dev_size], dataset[train_size:dev_size]
    test_text, test_dataset = plain_text[dev_size:], dataset[dev_size:]

    Path("extern_data/ud2/ud-treebanks-v2.7/ka-ilia/").mkdir(parents=True, exist_ok=True)
    prefix = "extern_data/ud2/ud-treebanks-v2.7"

    datasets = [
        ("ud-train", train_text, train_dataset),
        ("ud-dev", dev_text, dev_dataset),
        ("ud-test", test_text, test_dataset)
    ]
    for name, txt, data in tqdm(datasets, desc="Writing Splits", position=0):
        with Path(f"{prefix}/ka-ilia/ka_ilia-{name}.txt").open('w', encoding='utf-8') as ftxt, \
                Path(f"{prefix}/ka-ilia/ka_ilia-{name}.conllu").open('w', encoding='utf-8') as fconllu:
            for line, sample in tqdm(zip(txt, data), total=len(txt), desc=f"Writing '{name}'", position=1, leave=False):
                ftxt.write(line.strip() + " ")

                fconllu.write(f"# text = {line.strip()}\n")
                fconllu.write(sample.serialize())


def add_args(parser: ArgumentParser):
    parser.add_argument('--use_mwt', action='store_true', default=False)


if __name__ == '__main__':
    # create_dataset()
    # datasets_common.main(process_treebank, add_specific_args)

    training_common.main(run_treebank, "tokenize", "tokenizer", add_args)
