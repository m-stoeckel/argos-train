from argostrain.dataset import *


def prepare_data(source_path: Path, target_path: Path, run_path: Path, valid_size=2000):
    # Build dataset
    dataset = FileDataset(source_path.open(), target_path.open())
    print("Read data from file")

    # Split and write data
    source_data, target_data = dataset.data()
    source_data = list(source_data)
    target_data = list(target_data)

    assert len(source_data) > valid_size

    split_data_path = run_path / "split_data/"
    split_data_path.mkdir(exist_ok=True)

    with (split_data_path / "src-val.txt").open("w") as source_valid_file:
        source_valid_file.writelines(source_data[:valid_size])
    with (split_data_path / "src-train.txt").open("w") as source_train_file:
        source_train_file.writelines(source_data[valid_size:])
    with (split_data_path / "tgt-val.txt").open("w") as target_valid_file:
        target_valid_file.writelines(target_data[:valid_size])
    with (split_data_path / "tgt-train.txt").open("w") as target_train_file:
        target_train_file.writelines(target_data[valid_size:])
    print("Done splitting data")
