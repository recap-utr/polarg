from __future__ import annotations

import shutil
from pathlib import Path

import typer
from sklearn.model_selection import train_test_split

app = typer.Typer()

RANDOM_STATE = 0


@app.command()
def split(
    input: Path,
    pattern: str,
    output: Path,
    training_size: float = 0.8,
    skip_validation: bool = False,
):
    training_output = output / "training"
    test_output = output / "test"
    validation_output = output / "validation"

    for folder in (training_output, test_output, validation_output):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

    files = list(input.glob(pattern))
    training_files, test_files = train_test_split(
        files,
        train_size=training_size,
        random_state=RANDOM_STATE,
    )
    validation_files = []

    if not skip_validation:
        training_files, validation_files = train_test_split(
            training_files,
            train_size=training_size,
            random_state=RANDOM_STATE,
        )

    file: Path

    for file in training_files:
        shutil.copyfile(file, training_output / file.relative_to(input))

    for file in test_files:
        shutil.copyfile(file, test_output / file.relative_to(input))

    for file in validation_files:
        shutil.copyfile(file, validation_output / file.relative_to(input))


if __name__ == "__main__":
    app()
