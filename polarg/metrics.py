import json
import typing as t
from pathlib import Path

import typer
from arg_services.mining.v1beta import entailment_pb2
from sklearn import metrics

app = typer.Typer()


@app.command()
def main(
    path: Path,
):
    labels = load(path)
    show(labels)


LabelList = list[entailment_pb2.EntailmentType.ValueType]


class LabelsEntry(t.TypedDict):
    predicted: LabelList
    true: LabelList


class Labels(t.TypedDict):
    raw: LabelsEntry
    filtered: LabelsEntry


def serialize(
    predicted_labels: LabelList,
    true_labels: LabelList,
) -> Labels:
    filtered_true_labels = [
        label
        for idx, label in enumerate(true_labels)
        if predicted_labels[idx]
        != entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
    ]

    filtered_predicted_labels = [
        label
        for idx, label in enumerate(predicted_labels)
        if predicted_labels[idx]
        != entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
    ]

    serialized_labels: Labels = {
        "raw": {
            "predicted": predicted_labels,
            "true": true_labels,
        },
        "filtered": {
            "predicted": filtered_predicted_labels,
            "true": filtered_true_labels,
        },
    }

    return serialized_labels


def dump(
    labels: Labels,
    path: Path,
):
    with path.with_suffix(".json").open("w") as f:
        json.dump(labels, f)

    return labels


def load(path: Path) -> Labels:
    with path.with_suffix(".json").open("r") as f:
        data = json.load(f)

    return data


def show(
    labels: Labels,
):
    len_all_labels = len(labels["raw"]["true"])
    len_known_labels = len(labels["filtered"]["true"])
    len_unknown_labels = len_all_labels - len_known_labels

    label_types = sorted(
        set(labels["filtered"]["predicted"]).union(labels["filtered"]["true"])
    )

    found_labels = [entailment_pb2.EntailmentType.Name(label) for label in label_types]
    percent_unknown_labels = len_unknown_labels / len_all_labels
    accuracy = metrics.accuracy_score(
        labels["filtered"]["true"], labels["filtered"]["predicted"]
    )

    precision_recall_args = {}

    if len(label_types) != 2:
        precision_recall_args = {
            "average": None,
            "labels": label_types,
        }

    precision = metrics.precision_score(
        labels["filtered"]["true"],
        labels["filtered"]["predicted"],
        **precision_recall_args,
    )
    recall = metrics.recall_score(
        labels["filtered"]["true"],
        labels["filtered"]["predicted"],
        **precision_recall_args,
    )

    typer.echo(f"Labels: {found_labels}")
    typer.echo(f"Unknown labels: {len_unknown_labels} ({percent_unknown_labels:.2%})")
    typer.echo(f"Accuracy: {accuracy:.3f}")

    if len(label_types) != 2:
        typer.echo(f"Precision: {precision}")
        typer.echo(f"Recall: {recall}")
    else:
        typer.echo(f"Precision: {precision:.3f}")
        typer.echo(f"Recall: {recall:.3f}")


if __name__ == "__main__":
    app()
