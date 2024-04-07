import json
import typing as t
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar
import typer
from arg_services.mining.v1beta import entailment_pb2
from sklearn import metrics

app = typer.Typer()


@app.command()
def main(
    path: Path,
):
    labels = load(path)
    compute(labels, show=True)


@app.command()
def significance(
    path: Path,
    pattern1: t.Annotated[list[str], typer.Option(default_factory=list)],
    pattern2: t.Annotated[list[str], typer.Option(default_factory=list)],
):
    pattern1_labels: list[LabelsEntry] = []
    pattern2_labels: list[LabelsEntry] = []

    for pattern in pattern1:
        for file in sorted(path.glob(pattern)):
            pattern1_labels.append(load(file)["raw"])

    for pattern in pattern2:
        for file in sorted(path.glob(pattern)):
            pattern2_labels.append(load(file)["raw"])

    assert len(pattern1_labels) == len(pattern2_labels)
    y1 = []
    y2 = []
    y_unknown = 0

    for labels1, labels2 in zip(pattern1_labels, pattern2_labels):
        assert labels1["true"] == labels2["true"]
        true_labels = labels1["true"]

        for idx, (predicted1, predicted2) in enumerate(
            zip(labels1["predicted"], labels2["predicted"])
        ):
            if (
                predicted1 == entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
                or predicted2
                == entailment_pb2.EntailmentType.ENTAILMENT_TYPE_UNSPECIFIED
            ):
                y_unknown += 1
                continue

            true_label = true_labels[idx]
            y1.append(predicted1 == true_label)
            y2.append(predicted2 == true_label)

    contigency_table = metrics.confusion_matrix(y1, y2)
    mcnemar_result = mcnemar(contigency_table, exact=False, correction=True)

    print("Known labels:", len(y1))
    print("Unknown labels:", y_unknown)
    print(contigency_table)
    print(mcnemar_result)


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


def load(path: Path) -> Labels:
    with path.with_suffix(".json").open("r") as f:
        data = json.load(f)

    return data


def compute(
    labels: Labels,
    show: bool = True,
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

    final_metrics = {
        "labels": found_labels,
        "unknown_labels": len_unknown_labels,
        "accuracy": accuracy,
    }

    if len(label_types) == 2:
        final_metrics.update(
            {
                "precision": precision,
                "recall": recall,
            }
        )

    if show:
        typer.echo(f"Labels: {found_labels}")
        typer.echo(
            f"Unknown labels: {len_unknown_labels} ({percent_unknown_labels:.2%})"
        )
        typer.echo(f"Accuracy: {accuracy:.3f}")

        if len(label_types) != 2:
            typer.echo(f"Precision: {precision}")
            typer.echo(f"Recall: {recall}")
        else:
            typer.echo(f"Precision: {precision:.3f}")
            typer.echo(f"Recall: {recall:.3f}")

    return final_metrics


if __name__ == "__main__":
    app()
