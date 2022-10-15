import os
from pathlib import Path

import cohere
import pandas as pd
import pytest

from .classify import FewShotClassify


@pytest.fixture
def cohere_client():
    try:
        with Path("cohere_api_key").open("r") as f:
            cohere_api_key = f.read().strip()
    except FileNotFoundError:
        cohere_api_key = os.environ["COHERE_API_KEY"]
    return cohere.Client(cohere_api_key)


@pytest.fixture
def error_df():
    df = pd.DataFrame(
        {
            "text": [
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
                "this is a test",
            ],
            "label": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        }
    )
    return df


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "text": [
                "this is a test 1",
                "this is a test 2",
                "this is a test 3",
                "this is a test 4",
                "this is a test 5",
                "this is a test 6",
                "this is a test 7",
                "this is a test 8",
                "this is a test 9",
                "this is a test 10",
                "this is a test 11",
                "this is a test 12",
            ],
            "label": [
                "a",
                "a",
                "a",
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                "b",
                "b",
            ],
        }
    )
    return df


def test_makes_train_dataframes_raise_errors(error_df, cohere_client):
    train_counts = [2, 4]
    test_count = 2
    with pytest.raises(ValueError):
        FewShotClassify(
            error_df,
            cohere_client,
            train_counts,
            test_count,
            x_label="text",
            y_label="label",
        )


def test_makes_train_dataframes(df, cohere_client):
    train_counts = [2, 4]
    test_count = 2
    cfc = FewShotClassify(
        df,
        cohere_client,
        train_counts,
        test_count,
        x_label="text",
        y_label="label",
    )

    assert len(cfc.train_dfs) == 2
    assert len(cfc.train_dfs[0]) == 4
    assert len(cfc.train_dfs[1]) == 8
    assert len(cfc.test_df) == 4
    assert len(cfc.test_df[cfc.test_df["label"] == "a"]) == 2
    assert len(cfc.test_df[cfc.test_df["label"] == "b"]) == 2
