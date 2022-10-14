"""
Classification Utils built over Cohere API

Raises:
    ValueError: In cases of invalid test counts or other size mismatches
"""
from pathlib import Path
from typing import List, Tuple

import cohere
import pandas as pd
from cohere.classify import Example
from tqdm.notebook import tqdm

Path.ls = lambda x: list(x.iterdir())  # type: ignore


class CohereBase:
    """
    Base Utils for the Cohere API
    """

    def make_examples(self, df: pd.DataFrame) -> List[Example]:
        """
        #TODO: Iterative and slow, make this parallel and fast
        """
        examples = []
        for row in df.iterrows():
            text = row[1]["Name"]
            lbl = row[1]["Key"]
            examples.append(Example(text, lbl))
        return examples

    @staticmethod
    def parse_cohere_classification(
        classification: cohere.classify.Classification,
    ) -> Tuple[str, float]:
        """
        Parse a single classification from Cohere Response into a tuple of (label, score)

        Args:
            classification (cohere.classify.Classification): Single classification object

        Returns:
            Tuple[str, float]: lbl, score for a single classification
        """
        lbl = classification.prediction
        confidences = classification.confidence
        score = -1
        for confidence in confidences:
            score = max(score, confidence.confidence)
            # why is this max? lazy hack since there is no prediction confidence score in the response
        return lbl, score

    @staticmethod
    def parse_cohere_response(
        response: cohere.classify.Classifications,
    ) -> Tuple[List[str], List[float]]:
        """
        Parse a Cohere Response into a tuple of (labels, scores)

        Args:
            response ():

        Returns:
            Tuple[List[str], List[float]]: _description_
        """
        lbls, scores = [], []
        for classification in response.classifications:
            lbl, score = CohereBase.parse_cohere_classification(classification)
            lbls.append(lbl)
            scores.append(score)
        return lbls, scores

    def __repr__(self) -> str:
        return f"Train Counts: {self.train_counts}, Test Count: {self.test_count}, Text Column(x_label): {self.x_label}, Target Column(y_label): {self.y_label}"


class CohereFewShotClassify(CohereBase):
    """
    Data Management for Few Shot Classification using the Cohere API

    Inherits:
        CohereBase (CohereBase):  Base Utils for the Cohere API
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cohere_client: cohere.Client,
        train_counts: List[int] = [4, 8, 16, 32],
        test_count: int = 64,
        x_label: str = "Name",
        y_label: str = "Key",
    ) -> None:
        # pylint: disable=too-many-arguments, dangerous-default-value
        """
        Initialize the Few Shot Classification Data Manager

        Args:
            df (pd.DataFrame): Dataframe with all tagged samples. This will be split into train and test as needed
            co (cohere.Client): cohere Client object
            train_counts (List[int], optional): Defaults to [4, 8, 16, 32].
            test_count (int, optional): Number of test samples. Defaults to 64.
            x_label (str, optional): text column. Defaults to "Name".
            y_label (str, optional): label column. Defaults to "Key".

        Returns:
            None
        """
        self.df, self.train_counts, self.test_count, self.x_label, self.y_label = (
            df,
            train_counts,
            test_count,
            x_label,
            y_label,
        )
        self.cohere_client = cohere_client
        self.labels = list(self.df[y_label].unique())
        self.random_state = 37
        self.train_dfs, self.test_df = self.make_train_dataframes(
            self.train_counts, self.test_count, self.labels
        )

    def __repr__(self) -> str:
        return f"CohereFewShotClassify({super().__repr__()})"

    def make_train_dataframes(
        self, train_counts: List[int], test_count: int, labels: List[str]
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Make train dataframes for each label, keep a test_df as well

        Args:
            train_counts (List[int]): Number of examples to keep for training for each label
            test_count (int): Number of examples to keep for test for each label
            labels (List[str]): List of labels to keep

        Raises:
            ValueError:

        Returns:
            Tuple[List[pd.DataFrame], pd.DataFrame]: List of train dataframes, test dataframe
        """
        train_dfs = []
        for n in train_counts:
            trn = []
            test_lbl_cuts = []
            for lbl in labels:
                class_cut = self.df[self.df[self.y_label] == lbl]
                if len(class_cut) <= test_count:
                    raise ValueError(f"For label {lbl} insufficient number of samples")
                test_cut = class_cut.sample(test_count, random_state=self.random_state)
                test_lbl_cuts.append(test_cut)
                left_over = class_cut[
                    ~class_cut.apply(tuple, 1).isin(test_cut.apply(tuple, 1))
                ]
                trn.append(left_over.sample(n, random_state=self.random_state))
            train_dfs.append(pd.concat(trn))
        test_df = pd.concat(test_lbl_cuts)
        return train_dfs, test_df

    def predict(self) -> List[cohere.classify.Classifications]:
        """
        Predict on the test set

        Returns:
            cohere.response: COHERE_RESPONSE
        """
        responses = []
        for trn_df in tqdm(self.train_dfs):
            inputs = self.test_df[self.x_label].tolist()
            examples = self.make_examples(trn_df)
            response = self.cohere_client.classify(inputs=inputs, examples=examples)
        responses.append(response)
        return responses


# cohere_clf = CohereFewShotClassify(
#     df=df,
#     co=co,
#     train_counts=[4, 8, 16, 32],
#     test_count=64,
#     x_label="Name",
#     y_label="Key",
# )

# cohere_clf.predict()
# assert len(test_df) == 64*len(labels)
# assert len(train_cuts) == len(train_dfs)
# assert train_cuts[0]*len(labels) == len(train_dfs[0])
