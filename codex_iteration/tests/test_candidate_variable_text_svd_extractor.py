import numpy as np
import pandas as pd

from oci.models.candidate_variable_text_svd_extractor import (
    CandidateVariableTextSVDConfig,
    CandidateVariableTextSVDExtractor,
)


def test_candidate_variable_text_svd_extractor_shapes():
    df = pd.DataFrame(
        {
            "clinical_text": [
                "baseline note age 70 adenocarcinoma egfr wild type",
                "baseline note age 65 squamous egfr unknown",
                "follow up note age 72 adenocarcinoma egfr positive",
                "follow up note age 60 large cell egfr wild type",
                "baseline oncology note age 67 squamous egfr wild type",
                "staging note age 75 adenocarcinoma egfr unknown",
            ],
            "llm_extracted_age": ["70", "65", "72", "60", "67", "75"],
            "llm_extracted_histology": [
                "adenocarcinoma",
                "squamous",
                "adenocarcinoma",
                "large cell",
                "squamous",
                "adenocarcinoma",
            ],
        }
    )
    config = CandidateVariableTextSVDConfig(
        w_dim=2,
        max_features=50,
        min_df=1,
    )
    extractor = CandidateVariableTextSVDExtractor(config, random_state=7)

    extractor.fit(df.iloc[:5])
    features = extractor.transform(df.iloc[5:])

    assert features["X"].shape[0] == 1
    assert features["W"].shape == (1, 2)
    assert features["X"].dtype == np.float32
    assert features["W"].dtype == np.float32
    assert extractor.numeric_columns_ == ["llm_extracted_age"]
    assert extractor.categorical_columns_ == ["llm_extracted_histology"]
