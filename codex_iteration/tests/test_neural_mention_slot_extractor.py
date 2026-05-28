import numpy as np

from oci.models.neural_mention_slot_extractor import (
    NeuralMentionSlotConfig,
    NeuralMentionSlotExtractor,
    extract_mention_records,
)


def test_extract_mention_records_from_generic_note_shapes():
    texts = [
        "Patient: A\nAge: 70 years\n| Test | Result |\n| EGFR | wild type |",
        "Patient: B\nHemoglobin: 12.3 g/dL\nBrain MRI: no metastases",
    ]

    records = extract_mention_records(texts, max_mentions_per_patient=10)

    assert records
    assert {record.patient_index for record in records} == {0, 1}
    assert any("Age" in record.label for record in records)
    assert any("Brain MRI" in record.label for record in records)


def test_neural_mention_slot_extractor_uses_soft_patient_slots():
    texts = [
        "Age: 70 years\nHistology: adenocarcinoma",
        "Age: 61 years\nHistology: squamous",
        "Age: 75 years\nHistology: large cell",
    ]
    records = extract_mention_records(texts, max_mentions_per_patient=10)
    embeddings = np.eye(len(records), 4, dtype=np.float32)

    config = NeuralMentionSlotConfig(
        n_slots=3,
        top_assignments=2,
        assignment_temperature=0.5,
        random_state=3,
    )
    extractor = NeuralMentionSlotExtractor(config)
    features = extractor.fit_transform(records, embeddings, patient_indices=[0, 1, 2])

    assert features.shape == (3, 9)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert (features[:, :3].max(axis=1) > 0).all()
