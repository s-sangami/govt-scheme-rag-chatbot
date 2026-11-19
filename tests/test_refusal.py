from rag_pipeline import is_refusal


def test_refusal_detected():
    assert is_refusal("I could not find this information in the document.")
    assert is_refusal("  i could not find this information in the document ")


def test_real_answer_not_refusal():
    assert not is_refusal("PM-KISAN gives Rs 6000 per year in three installments.")
