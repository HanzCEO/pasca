from pasca import metrics

SAMPLE = """\
Have you ever played a game that makes you scared of a blink? I have. It's honestly crazy.

The engine was old, and the budget was tiny, but the atmosphere was literally perfect. You don't get safe rooms.
So you keep moving. Furthermore, the sound design does not let you rest.
"""


def test_split_sentences():
    sentences = metrics.split_sentences(SAMPLE)
    assert sentences[0].endswith("?")
    assert sentences[1] == "I have."
    assert len(sentences) == 7


def test_profile_counts():
    p = metrics.profile(SAMPLE)
    assert p["size"]["sentences"] == 7
    assert p["size"]["paragraphs"] == 2
    assert p["rhythm"]["questions_pct_of_sentences"] == round(100 / 7, 1)
    assert p["voice"]["second_person_per_1k"] > 0
    assert p["voice"]["contractions_per_1k"] > 0
    assert p["voice"]["expanded_forms_per_1k"] > 0  # "does not"
    assert "honestly" in p["stance"]["top_boosters"]
    assert p["sentence_openers_pct"]["formal_conjunctive"] > 0
    assert p["sentence_openers_pct"]["causal"] > 0


def test_profile_empty_text():
    p = metrics.profile("")
    assert p["size"]["words"] == 0
    assert p["rhythm"]["mean_sentence_length"] == 0.0


def test_mtld_rewards_diversity():
    repetitive = ("the cat sat " * 40).split()
    varied = [f"word{i}" for i in range(120)]
    assert metrics.mtld(varied) > metrics.mtld(repetitive)


def test_syllables():
    assert metrics.count_syllables("cat") == 1
    assert metrics.count_syllables("atmosphere") == 3
