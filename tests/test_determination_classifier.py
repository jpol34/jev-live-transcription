from jev_live_transcription import determination_classifier


def test_unregistered_field_is_always_uncertain():
    assert determination_classifier.classify("caller_name", "yes, that's correct") == "uncertain"


def test_empty_window_is_uncertain():
    assert determination_classifier.classify("permission_to_enter", "") == "uncertain"


def test_call_005_indirect_affirmative_with_no_literal_yes():
    # The exact phrasing from call 005's ground-truth-confirmed "yes" -- the caller never says
    # the word "yes" at all, which is precisely why GLiNER's own span extraction finds nothing
    # here (see the module docstring): this is the no-GLiNER-signal fallback path in action.
    window = (
        "Caller: Uh, I guess it's fine if I'm not there. Just, uh, fix the leak, please. "
        "Agent: Understood, I'll make a note that they have permission to enter."
    )
    assert determination_classifier.classify("permission_to_enter", window) == "yes"


def test_indirect_affirmative_without_agent_confirmation():
    window = "Caller: I guess it's fine if I'm not there, just go ahead and fix it."
    assert determination_classifier.classify("permission_to_enter", window) == "yes"


def test_no_need_to_call_first_is_affirmative():
    # Call 032's phrasing -- contains the literal word "no", but grants entry. A naive
    # bare-"no"-as-negative rule would misclassify this; the rules only look for explicit
    # entry-refusal phrasing, never a bare "no".
    window = "Caller: Oh, um, no need to call first. Just come by when you can fix it."
    assert determination_classifier.classify("permission_to_enter", window) == "yes"


def test_explicit_refusal_is_negative():
    window = "Caller: No, please don't come in, I need to be there when you fix it."
    assert determination_classifier.classify("permission_to_enter", window) == "no"


def test_must_be_present_is_negative():
    window = "Caller: I have to be home when the maintenance team comes by."
    assert determination_classifier.classify("permission_to_enter", window) == "no"


def test_callers_own_question_about_being_present_is_not_a_refusal():
    # Regression: call 005's caller asks "should I... need to be there when they come over?"
    # before ever answering -- this must not read as a negative determination just because the
    # question contains the same words a genuine refusal would use.
    window = (
        "Caller: Oh, okay. Uh, should I... like, need to be there when they come over? "
        "Agent: Good question. Do you prefer to be present, or can they enter if you're not home?"
    )
    assert determination_classifier.classify("permission_to_enter", window) == "uncertain"


def test_unrelated_window_is_uncertain():
    window = "Caller: My name is Jane and my unit number is 204."
    assert determination_classifier.classify("permission_to_enter", window) == "uncertain"


def test_contradictory_signals_are_uncertain():
    window = "Caller: I have to be home, but I guess it's fine if I'm not there too."
    assert determination_classifier.classify("permission_to_enter", window) == "uncertain"


def test_classify_is_case_insensitive():
    window = "CALLER: GO AHEAD AND ENTER, NO PROBLEM."
    assert determination_classifier.classify("permission_to_enter", window) == "yes"
