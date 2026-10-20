"""Black-box behavioral evaluation of the trip-budget / destination-weather agent.

Derived solely from spec.md (030_math_weather). Only the public entrypoint
`agent(user_request: str) -> str` is used; no implementation details are assumed.

Compatible with DeepEval 3.3.9 (LLMTestCase, GEval, LLMTestCaseParams, assert_test).
Run with:  deepeval test run test_deepeval.py   (or plain pytest)
"""

import re

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

JUDGE_MODEL = "gpt-4o"
DESTINATIONS = ("Curitiba", "Lisbon", "Tokyo")


def expected_destination(total: float) -> str:
    # select_location rule from the spec
    if total < 500:
        return "Curitiba"
    if total <= 1000:
        return "Lisbon"
    return "Tokyo"


def run(request: str) -> str:
    output = agent(request)
    assert isinstance(output, str), "Entrypoint must return str"
    return output


def number_present(text: str, value: float) -> bool:
    normalized = text.replace(",", "")
    for match in re.findall(r"\d+(?:\.\d+)?", normalized):
        if abs(float(match) - value) < 1e-6:
            return True
    return False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def success_response_metric() -> GEval:
    return GEval(
        name="Budget-Destination-Weather Response",
        criteria=(
            "The actual output must answer a trip request by stating (1) the total "
            "trip budget, (2) the selected destination, and (3) weather information "
            "for that same destination, OR state that weather data is unavailable "
            "for that destination. The total budget and destination stated must "
            "match the expected output. The destination named in the response must "
            "be the one the weather refers to."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
        threshold=0.7,
    )


def no_fabrication_metric() -> GEval:
    return GEval(
        name="Single Consistent Destination",
        criteria=(
            "The actual output must refer to exactly one selected destination, and "
            "any weather information must be about that destination only. The "
            "output must not present weather for other cities nor propose a "
            "different destination than the one stated in the expected output."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=JUDGE_MODEL,
        threshold=0.7,
    )


def invalid_request_metric() -> GEval:
    return GEval(
        name="Invalid Request Handling",
        criteria=(
            "The input is empty, ambiguous, missing the daily budget or number of "
            "days, or unrelated to trip-budget and destination-weather inquiries. "
            "The actual output must explain that the request cannot be completed "
            "and, when the daily budget or number of days is missing, ask for the "
            "missing value(s). The output must NOT state a computed total budget, "
            "must NOT select a destination (Curitiba, Lisbon, Tokyo), and must NOT "
            "provide weather information."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=JUDGE_MODEL,
        threshold=0.7,
    )


# --------------------------------------------------------------------------- #
# B1-B4 (+B7 tolerated): valid requests -> total budget, destination, weather
# --------------------------------------------------------------------------- #

VALID_CASES = [
    # (request, daily, days, extra)
    ("I have a daily budget of 50 for 4 days, plus 100 extra. "
     "Which destination can I go to and what's the weather there?", 50, 4, 100),
    ("My daily budget is 100 and I'll travel for 5 days, no extra money. "
     "Pick a destination for me and tell me the weather.", 100, 5, 0),
    ("Daily budget 200, trip of 5 days, with 300 in reserve. "
     "Where should I go and how is the weather there?", 200, 5, 300),
    ("I can spend 80 per day for 6 days. What destination fits my budget "
     "and what's the weather like?", 80, 6, 0),
    ("Daily budget 150 for 3 days and 50 extra. Choose my destination "
     "and give me its weather.", 150, 3, 50),
]


@pytest.mark.parametrize("request_text,daily,days,extra", VALID_CASES)
def test_valid_request_total_and_destination(request_text, daily, days, extra):
    """B1/B2/B3 invariants: total = daily*days + extra (0 if omitted);
    destination follows select_location thresholds."""
    total = daily * days + extra
    destination = expected_destination(total)
    output = run(request_text)

    assert number_present(output, total), (
        f"Expected total budget {total} in output:\n{output}"
    )
    assert destination.lower() in output.lower(), (
        f"Expected destination {destination} in output:\n{output}"
    )
    for other in DESTINATIONS:
        if other != destination:
            assert other.lower() not in output.lower(), (
                f"Unexpected destination {other} in output:\n{output}"
            )


@pytest.mark.parametrize("request_text,daily,days,extra", VALID_CASES)
def test_valid_request_response_quality(request_text, daily, days, extra):
    """B4/B7: response states total budget, selected destination, and its weather
    (or that weather data is unavailable there)."""
    total = daily * days + extra
    destination = expected_destination(total)
    output = run(request_text)
    test_case = LLMTestCase(
        input=request_text,
        actual_output=output,
        expected_output=(
            f"Total budget: {total}. Selected destination: {destination}. "
            f"Weather information for {destination} (or a statement that "
            f"weather data is unavailable for {destination})."
        ),
    )
    assert_test(test_case, [success_response_metric(), no_fabrication_metric()])


# --------------------------------------------------------------------------- #
# select_location boundaries (Curitiba < 500 <= Lisbon <= 1000 < Tokyo)
# --------------------------------------------------------------------------- #

BOUNDARY_CASES = [
    ("Daily budget 99 for 5 days, plus 4 extra. Which destination and what's the weather?", 499),
    ("Daily budget 100 for 5 days, no extra. Which destination and what's the weather?", 500),
    ("Daily budget 100 for 10 days, no extra. Which destination and what's the weather?", 1000),
    ("Daily budget 100 for 10 days, plus 1 extra. Which destination and what's the weather?", 1001),
]


@pytest.mark.parametrize("request_text,total", BOUNDARY_CASES)
def test_destination_threshold_boundaries(request_text, total):
    output = run(request_text)
    destination = expected_destination(total)
    assert number_present(output, total), f"Expected total {total}:\n{output}"
    assert destination.lower() in output.lower(), (
        f"Total {total} must select {destination}:\n{output}"
    )


def test_omitted_extra_treated_as_zero():
    """B2: omitted extra amount is treated as zero."""
    output = run("I have 70 per day for 7 days. Where should I go and what's the weather?")
    assert number_present(output, 490)
    assert "curitiba" in output.lower()


# --------------------------------------------------------------------------- #
# B5: invalid / incomplete / off-task requests
# --------------------------------------------------------------------------- #

INVALID_CASES = [
    "",
    "I want to travel for 5 days. What's the weather at my destination?",  # no daily budget
    "My daily budget is 100. Where should I go and what's the weather?",    # no days
    "Plan me a trip somewhere nice.",                                        # ambiguous
    "What is the capital of France?",                                        # off-task
]


@pytest.mark.parametrize("request_text", INVALID_CASES)
def test_invalid_request_is_refused(request_text):
    output = run(request_text)
    lowered = output.lower()
    for city in DESTINATIONS:
        assert city.lower() not in lowered, (
            f"Invalid request must not yield a destination ({city}):\n{output}"
        )
    test_case = LLMTestCase(input=request_text or "<empty request>", actual_output=output)
    assert_test(test_case, [invalid_request_metric()])


@pytest.mark.parametrize(
    "request_text,missing",
    [
        ("I want to travel for 5 days. What's the weather at my destination?", "daily budget"),
        ("My daily budget is 100. Where should I go and what's the weather?", "number of days"),
    ],
)
def test_missing_value_is_requested(request_text, missing):
    output = run(request_text)
    metric = GEval(
        name="Asks For Missing Value",
        criteria=(
            f"The actual output must explain the request cannot be completed and "
            f"explicitly ask the user to provide the missing {missing}."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=JUDGE_MODEL,
        threshold=0.7,
    )
    assert_test(LLMTestCase(input=request_text, actual_output=output), [metric])
