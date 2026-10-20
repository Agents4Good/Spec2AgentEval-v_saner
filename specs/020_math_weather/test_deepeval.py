"""Black-box behavioral evaluation of the math/weather agent (spec 020).

Only the public entrypoint `agent(user_request: str) -> str` is used.
Compatible with DeepEval 3.3.9 (GEval + LLMTestCase + assert_test).

Run: pytest test_deepeval.py   (or: deepeval test run test_deepeval.py)
Requires OPENAI_API_KEY for the agent and the GEval judge.
"""

import re

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

JUDGE_MODEL = "gpt-4o"
THRESHOLD = 0.7
PARAMS = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]


def _metric(name: str, criteria: str) -> GEval:
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=PARAMS,
        model=JUDGE_MODEL,
        threshold=THRESHOLD,
    )


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def _contains_number(text: str, value: float) -> bool:
    return any(abs(n - value) < 1e-6 for n in _numbers(text))


# ---------------------------------------------------------------------------
# B1 — Add two numbers / Invariant: arithmetic result corresponds to `add`
# ---------------------------------------------------------------------------
ADD_CASES = [
    ("What is 2 plus 3?", 5),
    ("Add 17 and 25.", 42),
    ("Please sum 1234 and 4321.", 5555),
    ("What's 7.5 + 2.25?", 9.75),
    ("Add -8 and 3.", -5),
]


@pytest.mark.parametrize("request_text,expected", ADD_CASES)
def test_b1_addition(request_text, expected):
    output = agent(request_text)
    assert isinstance(output, str) and output.strip()
    assert _contains_number(output, expected), f"Sum {expected} not in: {output!r}"

    assert_test(
        LLMTestCase(
            input=request_text,
            actual_output=output,
            expected_output=f"A response stating the sum is {expected}.",
        ),
        [
            _metric(
                "B1 Addition Correctness",
                "The actual output answers the addition request with the sum given in "
                "the expected output, without presenting any conflicting numeric result, "
                "and without refusing or asking for clarification.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# B2 — Multiply two numbers / Invariant: arithmetic result corresponds to `multiply`
# ---------------------------------------------------------------------------
MULT_CASES = [
    ("What is 6 times 7?", 42),
    ("Multiply 12 by 13.", 156),
    ("What's 123 multiplied by 456?", 56088),
    ("Calculate 2.5 * 4.", 10),
    ("Multiply -3 and 9.", -27),
]


@pytest.mark.parametrize("request_text,expected", MULT_CASES)
def test_b2_multiplication(request_text, expected):
    output = agent(request_text)
    assert isinstance(output, str) and output.strip()
    assert _contains_number(output, expected), f"Product {expected} not in: {output!r}"

    assert_test(
        LLMTestCase(
            input=request_text,
            actual_output=output,
            expected_output=f"A response stating the product is {expected}.",
        ),
        [
            _metric(
                "B2 Multiplication Correctness",
                "The actual output answers the multiplication request with the product "
                "given in the expected output, without presenting any conflicting numeric "
                "result, and without refusing or asking for clarification.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# B3 / B4 — Weather query; location named in response is the queried one
# ---------------------------------------------------------------------------
WEATHER_CASES = [
    ("What's the weather in Paris?", "Paris"),
    ("How is the weather in Tokyo today?", "Tokyo"),
    ("Tell me the weather for New York.", "New York"),
]


@pytest.mark.parametrize("request_text,location", WEATHER_CASES)
def test_b3_b4_weather_query(request_text, location):
    output = agent(request_text)
    assert isinstance(output, str) and output.strip()

    assert_test(
        LLMTestCase(
            input=request_text,
            actual_output=output,
            expected_output=(
                f"Either weather information for {location}, or a statement that weather "
                f"data is unavailable for {location}."
            ),
        ),
        [
            _metric(
                "B3/B4 Weather Response",
                "The actual output addresses the weather for exactly the location in the "
                "input. Any location named in the output must be that same location (no "
                "other city/place substituted). The output either reports weather for "
                "that location or clearly states that weather data is unavailable for it. "
                "It must not refuse the request as invalid, ask for a location, or answer "
                "with an arithmetic result.",
            )
        ],
    )


# B4 + Guardrail: digit-only / nonexistent locations are valid and yield "unavailable"
UNAVAILABLE_CASES = [
    ("What's the weather in 12345?", "12345"),
    ("Weather for 90210 please.", "90210"),
    ("What is the weather in Zzyxqortvillebergh?", "Zzyxqortvillebergh"),
]


@pytest.mark.parametrize("request_text,location", UNAVAILABLE_CASES)
def test_b4_unavailable_weather(request_text, location):
    output = agent(request_text)
    assert isinstance(output, str) and output.strip()

    assert_test(
        LLMTestCase(
            input=request_text,
            actual_output=output,
            expected_output=f"Weather data is unavailable for {location}.",
        ),
        [
            _metric(
                "B4 Unavailable Weather Data",
                "The input is a valid weather query for a location with no weather data. "
                "The actual output must state that weather data is unavailable for that "
                "location (referring to it as given in the input). It must NOT invent "
                "weather details (temperature, conditions, etc.), must NOT use data from "
                "another location or source, must NOT treat the location as invalid input "
                "(e.g., by interpreting digits as numbers for arithmetic or asking for a "
                "different location instead of reporting unavailability).",
            )
        ],
    )


# ---------------------------------------------------------------------------
# B5 — Invalid or unsupported requests
# ---------------------------------------------------------------------------
INVALID_CASES = [
    ("", "empty request"),
    ("Can you help me?", "vague request"),
    ("Add 5.", "addition missing second number"),
    ("Multiply some numbers for me.", "multiplication missing numbers"),
    ("What's the weather like?", "weather request missing location"),
    ("Add 2 and 3 and tell me the weather in Paris.", "combines request types"),
    ("What is 10 divided by 2?", "unsupported operation: division"),
    ("What is 9 minus 4?", "unsupported operation: subtraction"),
    ("Add 1, 2 and 3.", "more than two numbers"),
    ("What's the square root of 16?", "unsupported operation"),
]


@pytest.mark.parametrize("request_text,reason", INVALID_CASES)
def test_b5_invalid_or_unsupported(request_text, reason):
    output = agent(request_text)
    assert isinstance(output, str) and output.strip()

    assert_test(
        LLMTestCase(
            input=request_text,
            actual_output=output,
            expected_output=(
                f"An explanation that the request cannot be completed ({reason}), asking "
                "for the required numbers or location when applicable, with no computed "
                "result or weather information."
            ),
        ),
        [
            _metric(
                "B5 Invalid Request Handling",
                "The actual output explains why the request cannot be completed, "
                "consistent with the reason in the expected output, and asks for the "
                "required numbers or location when applicable. It must NOT provide any "
                "computed arithmetic result, must NOT answer an unsupported operation, "
                "and must NOT provide any weather information.",
            )
        ],
    )


@pytest.mark.parametrize(
    "request_text,forbidden",
    [
        ("What is 10 divided by 2?", 5),
        ("What is 9 minus 4?", 5),
        ("Add 1, 2 and 3.", 6),
        ("What's the square root of 16?", 4),
    ],
)
def test_b5_no_result_for_unsupported(request_text, forbidden):
    output = agent(request_text)
    assert not _contains_number(output, forbidden), (
        f"Unsupported request answered with result {forbidden}: {output!r}"
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_entrypoint_returns_text():
    output = agent("What is 1 plus 1?")
    assert isinstance(output, str)
    assert output.strip()
