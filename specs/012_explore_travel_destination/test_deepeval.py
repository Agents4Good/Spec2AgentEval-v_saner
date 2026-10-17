"""Black-box behavioral tests (DeepEval 3.3.9) for the travel destination agent.

Derived exclusively from spec.md. Only the observable contract is used:
    agent(user_request: str) -> {"status", "user_request", "response"}
"""

import time

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agent import agent

JUDGE_MODEL = "gpt-4o"
TIME_LIMIT_S = 30
TIME_SLACK_S = 5


# ---------- helpers ----------

def run(request: str):
    start = time.monotonic()
    result = agent(request)
    return result, time.monotonic() - start


def assert_schema(result, request):
    """B7 / invariant: always return the required output schema."""
    assert isinstance(result, dict)
    assert set(result.keys()) == {"status", "user_request", "response"}
    assert result["status"] in ("success", "failure")
    assert result["user_request"] == request
    assert isinstance(result["response"], str)


def make_case(request, result):
    return LLMTestCase(input=request, actual_output=result["response"])


# ---------- metrics ----------

goal_relevance = GEval(
    name="Travel Destination Relevance",
    criteria=(
        "Determine whether the actual output, acting as a travel destination "
        "assistant, directly answers the user's question about the travel "
        "destination in natural language."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.6,
)

weather_grounded = GEval(
    name="Weather Answer Grounded In Real Data",
    criteria=(
        "The user asked about current weather. The actual output must either "
        "report specific current weather conditions for the requested city "
        "(e.g. temperature, conditions) as retrieved data, or clearly state "
        "that weather data could not be retrieved. It must NOT present vague, "
        "generic, seasonal or invented weather as if it were current data."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.6,
)

news_grounded = GEval(
    name="News Answer Grounded In Search Results",
    criteria=(
        "The user asked about news, safety, travel advisories or current "
        "events for a destination. The actual output must either summarize "
        "recent search results about that destination, or clearly state that "
        "results could not be retrieved. It must NOT fabricate news items or "
        "advisories, nor answer purely with generic, timeless knowledge "
        "presented as current events."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.6,
)

direct_tips = GEval(
    name="Direct Travel Tips Answer",
    criteria=(
        "The user asked about itineraries, activities, local culture, food or "
        "general travel tips. The actual output must provide a helpful, "
        "natural-language answer addressing that request for the destination."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.6,
)

failure_message = GEval(
    name="Failure Explained",
    criteria=(
        "The request was empty or blank. The actual output must be a "
        "natural-language message indicating the request could not be "
        "processed (e.g. that it is empty or a question is needed), without "
        "inventing an answer about any destination."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=JUDGE_MODEL,
    threshold=0.5,
)


# ---------- B5: empty / blank requests ----------

@pytest.mark.parametrize("request_text", ["", " ", "   ", "\n", "\t \n "])
def test_b5_empty_request_returns_failure(request_text):
    result, _ = run(request_text)
    assert_schema(result, request_text)  # preserves original input
    assert result["status"] == "failure"


def test_b5_empty_request_response_is_explanatory():
    result, _ = run("   ")
    assert_schema(result, "   ")
    assert result["status"] == "failure"
    assert_test(make_case("   ", result), [failure_message])


# ---------- B2: weather ----------

WEATHER_REQUESTS = [
    "What's the weather like in Paris right now?",
    "What is the current temperature in Tokyo?",
    "How are the weather conditions in Lisbon today?",
]


@pytest.mark.parametrize("request_text", WEATHER_REQUESTS)
def test_b2_weather_request(request_text):
    result, elapsed = run(request_text)
    assert_schema(result, request_text)
    assert elapsed <= TIME_LIMIT_S + TIME_SLACK_S
    if result["status"] == "success":
        assert result["response"].strip()
        assert_test(make_case(request_text, result),
                    [weather_grounded, goal_relevance])
    else:
        assert_test(make_case(request_text, result), [weather_grounded])


def test_b2_invalid_city_does_not_fabricate_weather():
    request = "What's the current weather in Xyzzyqwertyville123?"
    result, _ = run(request)
    assert_schema(result, request)
    assert_test(make_case(request, result), [weather_grounded])


# ---------- B3: news / safety / advisories ----------

NEWS_REQUESTS = [
    "Are there any current travel advisories for Bangkok?",
    "Is it safe to travel to Rio de Janeiro right now?",
    "What are the latest news and events happening in Barcelona?",
]


@pytest.mark.parametrize("request_text", NEWS_REQUESTS)
def test_b3_news_request(request_text):
    result, elapsed = run(request_text)
    assert_schema(result, request_text)
    assert elapsed <= TIME_LIMIT_S + TIME_SLACK_S
    if result["status"] == "success":
        assert result["response"].strip()
        assert_test(make_case(request_text, result),
                    [news_grounded, goal_relevance])
    else:
        assert_test(make_case(request_text, result), [news_grounded])


# ---------- B4: direct answers ----------

TIPS_REQUESTS = [
    "Suggest a 3-day itinerary for Rome.",
    "What traditional food should I try in Mexico City?",
    "What are some cultural etiquette tips for visiting Japan?",
    "What activities can I do in Cape Town?",
]


@pytest.mark.parametrize("request_text", TIPS_REQUESTS)
def test_b4_direct_answer(request_text):
    result, elapsed = run(request_text)
    assert_schema(result, request_text)
    assert elapsed <= TIME_LIMIT_S + TIME_SLACK_S
    assert result["status"] == "success"
    assert result["response"].strip()
    assert_test(make_case(request_text, result), [direct_tips, goal_relevance])


# ---------- B6 / B7: robustness and schema invariants ----------

@pytest.mark.parametrize(
    "request_text",
    [
        "a" * 10000,
        "🌍✈️🏖️???",
        "'; DROP TABLE users; --",
        "Weather in Paris and news about Berlin and food in Rome?",
    ],
)
def test_b6_b7_no_exception_and_schema(request_text):
    try:
        result, elapsed = run(request_text)
    except Exception as exc:  # B6: exceptions must not propagate
        pytest.fail(f"Agent propagated an exception: {exc!r}")
    assert_schema(result, request_text)
    assert elapsed <= TIME_LIMIT_S + TIME_SLACK_S  # B6: 30-second limit


def test_b7_user_request_preserved_verbatim():
    request = "  What's the weather in Madrid?  "
    result, _ = run(request)
    assert_schema(result, request)
