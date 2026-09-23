from agent import agent

def test_invalid_query():
    s = agent("olá, conte uma piada")
    assert s["message"].startswith("Error:") and s["recipe"] is None

def test_positive_extraction():
    s = agent("quero uma receita com tomate e mandioca, sem peixe")
    assert s["ingredients"] == ["tomate", "mandioca"]

def test_state_shape():
    s = agent("receita de tomate")
    assert set(("query", "valid", "ingredients", "recipe", "message")) <= set(s)
