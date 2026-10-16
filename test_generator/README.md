# Trace Evaluation Generator

Gera testes baseados no trace do agente a partir de uma especificação.

**Uso:**

```bash
python whitebox_generator.py path/to/spec.md
```

O `test_trace.py` será gerado no mesmo diretório da especificação.

**Executar:** `pytest test_trace.py -vv`

# Black Box Evaluation Generator

Gera testes baseados nos comportamentos do agente a partir de uma especificação.

**Uso:**

```bash
python blackbox_generator.py path/to/spec.md
```

O `test_deepeval.py` será gerado no mesmo diretório da especificação.

**Executar:** `pytest test_deepeval.py -vv`
