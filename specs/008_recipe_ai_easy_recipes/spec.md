# Task

Você é um gerador de agentes de IA e abaixo está uma especificação de um agente. Gere um agente conforme a especificação. Considere que as ferramentas do agente devem ser geradas junto ao código do agente.

## Contexto

Um agente de IA é uma entidade que usa LLM e ferramentas para executar tarefas específicas de forma autônoma.

## Output

`agent.py`, `requirements.txt`, `tests.py`

## Contrato de interface com o benchmark

| Requisito de interface | Exemplo |
| --- | --- |
| Modelo | GPT-4o |
| Framework | LangGraph |
| GenAI Tool | ChatOpenAI |
| Entrypoint | `agent(...)` |
| Output | state: dict |

## Requisitos gerais

| Requisito | Significado | Especificação do agente |
| --- | --- | --- |
| Goal | Objetivo e comportamento principal do agente na voz ativa | O agente atua como um gerador de receitas personalizadas, adaptadas para utilizar os ingredientes colocados na query do usuário e para respeitar preferências e restrições dietéticas do usuário, quando houverem. |
| Persona (Opcional) | A personalidade que o agente assume ao realizar as tarefas | O agente assume a persona de um **chefe de cozinha**, cuja função é criar receitas personalizadas. |
| Inputs | Informação enviada para o agente | O agente recebe uma _query (em linguagem natural)_ do usuário, contendo preferências, ingredientes específicos e opcionalmente restrições dietéticas. |
| Policies/Invariants | Afirmações que descrevem o que é verdadeiro sobre o domínio ou sobre o estado do agente em um dado momento | Uma restrição dietética implica todos os seus ingredientes derivados, mesmo os não citados pelo usuário.  
  
Um ingrediente mencionado sob negação, ausência ou hipótese não é um ingrediente disponível.  
  
Validar se a entrada está relacionada a receitas e alimentos e não é vazia, caso contrário, retornar uma mensagem de erro com uma explicação clara do motivo. |
| Quality attributes (non-functional requirements) | Características observáveis do comportamento do agente durante a execução, que não descrevem o que ele faz nem o que ele nunca deve fazer, mas quão bem ele faz. | Nenhum. |
| Env | Chaves disponíveis através da configuração do ambiente | O agente deve acessar as variáveis de ambiente quando necessário.  
  
Chaves disponíveis: `OPENAI_API_KEY` |

## Comportamentos

| Comportamento | Trigger | Input | Saída | Efeito no ambiente | Constraints |
| --- | --- | --- | --- | --- | --- |
| **B1 — Extração de ingredientes** | Recebimento da query. | Query do usuário. | Lista de ingredientes identificados. Se nenhum ingrediente for identificado, `erro: nenhum ingrediente disponível`. | Nenhum. | A lista deve ser igual ao conjunto esperado, considerando variantes regionais, formas flexionadas, e excluindo ingredientes mencionados sob negação, ausência ou hipótese. O agente não deve fornecer/inventar respostas quando a query não for relacionada a receitas ou ingredientes, devendo interromper a geração. |
| **B2 — Geração da receita** | Recebimento dos ingredientes. | Query do usuário. | Receita com os campos:  
  
• **Título:** título da receita;  
• **Ingredientes:** lista de ingredientes com suas quantidades;  
• **Equipamentos:** lista de equipamentos necessários;  
• **Instruções:** instruções de preparo, passo a passo;  
• **Armazenamento:** recomendações de armazenamento. | Nenhum. | Todos os campos estão presentes e não vazios; nenhum ingrediente da receita viola as restrições dietéticas declaradas na query (se houverem); e **prioriza os ingredientes fornecidos**. |