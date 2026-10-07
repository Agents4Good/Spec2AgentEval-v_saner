# **Etapas**

- [ ] Especificar o agente de acordo com o modelo de especificação
- [ ] Preencher cada campo de acordo com o significado
- [ ] A saída deve estar no formato Markdown
- [ ] Manter o status na planilha agentes-benchmark-v2

## **Revisar a especificação seguindo os critérios: **

**Faça a revisão em duas etapas:**

1. Leia cada campo sozinho. Nos trechos que descrevem ações, verifique se a frase identifica o responsável, apresenta uma ação principal e pode ser entendida sem adivinhação.
2. Leia a especificação como um conjunto. Verifique se os nomes são consistentes e se os diferentes campos não se contradizem.
Busque pontos de ambiguidade. Se existe um ponto em que você fica com dúvida, provavelmente existe ambiguidade que deve ser resolvida.

**Checklist:**

- [ ] Antes de concluir, verifique:
- [ ] Nos campos que descrevem ações, está claro quem realiza cada ação?
- [ ] Cada declaração apresenta uma ação principal ou separa ações independentes?
- [ ] O mesmo nome é usado para o mesmo conceito em todos os campos?
- [ ] Ainda existem termos vagos ou não explicados?
- [ ] Comparações indicam uma referência?
- [ ] Pronomes têm um único significado possível?
- [ ] As relações entre condições estão claras?
- [ ] Palavras como “sempre” e “nunca” têm escopo definido?
- [ ] Documentos, políticas e ferramentas podem ser localizados?
- [ ] Quando uma restrição registrada em Constraints bloqueia uma ação, a resposta do agente está descrita em Behavioral Requirements?
- [ ] Ferramentas, dados, permissões e outras dependências estão declarados?
- [ ] Avaliar o agente com um revisor humano

> Para cada campo, dê um nota na escala Likert (1: muito ruim, 2: ruim, 3: moderado, 4: bom, 5: excelente) de acordo com os critérios de corretude

> Calcular a média da nota (a gente calcula?)

2. Avaliar o agente com um revisor LLM-as-a-judge

> Modelo: Gemini 3.1 Pro (disponível no chat)

**Prompt-rodada_1:** Você é um avaliador de especificações de agentes de IA. Avalie exclusivamente as informações explicitamente presentes na especificação. Não infira requisitos ausentes e não considere detalhes de implementação.
Contexto: A especificação especifica as funções e capacidades que um sistema de software deve oferecer, suas características e as restrições que deve respeitar. Ela deve descrever, com o nível de detalhamento necessário, o comportamento do sistema sob diversas condições, bem como atributos de qualidade desejados, como desempenho, segurança e usabilidade. A especificação serve de base para o planejamento, o projeto e a codificação subsequentes do projeto, além de fundamentar os testes do sistema e a documentação para o usuário. No entanto, não deve conter detalhes de projeto, construção, testes ou gestão de projetos, exceto as restrições conhecidas de projeto e implementação.

> Saída: Para cada campo, dê um nota na escala Likert (1: muito ruim, 2: ruim, 3: moderado, 4: bom, 5: excelente) e a justificativa de acordo com os critérios de corretude. Retorne em Markdown.

Calcular a média da nota

**Prompt-rodada_2:** Revise a avaliação anterior da especificação e identifique apenas erros nas notas ou justificativas, considerando os critérios de corretude e somente as informações explicitamente presentes na especificação. Não reavalie a especificação do zero nem infira informações ausentes. Para cada campo, indique se há problema e, quando houver, proponha a correção da nota e/ou justificativa.

>Saída: Retorne em Markdown: Campo | Problema? | Correção sugerida | Justificativa.

Concordância entre as notas:

Todas as notas devem alcançar no mínimo 4
Média no mínimo 4

Como calcular a concordância?

Iterar entre revisão-especificação, colocar outro prompt pro llm-as-a-judge na segunda revisão

**Critério de parada:**
Quando há um certo nível de acordo e a média é acima de 4

Adicionar em: https://github.com/Agents4Good/Spec2AgentEval-v_saner
