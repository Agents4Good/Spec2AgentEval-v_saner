Você é um especialista em **testes de agentes**. Sua tarefa é criar testes determinísticos e casos de teste **LLM-as-a-Judge**, utilizando **GEval do DeepEval**, para avaliar o agente abaixo de acordo com sua especificação e seu trace completo.

**SPEC:** especificação e regras que o agente deve seguir.  
**FORMATO DO TRACE:** JSON contendo a **estrutura** do trace, para saber como extrair os dados necessários. Os testes gerados devem ser implementados para receber, em sua execução, o trace correspondente ao caso de uso avaliado. 

**Objetivo da avaliação**

A avaliação deve responder a duas perguntas:

1. **O agente fez o que foi pedido?**  
2. **Ele fez isso com qualidade?**

### **Cobertura**

Antes de gerar os testes, produza um inventário numerado de todos os elementos verificáveis da SPEC (REQ, CON, POL, TOOL) e dos casos de uso derivados (UC), incluindo variações implícitas.

Cada teste determinístico e cada caso de judge deve declarar quais IDs cobre. Ao final, inclua a matriz inversa ID → testes.

Um ID só conta como coberto se existir um teste que falharia caso aquele requisito fosse violado. Para cada um, descreva em uma linha o comportamento hipotético que causaria a falha. Testes que passariam independentemente do comportamento do agente não contam.

Cobertura é sobre a superfície da especificação, nunca sobre trajetórias. Não trate a ausência de um caminho esperado como lacuna.

Todo ID observável do inventário deve aparecer no covers de ao menos um cenário, e todo ID em `covers` deve ser verificado por ao menos um item de `testes` ou `judge` daquele cenário. Ao final, produza a matriz inversa ID → cenários e ID → testes, com as lacunas explicitadas.

Classifique cada ID como crítico ou padrão. Críticos (invariantes, ações irreversíveis, comportamentos proibidos) exigem no mínimo dois testes, sendo um adversarial.

Garanta que os casos de judge distribuam-se pelas quatro faixas de resultado (falha completa, parcial, correta com falhas, correta e de alta qualidade).

Liste explicitamente os IDs que não são verificáveis a partir do trace, com justificativa. Não gere testes vazios para preencher a matriz.

Separe o que é possível avaliar deterministicamente e o que precisa de LLM-as-a-Judge. Os testes devem apresentar cobertura máxima para todos os casos de uso do agente. 

Priorize testes determinísticos sempre que uma propriedade puder ser verificada de forma objetiva a partir do trace. Utilize LLM-as-a-Judge apenas para aspectos que exigem interpretação semântica ou avaliação de qualidade e que não possam ser determinados de maneira confiável por regras.  

***Não exija um caminho específico quando existirem diferentes formas válidas de resolver uma tarefa. Modelos de fronteira podem encontrar soluções criativas ou superiores, e a avaliação deve reconhecer isso.***

Para cada caso de uso/teste, o trace será recebido, assegure que as dimensões abaixo serão cobertas, quando aplicável:

### **Dimensões a avaliar**

* **Convergência de Caminho (Path Evaluation):** avalia se o agente atingiu o **goal e behavioral requirements** de forma eficiente, evitando loops desnecessários, passos redundantes. Um passo é redundante/desnecessário quando sua remoção não alteraria o resultado final nem o conhecimento disponível nos passos seguintes.   
* **Uso de Ferramentas (Tool Selection & Calling):** avalia se escolheu a ferramenta correta e se extraiu e formatou corretamente seus parâmetros de acordo com as **tools e behavioral requirements** especificados e, sem inventar informações.  
* **Seguimento de Instruções (Instruction Following):** avalia se respeitou requisitos **constraints, policies/ invariants e behavioral requirements**, inclusive diante de solicitações conflitantes, como inputs que tentem fazer o agente executar um comportamento proibido.  
* **Correção Funcional (Functional Correctness):** dimensão binária que verifica se o  **goal e o expected outcome** foram efetivamente alcançados.  
* **Eficiência Computacional:** analisa-se métricas de infraestrutura, como o **uso de tokens (quality attributes)**, a latência (tempo de resposta) e o custo geral por tarefa.

### **Cenários**

Gere \`scenarios.json\`: um array de cenários. Cada cenário representa UMA execução  
do agente e produz UM trace, sobre o qual rodam todos os testes e casos de judge  
listados nele. Não duplique cenários para exercitar testes diferentes sobre a  
mesma situação.

{  
  "id": "SC-nn",  
  "descricao": "uma linha sobre a situação",  
  "input": "mensagem exata do usuário ao agente",  
  "contexto": { ... estado inicial relevante, se houver ... },  
  "mocks": { "nome\_ferramenta": \<valor\> | \[\<valor1\>, \<valor2\>, ...\] },  
  "covers": \["REQ-02", "TOOL-01", "POL-03"\],  
  "testes": \["test\_chamou\_buscar\_pedido", "test\_sem\_chamada\_redundante"\],  
  "judge": \["JG-fidelidade", "JG-instruction\_following"\],  
  "criterio\_sucesso": "o que caracteriza o melhor tratamento possível, em termos  
                       de RESULTADO — nunca em termos de sequência de passos"  
}

\`testes\` contém nomes de funções definidas em deterministic\_tests.py.

\`judge\` contém ids de casos definidos em cases\_judge.json.  
Ambos devem existir nos arquivos gerados — nomes inventados quebram a execução.

\`mocks\` vazio significa execução com backend real. Mocke apenas quando o cenário  
exige um retorno inalcançável de forma confiável no real: erro, timeout, dado  
incompleto, resposta vazia, ou sequência de retornos distintos entre chamadas.  
Use lista quando o retorno deve variar por chamada, consumida por ordem de  
invocação. O mock deve responder coerentemente a argumentos inválidos, nunca  
devolver o valor fixado independentemente da chamada.

\`criterio\_sucesso\` descreve resultado, não trajetória. Formas diferentes de  
resolver a mesma tarefa podem ser igualmente válidas.

### **Testes determinísticos**

Gere **deterministic\_tests.py** com testes objetivos para tudo que puder ser verificado deterministicamente a partir do trace.

Cubra, quando aplicável, seleção e chamadas de ferramentas, parâmetros, ações obrigatórias, resultados, loops, chamadas redundantes, violações objetivas. Garanta que tudo que estiver no **behavioral requirements** foi coberto. 

Cada função recebe o trace e retorna ("PASS"|"FAIL", justificativa). Em caso de `FAIL`, a justificativa deve citar o valor observado no trace que motivou a reprovação, não apenas repetir o nome do teste. Em `PASS`, retorne `None`.

### **Casos LLM-as-a-Judge**

Gere **cases\_judge.json** para uso com GEval do DeepEval.

Os casos devem avaliar aspectos que exigem julgamento semântico e representar diferentes níveis de sucesso, permitindo **crédito parcial**. Exemplo: um agente de suporte que identifica corretamente o problema e verifica o cliente, mas não consegue processar um reembolso, é significativamente melhor do que um que falha imediatamente. É importante representar esse espectro de sucesso nos resultados.

Diferencie falha completa, resolução parcial, resolução correta com pequenas falhas e resolução correta e de alta qualidade.

Cada caso em cases\_judge.json contém id, name, evaluation\_steps (lista de verificações objetivas em linguagem natural), evaluation\_params, threshold e covers.

Especifique explicitamente como o trace é mapeado para o LLMTestCase: quais campos vão para input, actual\_output e retrieval\_context. As rubricas devem se ancorar no próprio trace, comparar a resposta final contra os retornos de ferramenta nele registrados e não contra valores esperados externos.

### **Dificuldade**

Se os testes não forem difíceis o suficiente, o agente rapidamente atinge 100% de sucesso. Quando isso ocorre, o eval serve apenas para evitar regressões, mas perde totalmente a capacidade de medir novas melhorias ou o progresso da ferramenta. Por isso, os testes devem ser suficientemente difíceis para continuar sendo úteis quando o agente atingir alto desempenho. Evite casos triviais que levam rapidamente a 100% de sucesso.

Inclua, quando aplicável, situações com:

* ambiguidades;  
* informações incompletas;  
* instruções conflitantes;  
* tentativas de jailbreak;  
* múltiplas ferramentas possíveis;  
* parâmetros incorretos ou ausentes;  
* caminhos alternativos;  
* falhas intermediárias;  
* situações em que o agente deve reconhecer que não possui informação suficiente.

O objetivo é criar uma avaliação **abrangente, justa e discriminativa**, capaz tanto de detectar regressões quanto de medir melhorias na qualidade do agente.

Gere exatamente os três arquivos:

1. **deterministic\_tests.py**  
2. **cases\_judge.json**  
3. **scenarios.json**

**Importante:** o **deterministic\_tests.py** deve conter testes/funções avaliadoras que recebam o trace como entrada. O **cases\_judge.json** deve definir avaliações que também possam ser aplicadas ao trace correspondente recebido em cada execução. 

