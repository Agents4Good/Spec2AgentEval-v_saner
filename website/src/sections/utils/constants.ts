export interface AgentInfo {
    name: string;
    description: string;
  }
  
export const AGENT_DESCRIPTIONS: Record<string, string> = {
"001_list_issues_agent": "Agent capable of listing and filtering GitHub issues from a specified public repository using natural language input.",
"002_travel_planner": "Agent capable of planning a travel based on user destination city and interests using natural language as input.",
"003_classify_query": "Agent capable of evaluating Brazilian National High School Exam (ENEM) essays following INEP's official assessment rubric.",
"004_basic_agent": "Agent capable of answering user queries with accurate, concise, and factual information, strictly avoiding hallucinations.",
"005_simple_email_assistant": "Agent capable of triaging, classifying, and processing incoming emails. For emails requiring a response, it must generate a professional draft reply using solely the context provided in the email text.",
"006_essay_evaluator": "Agent capable of handling customer support queries based on natural language input.",
"007_personal_trainer": "Agent capable of creating personalized fitness routines based on a user's profile, goals, and physical constraints, following evidence-based training principles.",
"008_recipe_ai_easy_recipes": "...",
"011_list_prs_or_issues": "Agent capable of listing and filtering either GitHub Issues or Pull Requests from a specified public repository using natural language input.",
"012_explore_travel_destination": "Agent capable of answering travel destination questions about wather or news using natural language input.",
"013_essay_evaluator_or_search": "Agent capable of assisting users with essay writing using natural language input. The agent must reason about the user's intent to decide whether to fetch a document from a URL or search for information about a topic — and must be evaluable through automated functional execution.",
"014_academic_triage_agent": "Agent capable of triaging a user's academic request, selecting exactly one appropriate tool based on the subject domain (Exact Sciences/Tech, Life Sciences/Health, or General Concepts), and returning a formatted summary of the findings.",
"015_medium_email_assistant": "Agent capable of triaging, classifying, and processing incoming emails. The agent must act as an executive assistant, deciding whether to ignore, notify, or respond to an email.",
"016_support_medium": "...",
"021_classify_prs_by_review_status": "Agent capable of retrieving open pull requests from a repository, inspecting their review status via tool calls and classifying each pull request into: `needs_review` or `reviewed`.",
"022_explore_travel_viability": "Agent that analyzes whether a travel destination is viable by combining real-time weather data and recent news.",
"023_essay_writer_assistant": "Agent capable of writing a complete ENEM-style essay from a topic provided in a Google Doc, saving the result to a local file.",
"024_autonomous_research_agent": "Agent capable of performing multi-step investigative research. The agent must discover a hidden variable using a general knowledge tool, use that variable to query a specialized academic database, and return the final synthesis strictly as a parsable JSON string.",
"025_email_assistant": "Agent capable of triaging, classifying, and processing incoming emails. The agent must act as an executive assistant, deciding whether to ignore, notify, or respond to an email. For emails requiring a response, it must autonomously use the official Notion API to check availability in a Calendar Database and save a generated email draft directly into a Drafts Database."
};

export const getAgentDescription = (agentName: string): string => {
return AGENT_DESCRIPTIONS[agentName] || "Descrição não disponível para este agente.";
};