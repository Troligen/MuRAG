# Scientific paper passage prompt
SCIENTIFIC_PAPER_PROMPT = """
Please write a scientific paper passage to answer the question
Question: {question}
Passage:
"""

# Literature review prompt
LITERATURE_REVIEW_PROMPT = """
Provide a comprehensive literature review on the following topic:
Topic: {topic}
Literature Review:
"""

# Methodology description prompt
METHODOLOGY_PROMPT = """
Describe the methodology for the following experiment:
Experiment: {experiment}
Methodology:
"""

# Results analysis prompt
RESULTS_ANALYSIS_PROMPT = """
Analyze the following experimental results:
Results: {results}
Analysis:
"""

# Discussion section prompt
DISCUSSION_PROMPT = """
Write a discussion section addressing the following research question:
Research Question: {research_question}
Discussion:
"""

# Abstract writing prompt
ABSTRACT_PROMPT = """
Compose an abstract for a scientific paper on the following subject:
Subject: {subject}
Abstract:
"""

# Conclusion writing prompt
CONCLUSION_PROMPT = """
Write a conclusion for a scientific paper addressing the following hypothesis:
Hypothesis: {hypothesis}
Conclusion:
"""

# Dictionary of prompts
SCIENTIFIC_PROMPTS = {
    "paper_passage": SCIENTIFIC_PAPER_PROMPT,
    "literature_review": LITERATURE_REVIEW_PROMPT,
    "methodology": METHODOLOGY_PROMPT,
    "results_analysis": RESULTS_ANALYSIS_PROMPT,
    "discussion": DISCUSSION_PROMPT,
    "abstract": ABSTRACT_PROMPT,
    "conclusion": CONCLUSION_PROMPT,
}


# Function to get a specific prompt
def get_prompt(prompt_name):
    return SCIENTIFIC_PROMPTS.get(prompt_name, "Prompt not found")


# Function to list all available prompts
def list_prompts():
    return list(SCIENTIFIC_PROMPTS.keys())
