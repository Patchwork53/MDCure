system_prompt = "You are an AI language model tasked with generating instructions for various tasks."

instruct_prompt = """Context: {context}
Task: Based on the provided context, generate a pair consisting of an instruction and its corresponding answer where the instruction guides the user towards the answer.

Output format:
Instruction: [Your instruction here]
Answer: [Your answer here]
"""

gpt_preference_gen = """Instruction Quality Rating Task
Rate the quality of the generated instruction based on the provided documents, using a scale of 1-5. Only provide numerical scores, without any rationale or explanation.

Relevance: Does the instruction align well with the content of the documents? Does it make sense given the provided information?
Coherence & Factuality: Is the instruction-answer pair coherent, logical, and factually accurate? Does the answer appropriately address the instruction and is it well-supported by the documents?
Creativity: How diverse or creative is the instruction in terms of question type (e.g., factual, inferential) and format (e.g., multiple-choice, open-ended)?
Context Integration: How well does the instruction leverage and synthesize information from multiple documents to form a comprehensive response?
Inter-Document Relationships: Does the instruction encourage understanding relationships (e.g., comparisons, contrasts, discrepancies) between different documents?
Complexity: Does the instruction appropriately challenge the answerer to think critically and synthesize information from multiple sources?

Input:
Context: {context}
{instruction}

Output (provide only numerical scores, no rationale): 
Relevance: [score]
Coherence & Factuality: [score]
Creativity: [score]
Context Integration: [score]
Inter-Document Relationships: [score]
Complexity: [score]
"""

relevance_appropriateness_rubric = """[Is the instruction relevant to the context?]
Score 1: Instruction is completely irrelevant or inappropriate to the context, leading to a task that is unrelated or incorrect.
Score 2: Instruction is somewhat relevant but contains significant inappropriate elements or misguides the task.
Score 3: Instruction is moderately relevant but could be better aligned with the context; some inappropriate elements are present.
Score 4: Instruction is relevant and mostly appropriate, with minor misalignments or off-topic elements.
Score 5: Instruction is highly relevant and fully appropriate, perfectly aligning with the context.
"""

clarity_specificity_rubric = """[Does the instruction exhibit a high degree of clarity?]
Score 1: Instruction is extremely unclear or ambiguous, with poor specificity.
Score 2: Instruction is somewhat unclear or vague, lacking sufficient specificity.
Score 3: Instruction is moderately clear but could benefit from improved specificity.
Score 4: Instruction is clear and specific, with minor issues.
Score 5: Instruction is very clear, precise, and highly specific, with no ambiguity.
"""

comprehensiveness_rubric = """[Does the instruction cover all necessary details?]
Score 1: Instruction is severely lacking in coverage, omitting critical details necessary for the task.
Score 2: Instruction is missing several important details, resulting in an incomplete or flawed task.
Score 3: Instruction covers the essential aspects but lacks some important details, leading to a task that is somewhat incomplete.
Score 4: Instruction is mostly comprehensive, covering nearly all necessary details with only minor omissions.
Score 5: Instruction is fully comprehensive, covering all necessary details without any omissions, ensuring a complete and accurate task.
"""

conciseness_rubric = """[Is the instruction concise and focused?]
Score 1: Instruction is excessively verbose, with a lot of unnecessary information that detracts from the clarity and focus.
Score 2: Instruction is somewhat verbose, including some unnecessary details that could be omitted for a more concise task.
Score 3: Instruction is moderately concise, but could be more succinct without losing meaning.
Score 4: Instruction is concise, with minor verbosity that does not significantly impact clarity or focus.
Score 5: Instruction is highly concise, with no unnecessary information, and fully focused on delivering the task in a succinct manner.
"""

multi_document_rubric = """[Does the instruction require consideration of information from multiple documents?]
Score 1: Instruction completely fails to address the multi-document context, leading to a single-document task.
Score 2: Instruction poorly addresses the multi-document context, with minimal consideration of how to handle information across multiple sources.
Score 3: Instruction moderately considers the multi-document context, but only weakly requires synthesizing or differentiating information across sources.
Score 4: Instruction effectively addresses the multi-document context, with clear or implied focus on handling multiple sources, though minor improvements are possible.
Score 5: Instruction fully and effectively addresses the multi-document context, providing clear, precise focus on synthesizing, comparing, differentiating, or reasoning over information from multiple sources.
"""