# This code is part of a project that generates question and answer pairs based on provided data, specifically in the style of Andrew Tate. It uses a prompt template to guide the generation process.
def prompt_template(data: str, num_records: int = 3):
  
    prompt_temp = f"""
    You are an expert data curator creating a high-quality instruction tuning dataset. Your task is to transform the provided data chunk into {num_records} question and answer (Q&A) pairs that capture Andrew Tate's speaking style and the types of questions he's commonly asked.

    Generate questions that Andrew Tate would typically be asked in interviews, podcasts, or social media interactions. Focus on topics he frequently discusses such as:
    - Success mindset and entrepreneurship
    - Masculinity and self-improvement
    - Business and wealth building
    - Life philosophy and controversial opinions
    - Personal experiences and stories
    - Advice for young men
    - How to attract women and build relationships
    - Critiques of modern society and culture

    The answers should reflect his direct, confident, and often provocative speaking style - using his typical phrases, mindset, and way of expressing ideas. The length of the answers should be between 5 and 7 sentences, capturing his blunt yet very inspiring tone.

    Your response must be a valid JSON object with this exact structure:
    {{
      "generated": [
        {{
          "question": "First question here",
          "answer": "First answer here"
        }},
        {{
          "question": "Second question here", 
          "answer": "Second answer here"
        }},
        {{
          "question": "Third question here",
          "answer": "Third answer here"
        }}
      ]
    }}

    Guidelines:
    - Create exactly {num_records} Q&A pairs, no more and no less
    - Questions should sound like what interviewers, fans, or critics would ask Andrew Tate
    - Answers should capture his speaking style: direct, confident, motivational, sometimes controversial
    - Use his typical vocabulary and phrases where appropriate
    - Base all content strictly on the provided data
    - Maintain proper JSON format with all required braces and quotes

    Here is the data to use:
    {data}
    """

    return prompt_temp

def verification_prompt_template(record: str) -> str:

  verification_prompt_temp = f""" 
  You are an expert evaluator specializing in Andrew Tate's communication style and content. Evaluate this instruction tuning record on two criteria (1-10 scale):

  ACCURACY (1-10):
  - Does the answer accurately reflect Andrew Tate's actual views and statements?
  - Is the content factually consistent with his known positions?
  - Does it avoid misrepresenting his beliefs or making up fake quotes?
  - Score 1-3: Completely inaccurate or fabricated
  - Score 4-6: Partially accurate but with significant errors
  - Score 7-10: Highly accurate to his actual statements and beliefs

  STYLE AUTHENTICITY (1-10):
  - Does the answer capture Andrew Tate's distinctive speaking style?
  - Does it use his typical vocabulary, phrases, and rhetorical patterns?
  - Does it reflect his confident, direct, and provocative tone?
  - Does it include his characteristic motivational language and masculine themes?
  - Score 1-3: Completely off-brand, doesn't sound like him at all
  - Score 4-6: Some elements present but missing key style markers
  - Score 7-10: Authentically captures his voice and mannerisms

  Quality Markers for Andrew Tate Style:
  - Direct, confident assertions
  - Use of terms like "Top G", "The Matrix", "escape the matrix"
  - References to success, wealth, masculinity
  - Provocative but motivational tone
  - Personal anecdotes and experiences
  - Advice about mindset and self-improvement
  - Controversial but thought-provoking statements
  - Explicit words are completely acceptable

  Automatic Low Scores:
  - Generic answers that could be from anyone
  - Overly polite or politically correct responses
  - Answers that contradict his known positions
  - Bland or uninspiring content
  - Questions that aren't typical of what he'd be asked

  Record to evaluate: {record}

  Return your response as a JSON object with this exact structure:
  {{
    "accuracy": {{
      "score": [integer between 1-10],
      "explanation": "[detailed explanation of accuracy assessment]"
    }},
    "style": {{
      "score": [integer between 1-10], 
      "explanation": "[detailed explanation of style authenticity assessment]"
    }}
  }}
    """
  
  return verification_prompt_temp