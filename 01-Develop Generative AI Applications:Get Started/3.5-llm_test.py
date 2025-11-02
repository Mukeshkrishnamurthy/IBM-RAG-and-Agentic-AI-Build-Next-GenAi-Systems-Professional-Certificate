from model import llama3_response, granite_response, mixtral_response

def call_all_models(system_prompt, user_prompt):
    llama_result = llama3_response(system_prompt, user_prompt)
    granite_result = granite_response(system_prompt, user_prompt)
    mixtral_result = mixtral_response(system_prompt, user_prompt)

    print("Llama3 Response:\n", llama_result.content)
    print("\nGranite Response:\n", granite_result.content)
    print("\nMixtral Response:\n", mixtral_result.content)

# Example call to test all models
call_all_models("You are a helpful assistant who provides concise and accurate answers", "What is the capital of Canada? Tell me a cool fact about it as well")

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user\'s message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt, 'format_prompt':json_parser.get_format_instructions()})
    