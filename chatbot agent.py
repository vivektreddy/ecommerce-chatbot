import re
import logging
from langchain.tools import Tool
from pydantic.v1 import BaseModel, PrivateAttr
from typing import List, Dict, Tuple
from langchain.chat_models import ChatOpenAI
from tools import *
from constants import *
from helpers import *
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

class Agent(BaseModel):

    llm: object
    tools: list[Tool]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 0
    _logger: logging.Logger = PrivateAttr(default=logging.getLogger(__name__))


    # The stop pattern is used, so the LLM does not hallucinate until the end
    stop_pattern: List[str] = [f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}']

    class Config:
        arbitrary_types_allowed = True  # Allow `Tool` objects in Pydantic

    @property
    def tool_description(self) -> str:
        return "\n".join([f"<tool_name>{tool.name}</tool_name>\n<tool_description>{tool.description}</tool_description>" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, Tool]:
        return {tool.name: tool for tool in self.tools}

    def run(self, question, inputs):
        
        num_loops,self.max_loops = 0, 4
        
        previous_responses = inputs[:] if inputs else []
        
        while num_loops < self.max_loops:
            num_loops += 1

            # ensures the curly braces  are interpreted as literal strings
            sanitized_previous_responses = [response.replace("{", "{{").replace("}", "}}") for response in previous_responses]

            prompt = self.prompt_template.format(
                tool_description=self.tool_description,
                tool_names=self.tool_names,
                input=question,
                previous_responses='\n'.join(sanitized_previous_responses)
            )
            
            print('deciding next action')                
            try:
                generated, tool, tool_input = self.decide_next_action(prompt)
            
                print("decided next action")
                print('generated output from decide next action', generated)
                print('--------------------------------')


                if "Final Answer" in generated or tool == 'Final Answer':
                    print('final answer was generated.  tool is ', tool, 'output is: ', generated)
                    return generated.split("Final Answer:")[-1].strip()


                if tool:
                    tool_instance = self.tool_by_names[tool]
                    
                    # Handle different tool input scenarios
                    if tool == 'product_knowledge_base':
                        tool_input = question
                    elif not tool_input or tool_input == "{}":
                        tool_input = {}
                    else:
                        print('tool is ', tool)
                        tool_input = parse_tool_input(tool_input)
                        print('tool input is ', tool_input)
                        if not verify_tool_input(tool, tool_input):
                            raise ValueError(f"Invalid tool input {tool_input} for tool {tool}")
                    
                    # Process tool
                    tool_result = process_tool_input(tool_instance, tool, tool_input)
                    
                    print('tool result is ', tool_result)
                    generated += f"\n Got Response From Tool {tool}. The result is {tool_result}\n"
                    previous_responses.append(generated)
                else:
                    raise ValueError("No valid tool or final answer was generated")
                    exit()
            except Exception as e:
                self._logger.error(f"Error in run loop: {str(e)}")
                return f"An error occurred: {str(e)}"
            
        return tool_result

    def decide_next_action(self, prompt: str) -> str:

        generated = self.llm.invoke(prompt, stop=self.stop_pattern).content
        try:
            tool, tool_input = self._parse(generated)
            return generated, tool, tool_input
        except Exception as e:
            self._logger.error(f"ERROR OCCURED while parsing the generated output: {e}")
            print(e)
            return generated, None, None

    def _parse(self, generated: str) -> Tuple[str, str]:

        """
        This function parses the generated output to extract the tool name and tool input.
        It first checks if the output contains 'Final Answer' and returns it directly.
        If not, it uses a regex pattern to capture the tool name and tool input.
        """
        # Check if 'Final Answer' exists in the output and return it directly
        if FINAL_ANSWER_TOKEN in generated:
            actual_final_answer = generated.split(FINAL_ANSWER_TOKEN)[-1].strip()
            #tool and tool input return are final answer add some kind of check for if tool is final answer
            return 'Final Answer', actual_final_answer
        
        # otherwise, Use regex to capture tool name and tool input
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            print("no match for tool and tool input in prompt found in parse function")
            return None, None
        
        #tool name is the first group, tool input is the second group
        tool = match.group(1).strip()
        tool_input = match.group(2).strip().strip('"') 
        return tool, tool_input


#Define the tools
tools = [
    Tool(name="get_all_data", func=get_all_data_tool, description="Retrieve all records in the dataset."),
    Tool(name="get_customer_data", func=get_customer_data_tool, description="Retrieve records by Customer ID."),
    Tool(name="get_product_category_data", func=get_product_category_data_tool, description="Retrieve records by Product Category."),
    Tool(name="get_orders_by_priority", func=get_orders_by_priority_tool, description="Retrieve orders by priority."),
    Tool(name="total_sales_by_category", func=total_sales_by_category_tool, description="Get total sales by Product Category."),
    Tool(name="high_profit_products", func=high_profit_products_tool, description="Get high-profit products."),
    Tool(name="shipping_cost_summary", func=shipping_cost_summary_tool, description="Get shipping cost summary."),
    Tool(name="profit_by_gender", func=profit_by_gender_tool, description="Get profit summary by customer gender."),
    Tool(name="product_knowledge_base", func=product_knowledge_base_tool, description="Retrieve product information based on user query.")
]

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o")
agent = Agent(llm=llm,tools=tools)

print("Agent is ready. Type 'exit' to quit.")
inputs = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Agent: Goodbye!")
        break
    agent.run(question=user_input, inputs=inputs)

# Gradio Interface
def chatbot_interface(user_input):
    inputs = []
    response = agent.run(question=user_input, inputs=inputs)
    return response

