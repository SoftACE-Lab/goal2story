from colorama import Fore, Style
import json
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

description_prompt = "You are an agile requirement expert and good at analyzing goal, actor, impact, deliverable and user story."


client = OpenAI(
    # Read your API key from environment variables
    api_key="your_api_key", 
    base_url="your_base_url",
)

model = "model_name"


def read_csv_data(file_path):
    """Read CSV file and convert to required data structure"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Define required column names
        column_names = [
            "goal", "actor", "impact", "deliverable", 
            "action", "expected_outcome", "background", "problems", "solutions"
        ]
        
        data_rows = []
        
        for _, row in df.iterrows():
            mapping_json = json.loads(row.get("mapping_result", "{}"))
            description_json = json.loads(row.get("project_info", "{}"))
            
            row_data = [
                mapping_json.get('Goal', ''),
                mapping_json.get('Actor', ''),
                mapping_json.get('Impact', ''),
                mapping_json.get('Deliverable', ''),
                mapping_json.get('User Story', {}).get('action', ''),
                mapping_json.get('User Story', {}).get('expected_outcome', ''),
                description_json.get('background', ''),
                description_json.get('problems', ''),
                description_json.get('solutions', '')
            ]
            data_rows.append(row_data)
        
        return pd.DataFrame(data_rows, columns=column_names)
    except Exception as e:
        print(f"{Fore.RED}Error reading CSV file: {str(e)}{Style.RESET_ALL}")
        raise

def read_prompt(file_name):
    """Read prompt file content"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"{Fore.RED}Error reading prompt file: {str(e)}{Style.RESET_ALL}")
        raise

def build_prompt(template, **kwargs):
    """Build prompt by replacing variables
    
    Args:
        template (str): prompt template
        **kwargs: dictionary of variables to replace
    
    Returns:
        str: processed prompt
    """
    try:
        # Preprocess all JSON type values
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                # JSON objects need to be converted to string first, then processed with repr
                processed_kwargs[key] = repr(json.dumps(value, indent=2))[1:-1]
            else:
                processed_kwargs[key] = value
                
        # Use replace method to substitute all variables
        result = template
        for key, value in processed_kwargs.items():
            placeholder = '{' + key + '}'
            result = result.replace(placeholder, str(value))
            
        return result
    except Exception as e:
        print(f"{Fore.RED}Error building prompt: {str(e)}{Style.RESET_ALL}")
        raise
    

# Use formatting agent to fix JSON
def fix_json_with_agent(json_content):
    """Use formatting agent to fix JSON format"""
    try:
        
        format_prompt = f"""Please repair the following JSON format, only return the repaired valid JSON content, do not add any explanation or other text, do not use code block tags:
        
        {json_content}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,  # Low temperature for more deterministic output
            messages=[
                {"role": "system", "content": "You are a professional JSON format repair expert. Only return the repaired valid JSON content, do not add any explanation or other text. Do not use code block tags."},
                {"role": "user", "content": format_prompt}
            ]
        )
        
        formatted_json = response.choices[0].message.content.strip()
        
        # Clean the content returned by the formatting agent, remove possible ```json tags
        formatted_json = extract_json_content(formatted_json)
        
        # Validate if JSON is valid
        try:
            json.loads(formatted_json)
            print(f"{Fore.GREEN}The format agent successfully repaired the JSON response{Style.RESET_ALL}")
            return formatted_json
        except json.JSONDecodeError:
            print(f"{Fore.RED}The JSON returned by the format agent is still invalid{Style.RESET_ALL}")
            raise
            
    except Exception as e:
        print(f"{Fore.RED}Error repairing JSON with format agent: {str(e)}{Style.RESET_ALL}")
        raise

def extract_json_content(text):
    """Extract content between ```json and ``` in text"""
    try:
        # Check if there's a ```json tag
        if '```json' in text:
            start = text.find('```json')
            start = text.find('\n', start) + 1  # Skip the ```json line
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()
        # Check if there's a regular ``` tag
        elif '```' in text:
            start = text.find('```')
            start = text.find('\n', start) + 1  # Skip the ``` line
            end = text.find('```', start)
            if end != -1:
                return text[start:end].strip()
        
        # Try to find content between the first { and the last }
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return text[start:end + 1].strip()
            
        return text  # If no tags found, return original text
    except Exception as e:
        print(f"{Fore.RED}Error extracting JSON content: {str(e)}{Style.RESET_ALL}")
        return text

def validate_json(json_str):
    """Validate if string is valid JSON"""
    try:
        json.loads(json_str)
        return True
    except Exception:
        return False

# Modify clean_json_response function to use formatting agent
def clean_json_response(response_content):
    """Clean JSON content in AI response, use formatting agent if fails"""
    try:
        # Extract JSON content
        content = extract_json_content(response_content)
        
        # Validate extracted content
        if not content:
            print(f"{Fore.RED}Warning: Empty content after cleaning{Style.RESET_ALL}")
            print(f"Original response: {response_content}")
            raise ValueError("Empty response content")
        
        # Try to parse JSON
        try:
            json.loads(content)  # Validate if it's valid JSON
            return content
        except json.JSONDecodeError as e:
            print(f"{Fore.RED}JSON parsing error: {str(e)}")
            print(f"Extracted content: {content}{Style.RESET_ALL}")
            
            # Use formatting agent to fix JSON
            print(f"{Fore.YELLOW}Attempting to repair JSON with format agent...{Style.RESET_ALL}")
            fixed_json = fix_json_with_agent(content)
            return fixed_json
            
    except Exception as e:
        print(f"{Fore.RED}Error cleaning JSON response: {str(e)}")
        print(f"Original response: {response_content}{Style.RESET_ALL}")
        
        # Use formatting agent to fix JSON
        print(f"{Fore.YELLOW}Attempting to repair the entire response with format agent...{Style.RESET_ALL}")
        fixed_json = fix_json_with_agent(response_content)
        return fixed_json

# Extract input filename before calling read_csv_data
input_file_path = 'input_file_path'  # Original input file path
input_filename = os.path.splitext(os.path.basename(input_file_path))[0]  # Get filename without extension

# Define output file path
output_filename = f'user_stories_data_super-agent_{model}_{input_filename}.csv'

# Read CSV data
csv_data_df = read_csv_data(input_file_path)

# Check if output file exists, if so read already processed data
processed_goal_nums = set()
results_data = []

if os.path.exists(output_filename):
    try:
        existing_df = pd.read_csv(output_filename, encoding='utf-8')
        # Add processed goal_nums to the set
        processed_goal_nums = set(existing_df['goal_num'].tolist())
        # Load existing results into results_data
        for _, row in existing_df.iterrows():
            results_data.append(row.to_dict())
        print(f"{Fore.GREEN}Found existing output file, {len(processed_goal_nums)} rows have been processed{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error reading existing output file: {str(e)}{Style.RESET_ALL}")
        # If reading fails, start processing from scratch
        processed_goal_nums = set()
        results_data = []

total_rows = len(csv_data_df)
remaining_rows = total_rows - len(processed_goal_nums)
print(f"{Fore.CYAN}Starting to process data, {total_rows} rows in total, {remaining_rows} rows remaining{Style.RESET_ALL}")

# Create format repair agent

for row_index, csv_data in enumerate(csv_data_df.itertuples(index=False), 1):
    # Skip if current row has already been processed
    if row_index in processed_goal_nums:
        print(f"{Fore.YELLOW}Skipping row {row_index}/{total_rows} (already processed){Style.RESET_ALL}")
        continue
        
    print(f"{Fore.CYAN}Processing row {row_index}/{total_rows} ({row_index/total_rows*100:.2f}%){Style.RESET_ALL}")
    
    goal = csv_data.goal
    background = csv_data.background
    problems = csv_data.problems
    
    oneforall_prompt = read_prompt('../prompts_cot/single_agent_cot_prompt.txt')
    current_prompt = build_prompt(
        oneforall_prompt,
        goal=goal,
        background=background,
        problems=problems
    )
    
    # Add retry logic
    max_retries = 3
    valid_json_obtained = False
    json_content = ""
    
    for attempt in range(max_retries):
        print(f"{Fore.YELLOW}Attempt {attempt+1} of {max_retries}: Generating valid JSON...{Style.RESET_ALL}")
        # Add extra prompt for retries, emphasizing valid JSON return
        response = client.chat.completions.create(
            model=model,
            temperature=0.9,
            messages=[
                {"role": "system", "content": description_prompt},
                {"role": "user", "content": current_prompt}
            ],
        )
            
        print(response.choices[0].message.content)
        
        # Extract JSON content
        json_content = clean_json_response(response.choices[0].message.content)
        
        # Validate if JSON is valid
        if validate_json(json_content):
            valid_json_obtained = True
            print(f"{Fore.GREEN}Obtained valid JSON data{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Generated JSON is invalid, attempting to repair with format agent...{Style.RESET_ALL}")
            # Try to fix JSON with format repair agent
            try:
                # Use the newly defined fix_json_with_agent function instead of old call method
                fixed_json = fix_json_with_agent(json_content)
                
                # Validate fixed JSON
                if validate_json(fixed_json):
                    valid_json_obtained = True
                    json_content = fixed_json
                    print(f"{Fore.GREEN}The format repair agent successfully repaired the JSON{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Format repair failed, will retry...{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error in format repair process: {str(e)}, will retry...{Style.RESET_ALL}")
    
    if not valid_json_obtained:
        print(f"{Fore.RED}After {max_retries} attempts, still no valid JSON obtained, using the last generated content{Style.RESET_ALL}")
    
    results_data.append({
        'goal': goal,
        'background': background,
        'problems': problems,
        'userstories_data': json_content,
        'goal_num': row_index,
        'is_valid_json': valid_json_obtained
    })
    
    # Save results immediately after processing each row to prevent data loss in case of interruption
    output_df = pd.DataFrame(results_data)
    output_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"{Fore.GREEN}Saved current progress to {output_filename}{Style.RESET_ALL}")
    
    # if row_index == 2:
    #     break
    
# Final save of results to CSV file
output_df = pd.DataFrame(results_data)
output_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"{Fore.GREEN}Results saved to {output_filename}{Style.RESET_ALL}")

# Print validation result statistics
valid_count = sum(1 for item in results_data if item['is_valid_json'])
total_count = len(results_data)
print(f"{Fore.CYAN}JSON validation results: {valid_count}/{total_count} valid data ({valid_count/total_count*100:.2f}%){Style.RESET_ALL}")
