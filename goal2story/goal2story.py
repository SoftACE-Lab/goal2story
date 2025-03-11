from colorama import Fore, Style
import json
import csv
import pandas as pd
import os
import sys  # Import sys module for command line arguments
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

description_prompt = "You are an agile requirement expert and good at analyzing goal, actor, impact, deliverable and user story."

client = OpenAI(
    base_url="your_base_url",
    api_key="your_api_key",
)

model = "model name"

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
    with open(os.path.join('../prompts', file_name), 'r', encoding='utf-8') as f:
        return f.read()

def build_prompt(template, **kwargs):
    """Build prompt, handle variable replacement"""
    try:
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                processed_kwargs[key] = repr(json.dumps(value, indent=2))[1:-1]
            else:
                processed_kwargs[key] = value
        
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
        format_prompt = f"""
        Please repair the following JSON format, only return the repaired valid JSON content, do not add any explanation or other text, do not use code block tags:
        
        {json_content}
        """
        
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,  # Low temperature for more deterministic output
            messages=[
                {"role": "system", "content": "You are a professional JSON format repair expert. Only return the repaired valid JSON content, do not add any explanation or other text."},
                {"role": "user", "content": format_prompt}
            ]
        )
        
        formatted_json = response.choices[0].message.content.strip()
        
        # Clean up the content returned by the formatting agent, remove possible ```json tags
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
        print(f"{Fore.RED}Error repairing JSON with the format agent: {str(e)}{Style.RESET_ALL}")
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
            
        return text  # If no tags found, return the original text
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
    """Clean JSON content in AI response, use formatting agent if failed"""
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
            print(f"{Fore.YELLOW}Attempting to repair JSON with the format agent...{Style.RESET_ALL}")
            fixed_json = fix_json_with_agent(content)
            return fixed_json
            
    except Exception as e:
        print(f"{Fore.RED}Error cleaning JSON response: {str(e)}")
        print(f"Original response: {response_content}{Style.RESET_ALL}")
        
        # Use formatting agent to fix JSON
        print(f"{Fore.YELLOW}Attempting to repair the full response with the format agent...{Style.RESET_ALL}")
        fixed_json = fix_json_with_agent(response_content)
        return fixed_json


input_file_path = 'input_file_path'  # Default input file path

# Get filename without extension
input_filename = os.path.splitext(os.path.basename(input_file_path))[0]

csv_data_df = read_csv_data(input_file_path)
print(csv_data_df)

# Check if existing CSV file exists before starting main loop
output_csv_path = f'user_stories_data_goal2story_{model}_{input_filename}.csv'
existing_results = []
if os.path.exists(output_csv_path):
    try:
        existing_df = pd.read_csv(output_csv_path)
        existing_results = existing_df.to_dict('records')
        print(f"{Fore.GREEN}Found existing data file, processed rows: {len(existing_results)}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error reading existing CSV file: {str(e)}{Style.RESET_ALL}")

# Get actors
for row_index, csv_data in enumerate(csv_data_df.itertuples(index=False), 1):
    # Check if this row has already been processed
    if any(result['row_id'] == row_index for result in existing_results):
        print(f"{Fore.YELLOW}Skipping already processed row {row_index}{Style.RESET_ALL}")
        continue

    goal = csv_data.goal
    background = csv_data.background
    problems = csv_data.problems
    
    try:
        actor_prompt = read_prompt('actor_prompt.txt')
        current_prompt = build_prompt(
            actor_prompt,
            goal=goal,
            background=background,
            problems=problems
        )
        max_retries = 3
        retry_count = 0
        success = False
        all_actors = {"actors": {}}

        while retry_count < max_retries and not success:
            try:
                #response = agent.step(current_prompt)
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.9,
                    messages=[
                        {"role": "system", "content": description_prompt},
                        {"role": "user", "content": current_prompt}
                    ],

                )
                print(Fore.CYAN + response.choices[0].message.content + Style.RESET_ALL)
                # Clean response content
                cleaned_response = clean_json_response(response.choices[0].message.content)
                actor_response = json.loads(cleaned_response)
                
                # Convert array format to required dictionary format
                all_actors = {
                    "actors": {
                        actor["actor_id"]: {"name": actor["actor_name"]} 
                        for actor in actor_response["actors"]
                    }
                }
                success = True
                print(all_actors)
            except Exception as e:
                retry_count += 1
                print(f"{Fore.RED}Error getting actors (attempt {retry_count}/{max_retries}): {str(e)}{Style.RESET_ALL}")
                if retry_count == max_retries:
                    raise Exception("Unable to successfully get actors")

        # Get prompt template for single actor impact
        single_impact_prompt = read_prompt('impact_prompt.txt')

        # Store results for all impacts
        all_impacts = {
            "goal": goal,
            "actors": []
        }

        # Get impact for each actor separately
        for actor_id, actor_info in all_actors["actors"].items():
            # Build data structure for single actor
            single_actor = {
                "actor": {
                    "actor_id": actor_id,
                    "actor_name": actor_info["name"]
                }
            }   
            
            try:
                # Use build_prompt function to construct prompt
                current_prompt = build_prompt(single_impact_prompt, actor=single_actor, goal=goal, background=background, problems=problems)
                
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # response = agent.step(current_prompt)
                        response = client.chat.completions.create(
                            model=model, 
                            temperature=0.9,
                            messages=[
                                {"role": "system", "content": description_prompt},
                                {"role": "user", "content": current_prompt}
                            ],

                        )
                        # Add debug output for response content
                        print(f"\n{Fore.CYAN}Raw response from agent:{Style.RESET_ALL}")
                        print(response.choices[0].message.content)
                        
                        cleaned_response = clean_json_response(response.choices[0].message.content)
                        impact_response = json.loads(cleaned_response)
                        
                        # Build actor data structure in target format
                        actor_data = {
                            "actor_id": actor_id,
                            "actor_name": actor_info["name"],
                            "impacts": []
                        }
                        
                        # Add impacts
                        for impact in impact_response["impacts"]:
                            impact_data = {
                                "impact_id": impact["impact_id"],
                                "impact_description": impact["impact_name"]
                            }
                            actor_data["impacts"].append(impact_data)
                        
                        # Add actor data to results
                        all_impacts["actors"].append(actor_data)
                        
                        success = True
                        
                        # Print current actor's impact
                        print(f"\n{Fore.YELLOW}Impact for {actor_info['name']}:")
                        print(json.dumps(actor_data, indent=2))
                        print(Style.RESET_ALL)
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            print(f"{Fore.RED}Error processing impact response: {str(e)}{Style.RESET_ALL}")
                            raise
                        
            except Exception as e:
                print(f"{Fore.RED}Error processing actor {actor_info['name']}: {str(e)}{Style.RESET_ALL}")
                continue

        # Print final complete result
        print(f"\n{Fore.GREEN}Final combined impacts:")
        print(json.dumps(all_impacts, indent=2))
        print(Style.RESET_ALL)

        # Get prompt template for deliverable
        deliverable_prompt = read_prompt('deliverable_prompt.txt')

        # Store results for all deliverables
        all_deliverables = {
            "goal": all_impacts["goal"],
            "actors": []
        }

        # Get deliverable for each actor and each corresponding impact
        for actor in all_impacts["actors"]:
            actor_deliverables = {
                "actor_id": actor["actor_id"],
                "actor_name": actor["actor_name"],
                "impacts": []
            }
            
            # Process each impact for this actor
            for impact in actor["impacts"]:
                try:
                    # Build data structure for current analysis
                    current_data = {
                        "actor": {
                            "actor_id": actor["actor_id"],
                            "actor_name": actor["actor_name"]
                        },
                        "impact": {  # Changed to single impact
                            "impact_id": impact["impact_id"],
                            "impact_description": impact["impact_description"]
                        }
                    }
                    
                    
                    # Use build_prompt function to construct prompt
                    current_prompt = build_prompt(deliverable_prompt,goal=goal, background=background, problems=problems, **current_data)
                    
                    max_retries = 3
                    retry_count = 0
                    success = False
                    
                    while retry_count < max_retries and not success:
                        try:
                            # response = agent.step(current_prompt)
                            response = client.chat.completions.create(
                                model=model,
                                temperature=0.9,
                                messages=[
                                    {"role": "system", "content": description_prompt},
                                    {"role": "user", "content": current_prompt}
                                ],

                            )
                            # Add debug output for response content
                            print(f"\n{Fore.CYAN}Raw response from agent:{Style.RESET_ALL}")
                            print(response.choices[0].message.content)
                            
                            cleaned_response = clean_json_response(response.choices[0].message.content)
                            deliverable_result = json.loads(cleaned_response)
                            
                            # Build complete impact data including deliverables
                            impact_data = {
                                "impact_id": impact["impact_id"],
                                "impact_description": impact["impact_description"],
                                "deliverables": [
                                    {
                                        "deliverable_id": d["deliverable_id"],
                                        "deliverable_description": d["deliverable_name"]
                                    }
                                    for d in deliverable_result["deliverables"]
                                ]
                            }
                            
                            # Add to actor's impacts list
                            actor_deliverables["impacts"].append(impact_data)
                            
                            success = True
                            
                            # Print current deliverable result
                            print(f"\n{Fore.YELLOW}Deliverables for {actor['actor_name']} - Impact {impact['impact_id']}:")
                            print(json.dumps(impact_data["deliverables"], indent=2))
                            print(Style.RESET_ALL)
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count == max_retries:
                                print(f"{Fore.RED}Error processing deliverable response (attempt {retry_count}/{max_retries}): {str(e)}{Style.RESET_ALL}")
                                if retry_count == max_retries:
                                    # If all retries fail, continue with empty deliverables list
                                    print(f"{Fore.YELLOW}Using empty deliverables list for this impact{Style.RESET_ALL}")
                                    deliverable_result = {"deliverables": []}
                                    success = True
                            
                except Exception as e:
                    print(f"{Fore.RED}Error processing actor {actor['actor_name']} impact {impact['impact_id']}: {str(e)}{Style.RESET_ALL}")
                    continue
            
            # Add processed actor data to results
            all_deliverables["actors"].append(actor_deliverables)

        # Print final complete deliverables result
        print(f"\n{Fore.GREEN}Final combined deliverables:")
        print(json.dumps(all_deliverables, indent=2))
        print(Style.RESET_ALL)

        # Get prompt template for user story
        userstory_prompt = read_prompt('userstory_prompt.txt')

        # Store results for all user stories
        all_userstories = {
            "goal": all_deliverables["goal"],
            "actors": []
        }

        # Get user story for each actor and each corresponding impact and deliverable
        for actor in all_deliverables["actors"]:
            actor_data = {
                "actor_id": actor["actor_id"],
                "actor_name": actor["actor_name"],
                "impacts": []
            }
            
            # Process each impact for this actor
            for impact in actor["impacts"]:
                current_data = {
                        "actor": {
                            "actor_id": actor["actor_id"],
                            "actor_name": actor["actor_name"]
                        },
                        "impact": {
                            "impact_id": impact["impact_id"],
                            "impact_name": impact["impact_description"]
                        },
                        "deliverable": impact["deliverables"]
                    }
                # Use build_prompt function to construct prompt
                current_prompt = build_prompt(userstory_prompt, goal=goal, background=background, problems=problems, **current_data)
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # response = agent.step(current_prompt)
                        response = client.chat.completions.create(
                            model=model,
                            temperature=0.9,
                            messages=[
                                {"role": "system", "content": description_prompt},
                                {"role": "user", "content": current_prompt}
                            ],
                        )
                        cleaned_response = clean_json_response(response.choices[0].message.content)
                        userstory_result = json.loads(cleaned_response)
                        
                        # Build complete impact data including all deliverables' user stories
                        impact_data = {
                            "impact_id": impact["impact_id"],
                            "impact_description": impact["impact_description"],
                            "deliverables": [
                                {
                                    "deliverable_id": d["deliverable_id"],
                                    "deliverable_description": d["deliverable_description"],
                                    "user_story": us["user_story"]
                                }
                                for d, us in zip(impact["deliverables"], userstory_result["deliverables"])
                            ]
                        }
                        
                        success = True
                        
                        # Print current user stories result
                        print(f"\n{Fore.YELLOW}User Stories for {actor['actor_name']} - Impact {impact['impact_id']}:")
                        print(json.dumps(impact_data["deliverables"], indent=2))
                        print(Style.RESET_ALL)
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            print(f"{Fore.RED}Error processing user stories response: {str(e)}{Style.RESET_ALL}")
                            raise
                            
                # Add processed impact data to actor's impacts list
                actor_data["impacts"].append(impact_data)

            # Add processed actor data to results
            all_userstories["actors"].append(actor_data)

        # Print final complete user stories result
        print(f"\n{Fore.GREEN}Final combined user stories:")
        print(json.dumps(all_userstories, indent=2))
        print(Style.RESET_ALL)

        # Save after completing each row of data
        current_result = {
            'row_id': row_index,
            'goal': goal,
            'actors_data': json.dumps(all_actors, ensure_ascii=False),
            'impacts_data': json.dumps(all_impacts, ensure_ascii=False),
            'deliverables_data': json.dumps(all_deliverables, ensure_ascii=False),
            'userstories_data': json.dumps(all_userstories, ensure_ascii=False),
            'goal_num': row_index
        }
        
        # Add new result to existing results
        existing_results.append(current_result)
        
        # Save all updated results
        updated_df = pd.DataFrame(existing_results)
        updated_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"{Fore.GREEN}Saved row {row_index} results to {output_csv_path}{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Error processing row {row_index}: {str(e)}{Style.RESET_ALL}")
        # Previous results are already saved even if an error occurs
        continue

    # if row_index == 2:
    #     break