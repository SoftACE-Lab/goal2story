from colorama import Fore, Style
import json
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
load_dotenv()

client = OpenAI(
    base_url="your_base_url",
    api_key="your_api_key",
)

model = "model_name"



def read_prompt(file_name):
    """Read prompt file content"""
    with open(os.path.join('', file_name), 'r', encoding='utf-8') as f:
        return f.read()

def build_prompt(template, **kwargs):
    """Build prompt by replacing variables in the template"""
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

def validate_user_story(goal, us_actor, us_action, us_expected_outcome):
    """Validate user story and return validation result"""
    # Read CQUS-prompt
    prompt_template = read_prompt("cqus_prompt.txt")
    
    # Build prompt
    prompt = build_prompt(
        prompt_template,
        goal=goal,
        actor=us_actor,
        action=us_action,
        expected_outcome=us_expected_outcome
    )
    # print(prompt)
    
    # Call LLM
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # Lower temperature for more deterministic results
        messages=[
            {"role": "system", "content": "You are a validation agent that will validate the user story based on several criterias if the user story meets every criteria, return 1, otherwise return the failed criteria name."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content.strip()
    # Try to extract 0 or 1 from the result
    if "0" in result:
        return 0
    elif "1" in result:
        return 1
    else:
        print(f"{Fore.YELLOW}Warning: Unable to extract a clear validation result from the response, returning the original response: {result}{Style.RESET_ALL}")
        return result

def process_csv(input_file, output_file):
    """Process CSV file and add validation results"""
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Create new validation column
        df['validation'] = None
        
        # Add statistics variables
        valid_count = 0
        total_processed = 0
        skipped_count = 0
        
        # Process each row
        for index, row in df.iterrows():
            print(f"{Fore.BLUE}Processing row {index+1}/{len(df)}...{Style.RESET_ALL}")
            
            # Extract required fields and handle NaN values
            goal = str(row.get('goal', '')).strip()
            us_actor = str(row.get('generated_us_actor', '')).strip()
            us_action = str(row.get('generated_us_action', '')).strip()
            us_expected_outcome = str(row.get('generated_us_expected_outcome', '')).strip()
            
            # Check if any fields are empty or 'nan'
            if any(field.lower() in ['', 'nan', 'none'] for field in [goal, us_actor, us_action, us_expected_outcome]):
                print(f"{Fore.YELLOW}Skipping row {index + 1}: Empty fields or NaN values exist")
                print(f"goal: '{goal}'")
                print(f"us_actor: '{us_actor}'")
                print(f"us_action: '{us_action}'")
                print(f"us_expected_outcome: '{us_expected_outcome}'{Style.RESET_ALL}")
                skipped_count += 1
                continue
            
            print(f" As a {us_actor}, I want to {us_action}, so that {us_expected_outcome}")
            
            # Validate user story
            validation_result = validate_user_story(
                 goal, us_actor, us_action, us_expected_outcome
            )
            
            # Save validation result
            df.at[index, 'validation'] = validation_result
            
            # Update statistics
            total_processed += 1
            if validation_result == 1:
                valid_count += 1
            
            print(f"{Fore.GREEN}Row {index+1} validation result: {validation_result}{Style.RESET_ALL}")
        
        # Save results to new CSV file
        df.to_csv(output_file, index=False)
        
        # Output statistics
        print(f"{Fore.CYAN}Statistics:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Total rows: {len(df)} rows{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Skipped rows: {skipped_count} rows{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Processed: {total_processed} user stories{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Valid user stories (result is 1): {valid_count} user stories{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Valid rate: {valid_count/total_processed*100:.2f}%{Style.RESET_ALL}" if total_processed > 0 else "No records processed")
        
        print(f"{Fore.GREEN}Processing completed! Results saved to {output_file}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}Error processing CSV: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    # Check command line arguments
    parser = argparse.ArgumentParser(description='Validate user stories and generate validation results')
    parser.add_argument('input', type=str, help='Input CSV file path')
    # Output file is fixed to us_eval.csv in the same directory
    args = parser.parse_args()
    output = os.path.join(os.path.dirname(args.input), 'us_eval.csv')
    print(f"{Fore.BLUE}Processing CSV file: {args.input}{Style.RESET_ALL}")
    process_csv(args.input, output)
