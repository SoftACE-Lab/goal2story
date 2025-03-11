import pandas as pd
import json
import os
from pathlib import Path

def extract_user_stories(json_str, goal_num):
    """Extract all user stories from the JSON string"""
    if not json_str or pd.isna(json_str):
        return []
        
    data = json.loads(json_str)
    stories = []
    
    # Iterate through all actors
    for actor in data.get('actors', []):
        # Iterate through each actor's impacts
        for impact in actor.get('impacts', []):
            # Iterate through each impact's deliverables
            for deliverable in impact.get('deliverables', []):
                # Get user story
                if 'user_story' in deliverable:
                    story = deliverable['user_story']
                    stories.append({
                        'story_id': story.get('story_id'),
                        'us_actor': story.get('actor'),
                        'us_action': story.get('action'),
                        'us_expected_outcome': story.get('expected_output'),
                        'goal_num': goal_num
                    })
    
    return stories

def process_csv_file(input_file, output_dir):
    """Process a single CSV file and save the results"""
    df = pd.read_csv(input_file)
    
    # Store all parsed user stories
    all_stories = []
    
    # Iterate through all_userstories column and goal_num column
    for json_str, goal_num in zip(df['userstories_data'], df['goal_num']):
        try:
            # Skip empty rows or NaN values
            if pd.isna(json_str) or json_str == '':
                continue
                
            stories = extract_user_stories(json_str, goal_num)
            all_stories.extend(stories)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in {input_file}: {e}")
        except Exception as e:
            print(f"Error in {input_file}: {e}")
    
    # Convert results to DataFrame and save
    stories_df = pd.DataFrame(all_stories)
    
    # Build the output file path
    input_filename = os.path.basename(input_file)
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the results
    if not stories_df.empty:
        stories_df.to_csv(output_path, index=False)
        print(f"File {input_filename} processed, {len(stories_df)} user stories parsed")
        print(f"Results saved to {output_path}")
    else:
        print(f"File {input_filename} processed, but no user stories parsed")

def main():
    # Set the input and output directories
    input_dir = 'generated_answer_file_path'
    output_dir = 'processed_generated_answer_file_path'
    
    # Create the output directory (if it doesn't exist)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Process each CSV file
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        process_csv_file(input_path, output_dir)
    
    print(f"All files processed, results saved in {output_dir} directory")

if __name__ == "__main__":
    main() 