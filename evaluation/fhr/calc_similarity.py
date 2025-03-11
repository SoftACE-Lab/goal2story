import os
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")

def load_csv_data(file_path):
    """Load CSV file data"""
    return pd.read_csv(file_path)

def get_embeddings(text, client):
    """Get the embeddings of the text"""
    if pd.isna(text):
        return np.zeros(1024)  # Use a 1024-dimensional zero vector to represent empty values
    
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=str(text),
        dimensions=1024,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

def calculate_similarity(row1, row2, client, columns):
    """Calculate the similarity of the corresponding columns in two rows of data"""
    similarities = {}
    
    for col in columns:
        # Get the embeddings of the two texts
        emb1 = get_embeddings(row1[col], client)
        emb2 = get_embeddings(row2[col], client)
        
        # Calculate the cosine similarity
        similarity = cosine_similarity(
            emb1.reshape(1, -1), 
            emb2.reshape(1, -1)
        )[0][0]
        
        similarities[col] = similarity
    
    return similarities

def main():
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # Load CSV file
    df1 = load_csv_data('standard_answer_file_path')
    input_file = 'generated_answer_file_path'
    df2 = load_csv_data(input_file)
    
    # Get the directory and file name of the input file
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    output_filename = f'similarity_{input_filename}'
    
    # The columns to compare
    columns_to_compare = ['us_actor', 'us_action', 'us_expected_outcome']
    all_results = []
    # Process each row of the first file
    for row_index, (idx1, row1) in enumerate(df1.iterrows(), 1):
        print(f"\nProcessing the user story of goal_num {row_index}:")
        
        # Filter the rows in the second file with the same goal_num
        matching_rows = df2[df2['goal_num'] == row_index]
        
        # Calculate the similarity for each row in the filtered rows
        results = []
        for idx2, row2 in matching_rows.iterrows():
            similarities = calculate_similarity(row1, row2, client, columns_to_compare)
            results.append({
                'row_index': idx2,
                'goal_num': row_index,
                **similarities
            })
        
        if not results:
            print(f"No matching item found in the second file for goal_num {row_index}")
            continue
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        print(results_df)
        all_results.append(results_df)
        # Print detailed matching information
        for col in columns_to_compare:
            max_sim_idx = results_df[col].idxmax()
            print(f"\nThe best match for the {col} column:")
            print(f"Similarity: {results_df.iloc[max_sim_idx][col]:.4f}")
            print(f"Original text: {row1[col]}")
            matched_row_index = results_df.iloc[max_sim_idx]['row_index']
            matched_text = df2.loc[matched_row_index, col]
            print(f"Matched text: {matched_text}")
        # if row_index == 2:
        #     break
    # Merge all results into a DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Save the merged results to a CSV file
    all_results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    main()
