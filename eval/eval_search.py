import json
import os
from pathlib import Path

import fire
import pandas as pd

from tests.test_tool.test_search_tool import evaluate


def eval_search(env_names: str = "eval:2Wiki,eval:PopQA,eval:TriviaQA,eval:HotpotQA,eval:Bamboogle,eval:NaturalQuestions,eval:Musique",
                model_name: str = "Qwen/Qwen3-1.7B",
                output_dir: str = None,
                **kwargs):
    
    env_names = env_names.split(",")
    
    # Determine output directory
    save_results = False
    if output_dir:
        save_results = True
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Check if model_name is a local directory
        if os.path.isdir(model_name):
            output_dir = Path(model_name)
            output_dir = os.path.join(output_dir.parent, f"eval_{output_dir.stem}")
            save_results = True
            os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    results = []
    all_episodes = {}
    
    print(f"Running evaluation on {len(env_names)} environments...")
    print(f"Model: {model_name}")
    if save_results:
        print(f"Output directory: {output_dir}")
    else:
        print("Results will not be saved (output_dir not specified and model_name is not a local directory)")
    
    # Run evaluation for each environment
    for env_name in env_names:
        print(f"\nEvaluating on {env_name}...")
        
        try:
            acc, episodes = evaluate(
                model_name=model_name,
                env_name=env_name,
                **kwargs
            )
            
            results.append({
                'env_name': env_name,
                'model_name': model_name,
                'accuracy': acc,
                'num_episodes': len(episodes)
            })
            
            all_episodes[env_name] = episodes
            
            print(f"✓ {env_name}: {acc:.2%} accuracy")
            
        except Exception as e:
            print(f"✗ {env_name}: Error - {str(e)}")
            results.append({
                'env_name': env_name,
                'model_name': model_name,
                'accuracy': None,
                'num_episodes': 0,
                'error': str(e)
            })
    
        # Save results if output directory is determined
        if save_results:
            # Save accuracy results to CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, 'evaluation_results.csv')
            df.to_csv(csv_path, index=False)
            
            # Save episodes to JSON
            json_path = os.path.join(output_dir, 'evaluation_episodes.json')
            with open(json_path, 'w') as f:
                json.dump(all_episodes, f, indent=2)
        
    # Print summary
    if save_results:
        print(f"\nAccuracy results saved to: {csv_path}")
        print(f"Episodes saved to: {json_path}")
    print(f"\nSummary:")
    print(f"Total environments: {len(env_names)}")
    successful_results = [r for r in results if r['accuracy'] is not None]
    if successful_results:
        avg_acc = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        print(f"Average accuracy: {avg_acc:.2%}")
    
    return


if __name__ == "__main__":
    fire.Fire(eval_search)