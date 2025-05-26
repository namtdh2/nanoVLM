import sys
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from safetensors.torch import load_file

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import VQADataset
from data.collators import VQACollator
from models.vision_language_model import VisionLanguageModel
import models.config as cfg

def load_model_and_dataset(checkpoint_path, dataset_path):
    # Initialize tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    vqa_dataset = VQADataset(dataset['train'], tokenizer, image_processor)
    
    # Initialize model
    model = VisionLanguageModel(cfg.VLMConfig())
    
    # Load checkpoint from safetensors
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, vqa_dataset, tokenizer

def evaluate_model(model, dataset, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    collator = VQACollator(tokenizer, max_length=cfg.VLMConfig.lm_max_length)
    
    # Metrics
    total_samples = len(dataset)
    correct_predictions = 0
    predictions = []
    ground_truth = []
    
    # Evaluation loop
    with torch.no_grad():
        for idx in tqdm(range(total_samples), desc="Evaluating"):
            sample = dataset[idx]
            
            # Prepare input
            image = sample['image'].unsqueeze(0).to(device)
            text_data = sample['text_data']
            answer = sample['answer']
            
            # Get model prediction
            outputs = model.generate(
                image=image,
                text=text_data,
                max_length=cfg.VLMConfig.lm_max_length,
                num_beams=1,
                do_sample=False
            )
            
            # Decode prediction
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = predicted_text.split("Answer:")[-1].strip()
            
            # Store results
            predictions.append(predicted_answer)
            ground_truth.append(answer)
            
            # Check if prediction is correct
            if predicted_answer == answer:
                correct_predictions += 1
    
    # Calculate metrics
    accuracy = correct_predictions / total_samples
    
    # Calculate confusion matrix for multiple choice answers
    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    confusion_matrix = np.zeros((4, 4))
    
    for pred, true in zip(predictions, ground_truth):
        if pred in answer_mapping and true in answer_mapping:
            pred_idx = answer_mapping[pred]
            true_idx = answer_mapping[true]
            confusion_matrix[true_idx][pred_idx] += 1
    
    # Calculate per-class accuracy
    per_class_accuracy = []
    for i in range(4):
        total = np.sum(confusion_matrix[i])
        if total > 0:
            per_class_accuracy.append(confusion_matrix[i][i] / total)
        else:
            per_class_accuracy.append(0)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'per_class_accuracy': per_class_accuracy,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

def print_results(results):
    print("\n=== Benchmark Results ===")
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    
    print("\nPer-class Accuracy:")
    classes = ['A', 'B', 'C', 'D']
    for cls, acc in zip(classes, results['per_class_accuracy']):
        print(f"Class {cls}: {acc:.2%}")
    
    print("\nConfusion Matrix:")
    print("True\Pred\tA\tB\tC\tD")
    for i, cls in enumerate(classes):
        row = results['confusion_matrix'][i]
        print(f"{cls}\t\t{int(row[0])}\t{int(row[1])}\t{int(row[2])}\t{int(row[3])}")
    
    print("\nSample Predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"Ground Truth: {results['ground_truth'][i]}")
        print(f"Prediction: {results['predictions'][i]}")
        print("---")

def main():
    # Configuration
    checkpoint_path = "checkpoints/nanoVLM-sample/model.safetensors"
    dataset_path = "playground/sample_dataset"
    
    # Load model and dataset
    print("Loading model and dataset...")
    model, dataset, tokenizer = load_model_and_dataset(checkpoint_path, dataset_path)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluate_model(model, dataset, tokenizer)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    main() 