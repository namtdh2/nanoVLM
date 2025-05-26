import sys
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoImageProcessor
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data.datasets import VQADataset
from data.collators import VQACollator
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
import models.config as cfg

def load_model_and_dataset(checkpoint_path, dataset_path):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading weights from: {checkpoint_path}")
    model = VisionLanguageModel.from_pretrained(checkpoint_path).to(device)
    model.eval()

    # Initialize tokenizer and image processor
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    vqa_dataset = VQADataset(dataset['train'], tokenizer, image_processor)
    
    return model, vqa_dataset, tokenizer, device

def evaluate_model(model, dataset, tokenizer, device):
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
            image = sample['image']
            if hasattr(image, 'pixel_values'):
                image = torch.tensor(image.pixel_values).to(device)
            else:
                image = torch.tensor(image).to(device)
            image = image.unsqueeze(0)  # Add batch dimension
            
            # Prepare text input
            text_data = sample['text_data']
            encoded = tokenizer.batch_encode_plus([text_data], return_tensors="pt")
            tokens = encoded["input_ids"].to(device)
            
            # Get model prediction
            outputs = model.generate(
                tokens,
                image,
                max_new_tokens=20,  # Same as generate.py default
            )
            
            # Decode prediction
            predicted_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            predicted_answer = predicted_text.split("Answer:")[-1].strip()
            answer = sample['answer']
            
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
    checkpoint_path = "checkpoints/nanoVLM-sample/"
    dataset_path = "playground/sample_dataset"
    
    # Load model and dataset
    print("Loading model and dataset...")
    model, dataset, tokenizer, device = load_model_and_dataset(checkpoint_path, dataset_path)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluate_model(model, dataset, tokenizer, device)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    main() 