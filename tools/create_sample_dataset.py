from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as DatasetImage
import os
import shutil

def create_sample_dataset():
    # Load the image
    image_path = "assets/image.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Create sample text
    sample = {
        "user": "Question: How many actions are depicted in the diagram?\nChoices:\nA. 6.\nB. 4.\nC. 8.\nD. 7.\nAnswer with the letter.",
        "assistant": "Answer: D",
        "source": "TQA"
    }

    # Define features explicitly to ensure proper image handling
    features = Features({
        'images': Sequence(DatasetImage()),
        'texts': {
            'user': Value('string'),
            'assistant': Value('string'),
            'source': Value('string')
        }
    })

    # Create dataset entries
    dataset_entries = []
    for _ in range(100):  # Create 100 samples
        entry = {
            'images': [image],  # List of images
            'texts': sample   # Single text sample dictionary
        }
        dataset_entries.append(entry)
    
    # Create the dataset with explicit features
    dataset = Dataset.from_list(dataset_entries, features=features)
    
    # Create a DatasetDict with train split
    dataset_dict = DatasetDict({
        'train': dataset
    })

    # Save the dataset
    output_dir = "playground/sample_dataset"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_dict.save_to_disk(output_dir)
    print(f"Sample dataset created and saved to '{output_dir}' directory")
    print(f"Dataset size: {len(dataset)} samples")

if __name__ == "__main__":
    create_sample_dataset()