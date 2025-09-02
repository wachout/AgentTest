import numpy as np
import os

def generate_data(num_samples, num_features, num_classes, train_file, label_file):
    print(f"Generating {num_samples} samples with {num_features} features...")

    try:
        # Generate random data
        X = np.random.rand(num_samples, num_features).astype(np.float32)

        # Generate random labels
        y = np.random.randint(0, num_classes, size=num_samples)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(train_file), exist_ok=True)

        # Save training data
        print(f"Saving training data to {train_file}...")
        with open(train_file, 'w') as f:
            for i in range(num_samples):
                f.write(','.join(map(str, X[i])))
                f.write('\n')

        # Save labels
        print(f"Saving labels to {label_file}...")
        with open(label_file, 'w') as f:
            for label in y:
                f.write(str(label))
                f.write('\n')

        print("Data generation complete.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    NUM_SAMPLES = 100
    NUM_FEATURES = 10000
    NUM_CLASSES = 5

    generate_data(NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES, 'data/train_large.txt', 'data/label_large.txt')
