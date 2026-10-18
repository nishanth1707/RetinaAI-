import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

print("=" * 60)
print("  DR Detection System - Model Training Pipeline")
print("=" * 60)
print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-data", type=str, default=None, help="Path to APTOS dataset")
    parser.add_argument("--segmenter-data", type=str, default=None, help="Path to IDRiD dataset")
    parser.add_argument("--classifier-only", action="store_true", help="Train only classifier")
    parser.add_argument("--segmenter-only", action="store_true", help="Train only segmenter")
    parser.add_argument("--epochs-cls", type=int, default=20, help="Classifier epochs")
    parser.add_argument("--epochs-seg", type=int, default=30, help="Segmenter epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for both")
    args = parser.parse_args()

    if not args.classifier_only and not args.segmenter_only:
        print("Will train both models.")
        print()

    if not args.segmenter_only:
        print("-" * 40)
        print("  Training DR Classifier (Module 1)")
        print("-" * 40)
        from src.train_classifier import train_classifier
        bs = args.batch_size if args.batch_size else 16
        train_classifier(data_dir=args.classifier_data, num_epochs=args.epochs_cls, batch_size=bs)
        print()

    if not args.classifier_only:
        print("-" * 40)
        print("  Training DR Segmenter (Module 2)")
        print("-" * 40)
        from src.train_segmenter import train_segmenter
        bs = args.batch_size if args.batch_size else 4
        train_segmenter(data_dir=args.segmenter_data, num_epochs=args.epochs_seg, batch_size=bs)
        print()

    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print()
    print("Run the dashboard:")
    print("  streamlit run app/dashboard.py")
