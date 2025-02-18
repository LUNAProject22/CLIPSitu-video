from pycocoevalcap.cider.cider import Cider
import evaluate
import numpy as np
metrics = {}
for name in ['rouge', 'bleu', 'meteor']:
    metrics[name] = evaluate.load(name)
metrics['cider'] = Cider()

def compute_metrics(decoded_labels, decoded_predictions):
    print("GT: ", decoded_labels)
    print("Pred: ", decoded_predictions)
    
    result = {}
    for key, metric in metrics.items():
        if key in ['rouge', 'bleu', 'meteor']:
            result = metric.compute(predictions=decoded_predictions, references=decoded_labels)
        elif key in ['cider']:
            result = {'cider':metric.compute_score({cs: [decoded_labels[cs]] for cs in range(len(decoded_labels))}, {cs: [decoded_predictions[cs]] for cs in range(len(decoded_labels))})[1]}
    return result