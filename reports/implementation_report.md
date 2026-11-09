# Early Behavioral Drift Detection in Industrial Machines

## Abstract
This report records the executed AI4I prototype: leakage-aware preprocessing, normal-only autoencoder training with Adam optimization, anomaly scoring, temporal warning outputs, controlled synthetic drift evaluation, and held-out baseline comparison.

## Dataset and preprocessing
The CSV has 10,000 rows and 14 columns, with no missing values or duplicate rows. It contains identifiers UDI and Product ID; categorical Type; five raw sensor inputs; and six failure-related targets. UDI, Product ID, and all targets are excluded from model inputs. Type is one-hot encoded; numeric scaling and category vocabularies are fitted on training rows only. Domain physics signals (Temperature Difference, Power, and Overstrain) are derived to model operational regimes without label leakage. The seed 42 split contains train/validation/test sizes 6999/1500/1501. The autoencoder uses 6762 non-failure training examples.

## Model and threshold
A fine-tuned NumPy neural network has a 7-unit tanh bottleneck, linear decoder, mini-batch Adam optimization with weight decay, gradient clipping, and mean squared reconstruction loss. Early stopping monitors normal validation loss. Training stopped after 150 epochs; final training and validation losses were 0.0035572632 and 0.003581941. The anomaly threshold is 0.021154126, derived from the 98.0th percentile of normal validation reconstruction errors.

## Held-out evaluation
The test partition contains 51 failures and 1450 non-failures.
- **Autoencoder**: accuracy 0.9654, precision 0.4889, recall 0.4314, F1 0.4583, PR-AUC average precision 0.4743, balanced accuracy 0.7078; TN=1427, FP=23, FN=29, TP=22.
- **Statistical Mahalanobis baseline**: accuracy 0.9647, precision 0.4783, recall 0.4314, F1 0.4536, PR-AUC 0.4227.
- **Isolation Forest baseline**: accuracy 0.9607, precision 0.3571, recall 0.1961, F1 0.2532, PR-AUC 0.2457.

## Temporal drift and warnings
Rolling exceedance fraction across 25 adjacent source rows with persistence threshold 0.6 provides ordered-row drift triage. In a controlled synthetic sequence, a sensor shift was injected at index 300; the detector delay was 14 observations.

## Reproducibility and outputs
Run `python main.py` from the repository root. Configuration and random seed are in `config.json`; library versions, audit and metrics are in `outputs/metrics/`. Predictions, baseline metrics, figures, and saved model artifacts are under `outputs/`.
