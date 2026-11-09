import unittest
import numpy as np
import pandas as pd
from main import (
    Autoencoder,
    IsolationForest,
    prepare,
    reconstruction_scores,
    percentile_threshold,
    drift_analysis,
    classify_warning,
    metrics,
    add_physics_features,
)


class PipelineTests(unittest.TestCase):
    def test_preprocessing_excludes_targets_and_fits_train_only(self):
        d = pd.DataFrame({
            "Air temperature [K]": [300.0, 301.0, 999.0],
            "Process temperature [K]": [310.0, 311.0, 999.0],
            "Rotational speed [rpm]": [1000, 1100, 5000],
            "Torque [Nm]": [30.0, 31.0, 99.0],
            "Tool wear [min]": [1, 2, 999],
            "Type": ["L", "M", "H"],
            "Machine failure": [0, 0, 1],
        })
        # Base features (without physics features)
        f_base, s_base = prepare(d, np.array([0, 1]), include_physics=False)
        x_base = f_base(d)
        self.assertEqual(x_base.shape, (3, 8))
        self.assertEqual(s_base["category_levels"], ["H", "L", "M"])
        self.assertNotIn("Machine failure", s_base["input_features"])
        self.assertGreater(x_base[2, 0], 100)

        # Physics features enabled
        f_phys, s_phys = prepare(d, np.array([0, 1]), include_physics=True)
        x_phys = f_phys(d)
        self.assertEqual(x_phys.shape, (3, 11))
        self.assertIn("Power [W]", s_phys["numeric_features"])
        self.assertIn("Temp difference [K]", s_phys["numeric_features"])
        self.assertIn("Overstrain [min Nm]", s_phys["numeric_features"])

    def test_score_threshold_and_model(self):
        m = Autoencoder(3, 2, seed=1)
        x = np.zeros((4, 3))
        self.assertTrue(np.all(reconstruction_scores(m, x) >= 0))
        self.assertEqual(percentile_threshold([1, 2, 3], 95), 2.9)

    def test_adam_training_reduces_loss(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 6))
        m = Autoencoder(6, 3, seed=42, lr=0.01)
        initial_loss = float(np.mean((m.forward(x) - x) ** 2))
        for _ in range(30):
            m.train_batch(x)
        final_loss = float(np.mean((m.forward(x) - x) ** 2))
        self.assertLess(final_loss, initial_loss)

    def test_metrics_include_imbalance_aware_and_accuracy_metrics(self):
        result = metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("pr_auc_average_precision", result)

    def test_drift_persistence_and_warning(self):
        r, d = drift_analysis([0, 2, 2, 2, 2], 1, 3, 0.6)
        self.assertTrue(d[-1])
        state, sev, reason = classify_warning(2, 1, float(r[-1]), 0.01)
        self.assertEqual(state, "SUSTAINED_HIGH")
        self.assertEqual(sev, "high")

    def test_isolation_forest_scores(self):
        x = np.random.default_rng(1).normal(size=(40, 2))
        model = IsolationForest(n_trees=5, sample_size=16, seed=1).fit(x)
        s = model.score_samples(np.array([[0.0, 0.0], [8.0, 8.0]]))
        self.assertEqual(s.shape, (2,))
        self.assertTrue(np.all(np.isfinite(s)))


if __name__ == "__main__":
    unittest.main()
