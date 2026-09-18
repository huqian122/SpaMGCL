import unittest

import numpy as np

from src.clustering.refinement import (
    apply_embedding_refinement,
    boundary_aware_spatial_residual_refinement,
)


def _synthetic_inputs():
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    embedding = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.8, 0.2, 0.1],
            [0.1, 0.9, 0.0],
            [0.9, 0.1, 0.3],
            [0.0, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    return embedding, coordinates


class BSRRTest(unittest.TestCase):
    def test_bsrr_synthetic_contract(self):
        embedding, coordinates = _synthetic_inputs()

        refined, diagnostics = boundary_aware_spatial_residual_refinement(
            embedding, coordinates, spatial_k=3
        )

        self.assertEqual(refined.shape, embedding.shape)
        self.assertTrue(np.isfinite(refined).all())
        self.assertGreater(diagnostics["sigma_spatial"], 0)
        self.assertGreater(diagnostics["sigma_latent"], 0)
        self.assertGreaterEqual(diagnostics["confidence_min"], 0.0)
        self.assertLessEqual(diagnostics["confidence_min"], 1.0)
        self.assertGreaterEqual(diagnostics["confidence_max"], 0.0)
        self.assertLessEqual(diagnostics["confidence_max"], 1.0)

    def test_bsrr_rejects_spot_count_mismatch(self):
        embedding, coordinates = _synthetic_inputs()

        with self.assertRaisesRegex(ValueError, "same number of spots"):
            boundary_aware_spatial_residual_refinement(
                embedding, coordinates[:-1], spatial_k=3
            )

    def test_disabled_refinement_preserves_embedding(self):
        embedding, coordinates = _synthetic_inputs()

        unchanged, diagnostics = apply_embedding_refinement(
            embedding,
            coordinates,
            enabled=False,
            method="bsrr",
            spatial_k=3,
        )

        self.assertTrue(np.array_equal(unchanged, embedding))
        self.assertIsNot(unchanged, embedding)
        self.assertIs(diagnostics["enabled"], False)


if __name__ == "__main__":
    unittest.main()
