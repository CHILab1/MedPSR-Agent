"""
utils/test_subject_filter.py
────────────────────────────
Unit tests for subject_id filtering in SimilarityEngine and ContextConstructor.
"""

import sys
import os
import unittest
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import Settings
from agents.similarity_engine import SimilarityEngine
from agents.context_constructor import ContextConstructor


class DummyDataGatherer:
    """Mock DataGatherer for testing context construction."""
    def __init__(self, hadm_to_subject):
        self._hadm_to_subject = hadm_to_subject
        self._hadm_to_idx = {hid: i for i, hid in enumerate(hadm_to_subject.keys())}

    @property
    def hadm_to_subject(self):
        return self._hadm_to_subject

    def get_subject_id(self, hadm_id):
        return self._hadm_to_subject.get(hadm_id)

    def get_patient_data(self, hadm_id, attributes_list=None, is_target=True):
        return {
            "rank": 0 if is_target else 1,
            "hadm_id": hadm_id,
            "subject_id": self._hadm_to_subject.get(hadm_id),
            "diagnoses": ["I10", "E11"],
        }

    def get_patients_data_batch(self, hadm_ids, attributes_list=None):
        results = []
        for i, hid in enumerate(hadm_ids, 1):
            data = self.get_patient_data(hid, attributes_list, is_target=False)
            data["rank"] = i
            results.append(data)
        return results

    def build_diagnosis_glossary(self, codes):
        return {c: f"Description of {c}" for c in codes}


class TestSubjectFiltering(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.engine = SimilarityEngine(self.settings)

        # Setup 6 admissions across 4 subjects
        # hadm_id 101, 102: subject 1000 (Target patient and target's readmission)
        # hadm_id 201, 202: subject 2000 (Peer patient with 2 admissions)
        # hadm_id 301:      subject 3000 (Peer patient)
        # hadm_id 401:      subject 4000 (Peer patient)
        self.admissions = [101, 102, 201, 202, 301, 401]
        self.subjects   = [1000, 1000, 2000, 2000, 3000, 4000]

        self.hadm_to_idx = {h: i for i, h in enumerate(self.admissions)}
        self.hadm_to_subject = dict(zip(self.admissions, self.subjects))

        # Create synthetic normalized embeddings (dim=8)
        # target (idx 0):
        # 102 (idx 1) is very close to target (same patient profile)
        # 201 (idx 2) is close
        # 202 (idx 3) is close
        # 301 (idx 4) is medium
        # 401 (idx 5) is further
        n = len(self.admissions)
        dim = 8
        np.random.seed(42)

        emb_diag = np.random.randn(n, dim).astype(np.float32)
        # Make idx 1 (hadm 102) almost identical to idx 0 (hadm 101)
        emb_diag[1] = emb_diag[0] + 0.01 * np.random.randn(dim)
        # Normalize
        emb_diag = emb_diag / np.linalg.norm(emb_diag, axis=1, keepdims=True)

        emb_num = np.random.randn(n, dim).astype(np.float32)
        emb_num[1] = emb_num[0] + 0.01 * np.random.randn(dim)
        emb_num = emb_num / np.linalg.norm(emb_num, axis=1, keepdims=True)

        emb_cat = np.random.randn(n, dim).astype(np.float32)
        emb_cat[1] = emb_cat[0]
        emb_cat = emb_cat / np.linalg.norm(emb_cat, axis=1, keepdims=True)

        self.engine._embeddings_diagnosis = emb_diag
        self.engine._embeddings_numeric = emb_num
        self.engine._embeddings_categorical = emb_cat
        self.engine._hadm_to_idx = self.hadm_to_idx
        self.engine._idx_to_hadm = {v: k for k, v in self.hadm_to_idx.items()}
        self.engine._hadm_to_subject = self.hadm_to_subject
        self.engine._idx_to_subject = {
            idx: self.hadm_to_subject[hid] for hid, idx in self.hadm_to_idx.items()
        }
        self.engine.use_milvus = False  # Test via local cosine fallback
        self.engine._loaded = True

    def test_exclude_same_subject_as_target(self):
        """Verify that hadm_id with the same subject_id as target is strictly excluded."""
        results = self.engine.find_similar(
            target_hadm_id=101,
            k=5,
            exclude_same_subject=True,
        )
        returned_hadms = [hid for hid, score in results]

        # 101 is target itself, must not be present
        self.assertNotIn(101, returned_hadms)
        # 102 belongs to subject 1000, must NOT be present
        self.assertNotIn(102, returned_hadms, "Admissions with same subject_id as target must be excluded!")

        # All returned hadm_ids must have different subject_ids
        for hid in returned_hadms:
            self.assertNotEqual(
                self.hadm_to_subject[hid],
                1000,
                f"hadm_id {hid} has subject_id 1000 which is the target's subject_id!"
            )

    def test_deduplicate_peer_subjects(self):
        """Verify that multiple admissions of another peer patient are deduplicated."""
        results = self.engine.find_similar(
            target_hadm_id=101,
            k=5,
            exclude_same_subject=True,
        )
        returned_hadms = [hid for hid, score in results]
        returned_subjects = [self.hadm_to_subject[hid] for hid in returned_hadms]

        # No duplicate subject_ids in returned peers
        self.assertEqual(len(returned_subjects), len(set(returned_subjects)))

        # 201 and 202 belong to subject 2000, so only one of them should be returned
        has_201 = 201 in returned_hadms
        has_202 = 202 in returned_hadms
        self.assertTrue(has_201 ^ has_202 or (not has_201 and not has_202), "Only one admission per peer subject should be returned")

    def test_include_same_subject_when_flag_false(self):
        """Verify that when exclude_same_subject=False, same subject admissions are retained."""
        results = self.engine.find_similar(
            target_hadm_id=101,
            k=5,
            exclude_same_subject=False,
        )
        returned_hadms = [hid for hid, score in results]

        # 101 itself is target, so excluded by index
        self.assertNotIn(101, returned_hadms)
        # 102 was crafted to be almost identical, so it should be retrieved
        self.assertIn(102, returned_hadms, "With exclude_same_subject=False, 102 should be allowed")

    def test_context_constructor_integration(self):
        """Verify ContextConstructor correctly propagates and safeguards subject filtering."""
        gatherer = DummyDataGatherer(self.hadm_to_subject)
        constructor = ContextConstructor(self.settings)

        routing = {
            "patient_id": 101,
            "similarity": True,
            "k_similar": 3,
            "exclude_same_subject": True,
        }

        ctx = constructor.build_context(
            routing=routing,
            data_gatherer=gatherer,
            similarity_engine=self.engine,
        )

        target = ctx["target_patient"]
        self.assertEqual(target["hadm_id"], 101)
        self.assertEqual(target["subject_id"], 1000)

        similar = ctx["similar_patients"]
        for pat in similar:
            self.assertNotEqual(
                pat["subject_id"],
                1000,
                f"Similar patient {pat['hadm_id']} has the same subject_id as target!"
            )


if __name__ == "__main__":
    unittest.main()
