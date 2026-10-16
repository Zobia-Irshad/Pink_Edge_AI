import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = APP_DIR / "Tools"
for directory in (str(APP_DIR), str(TOOLS_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import dicom_adapter
import dicom_pipeline
import pynetdicom.sop_class as sop_classes


class SyntheticDataset:
    def __init__(self, payload=b"synthetic-dicom", sop_instance_uid="1.2.826.0.1.3680043.10.543.1", modality="US"):
        self.payload = payload
        self.SOPInstanceUID = sop_instance_uid
        self.Modality = modality
        self.file_meta = None

    def save_as(self, path, write_like_original=False):
        Path(path).write_bytes(self.payload)


class TestDicomIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pynetdicom import AE

        sys.modules.pop("receiver", None)
        with (
            patch("os.makedirs"),
            patch.object(AE, "start_server") as start_server,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            cls.receiver = importlib.import_module("receiver")
        cls.listener_started_on_import = start_server.called
        cls.pipeline = dicom_pipeline

    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.previous_output_dir = self.receiver.OUTPUT_DIR
        self.receiver.OUTPUT_DIR = self.output_dir.name

    def tearDown(self):
        self.receiver.OUTPUT_DIR = self.previous_output_dir
        self.output_dir.cleanup()

    @staticmethod
    def valid_inference_result():
        return {
            "verdict": "synthetic result",
            "confidence": 90.0,
            "source": "synthetic test",
            "is_critical": False,
            "css": "success",
        }

    @staticmethod
    def process_result(status="analysis_completed"):
        result = {"status": status, "modality": "US", "image_size": [2, 2]}
        if status == "analysis_completed":
            result["inference"] = TestDicomIntegration.valid_inference_result()
        return result

    @staticmethod
    def event_for(dataset):
        return SimpleNamespace(dataset=dataset, file_meta=object())

    @staticmethod
    def dataset_for_sop_class(sop_class, modality):
        sop_class_uid = str(sop_class)
        return SimpleNamespace(
            Modality=modality,
            SOPClassUID=sop_class_uid,
            file_meta=SimpleNamespace(MediaStorageSOPClassUID=sop_class_uid),
        )

    def test_receiver_import_does_not_start_listener(self):
        self.assertFalse(self.listener_started_on_import)

    def test_invalid_sop_instance_uids_are_rejected(self):
        for uid in ("", "1.2.invalid", "1" * 65):
            with self.subTest(uid_length=len(uid)):
                with self.assertRaises(ValueError):
                    self.receiver._save_received_instance(SyntheticDataset(), uid)
        self.assertEqual(os.listdir(self.receiver.OUTPUT_DIR), [])

    def test_identical_duplicate_preserves_existing_scan(self):
        dataset = SyntheticDataset(payload=b"original synthetic bytes")
        path, is_new = self.receiver._save_received_instance(
            dataset, dataset.SOPInstanceUID
        )
        original_bytes = Path(path).read_bytes()

        duplicate_path, duplicate_is_new = self.receiver._save_received_instance(
            SyntheticDataset(payload=b"original synthetic bytes"),
            dataset.SOPInstanceUID,
        )

        self.assertTrue(is_new)
        self.assertEqual(path, duplicate_path)
        self.assertFalse(duplicate_is_new)
        self.assertEqual(Path(path).read_bytes(), original_bytes)

    def test_conflicting_duplicate_is_rejected_without_overwrite(self):
        dataset = SyntheticDataset(payload=b"original synthetic bytes")
        path, _ = self.receiver._save_received_instance(
            dataset, dataset.SOPInstanceUID
        )
        original_bytes = Path(path).read_bytes()

        with self.assertRaises(self.receiver.DuplicateSOPInstanceError):
            self.receiver._save_received_instance(
                SyntheticDataset(payload=b"different synthetic bytes"),
                dataset.SOPInstanceUID,
            )

        self.assertEqual(Path(path).read_bytes(), original_bytes)

    def test_handler_rejects_conflicting_duplicate_and_preserves_scan(self):
        original = SyntheticDataset(payload=b"original handler bytes")
        path, _ = self.receiver._save_received_instance(
            original, original.SOPInstanceUID
        )
        original_bytes = Path(path).read_bytes()
        conflict = SyntheticDataset(
            payload=b"conflicting handler bytes",
            sop_instance_uid=original.SOPInstanceUID,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            status = self.receiver.handle_store(self.event_for(conflict))

        self.assertEqual(status, 0xC211)
        self.assertEqual(Path(path).read_bytes(), original_bytes)
        self.assertFalse(Path(path).with_name(Path(path).stem + "_result.json").exists())

    def test_handler_storage_failure_returns_failure_without_file(self):
        dataset = SyntheticDataset(payload=b"storage failure bytes")
        with (
            patch.object(dataset, "save_as", side_effect=OSError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        self.assertEqual(status, 0xC211)
        self.assertEqual(list(Path(self.receiver.OUTPUT_DIR).iterdir()), [])

    def test_handler_inference_failure_acknowledges_stored_scan_and_sidecar(self):
        dataset = SyntheticDataset(payload=b"inference failure bytes")
        with (
            patch.object(self.receiver, "process_dicom", side_effect=RuntimeError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
        result_path = path.with_name(path.stem + "_result.json")
        self.assertEqual(status, 0x0000)
        self.assertEqual(path.read_bytes(), dataset.payload)
        self.assertEqual(json.loads(result_path.read_text())["status"], "inference_failed")

    def test_handler_result_write_failure_acknowledges_stored_scan(self):
        dataset = SyntheticDataset(payload=b"result write failure bytes")
        with (
            patch.object(self.receiver, "process_dicom", return_value=self.process_result()),
            patch.object(self.receiver, "save_result", side_effect=OSError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
        self.assertEqual(status, 0x0000)
        self.assertEqual(path.read_bytes(), dataset.payload)
        self.assertFalse(path.with_name(path.stem + "_result.json").exists())

    def test_all_supported_routes_call_predictors_offline(self):
        routes = (
            ("US", "predict_maternal"),
            ("MG", "predict_mammography"),
            ("MAMMOGRAPHY", "predict_mammography"),
            ("CR", "predict_tb"),
            ("DX", "predict_tb"),
        )
        image = object()
        for modality, predictor_name in routes:
            with self.subTest(modality=modality):
                with patch.object(
                    self.pipeline.inference,
                    predictor_name,
                    return_value=self.valid_inference_result(),
                ) as predictor:
                    result = self.pipeline.run_inference(image, modality)
                self.assertEqual(result["status"], "analysis_completed")
                predictor.assert_called_once_with(image, allow_network=False)

    def test_unsupported_modality_does_not_call_predictors(self):
        with (
            patch.object(self.pipeline.inference, "predict_maternal") as maternal,
            patch.object(self.pipeline.inference, "predict_mammography") as mammo,
            patch.object(self.pipeline.inference, "predict_tb") as tb,
        ):
            result = self.pipeline.run_inference(object(), "CT")
        self.assertEqual(result, {"status": "unsupported_modality"})
        maternal.assert_not_called()
        mammo.assert_not_called()
        tb.assert_not_called()

    def test_adapter_converts_generated_synthetic_dicom(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, UltrasoundImageStorage, generate_uid

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.dcm"
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = UltrasoundImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = FileDataset(
                str(path), {}, file_meta=file_meta, preamble=b"\0" * 128
            )
            dataset.SOPClassUID = UltrasoundImageStorage
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            dataset.Modality = "US"
            dataset.Rows = 2
            dataset.Columns = 2
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = 8
            dataset.BitsStored = 8
            dataset.HighBit = 7
            dataset.PixelRepresentation = 0
            dataset.PixelData = bytes((0, 20, 100, 255))
            dataset.save_as(path, enforce_file_format=True)

            image, converted_dataset = dicom_adapter.dicom_to_pil(path)

        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(converted_dataset.Modality, "US")

    def test_pipeline_reuses_adapter_and_returns_rgb(self):
        image = dicom_adapter.Image.new("L", (2, 2), color=10)
        dataset = SimpleNamespace(Modality="US")
        with (
            patch.object(self.pipeline.os.path, "isfile", return_value=True),
            patch.object(
                self.pipeline,
                "_adapter_dicom_to_pil",
                return_value=(image, dataset),
            ) as adapter,
        ):
            converted, returned_dataset = self.pipeline.dicom_to_pil("synthetic.dcm")
        adapter.assert_called_once_with("synthetic.dcm")
        self.assertEqual(converted.mode, "RGB")
        self.assertIs(returned_dataset, dataset)

    def test_process_dicom_routes_with_synthetic_dataset(self):
        image = dicom_adapter.Image.new("RGB", (2, 2))
        dataset = self.dataset_for_sop_class(
            sop_classes.DigitalMammographyXRayImageStorageForPresentation,
            "MG",
        )
        with (
            patch.object(self.pipeline, "dicom_to_pil", return_value=(image, dataset)),
            patch.object(
                self.pipeline.inference,
                "predict_mammography",
                return_value=self.valid_inference_result(),
            ) as predictor,
        ):
            result = self.pipeline.process_dicom("synthetic.dcm")
        self.assertEqual(result["status"], "analysis_completed")
        self.assertEqual(result["modality"], "MG")
        predictor.assert_called_once_with(image, allow_network=False)

    def test_supported_sop_classes_route_to_expected_predictors(self):
        routes = (
            ("UltrasoundImageStorage", "US", "predict_maternal"),
            ("DigitalMammographyXRayImageStorageForPresentation", "MG", "predict_mammography"),
            ("DigitalMammographyXRayImageStorageForProcessing", "MG", "predict_mammography"),
            ("ComputedRadiographyImageStorage", "CR", "predict_tb"),
            ("DigitalXRayImageStorageForPresentation", "DX", "predict_tb"),
            ("DigitalXRayImageStorageForProcessing", "DX", "predict_tb"),
        )
        image = dicom_adapter.Image.new("RGB", (2, 2))
        tested = 0
        for sop_class_name, modality, expected_predictor in routes:
            with self.subTest(sop_class=sop_class_name):
                sop_class = getattr(sop_classes, sop_class_name, None)
                if sop_class is None:
                    continue
                dataset = self.dataset_for_sop_class(sop_class, modality)
                predictors = {
                    name: patch.object(
                        self.pipeline.inference,
                        name,
                        return_value=self.valid_inference_result(),
                    )
                    for name in (
                        "predict_maternal",
                        "predict_mammography",
                        "predict_tb",
                    )
                }
                with patch.object(
                    self.pipeline,
                    "dicom_to_pil",
                    return_value=(image, dataset),
                ):
                    with contextlib.ExitStack() as stack:
                        mocks = {
                            name: stack.enter_context(patcher)
                            for name, patcher in predictors.items()
                        }
                        result = self.pipeline.process_dicom("synthetic.dcm")

                self.assertEqual(result["status"], "analysis_completed")
                mocks[expected_predictor].assert_called_once_with(
                    image, allow_network=False
                )
                for name, predictor in mocks.items():
                    if name != expected_predictor:
                        predictor.assert_not_called()
                tested += 1

        self.assertGreater(tested, 0)

    def test_sop_class_modality_mismatches_are_rejected_before_inference(self):
        routes = (
            ("UltrasoundImageStorage", "US", "MG"),
            ("DigitalMammographyXRayImageStorageForPresentation", "MG", "CR"),
            ("DigitalMammographyXRayImageStorageForProcessing", "MG", "CR"),
            ("ComputedRadiographyImageStorage", "CR", "DX"),
            ("DigitalXRayImageStorageForPresentation", "DX", "CR"),
            ("DigitalXRayImageStorageForProcessing", "DX", "CR"),
        )
        image = dicom_adapter.Image.new("RGB", (2, 2))
        tested = 0
        for sop_class_name, _, mismatched_modality in routes:
            with self.subTest(sop_class=sop_class_name):
                sop_class = getattr(sop_classes, sop_class_name, None)
                if sop_class is None:
                    continue
                dataset = self.dataset_for_sop_class(
                    sop_class, mismatched_modality
                )
                with (
                    patch.object(
                        self.pipeline,
                        "dicom_to_pil",
                        return_value=(image, dataset),
                    ),
                    patch.object(self.pipeline.inference, "predict_maternal") as maternal,
                    patch.object(self.pipeline.inference, "predict_mammography") as mammography,
                    patch.object(self.pipeline.inference, "predict_tb") as tb,
                ):
                    result = self.pipeline.process_dicom("synthetic.dcm")

                self.assertEqual(result["status"], "modality_mismatch")
                maternal.assert_not_called()
                mammography.assert_not_called()
                tb.assert_not_called()
                tested += 1

        self.assertGreater(tested, 0)

    def test_missing_malformed_and_conflicting_sop_class_identifiers_are_rejected(self):
        image = dicom_adapter.Image.new("RGB", (2, 2))
        ultrasound_uid = str(sop_classes.UltrasoundImageStorage)
        mammography_uid = str(
            sop_classes.DigitalMammographyXRayImageStorageForPresentation
        )
        datasets = (
            (SimpleNamespace(Modality="US"), "unsupported_sop_class"),
            (
                SimpleNamespace(Modality="US", SOPClassUID="not-a-sop-class-uid"),
                "unsupported_sop_class",
            ),
            (
                SimpleNamespace(
                    Modality="US",
                    SOPClassUID=ultrasound_uid,
                    file_meta=SimpleNamespace(
                        MediaStorageSOPClassUID=mammography_uid
                    ),
                ),
                "sop_class_mismatch",
            ),
            (
                self.dataset_for_sop_class(sop_classes.UltrasoundImageStorage, ""),
                "modality_mismatch",
            ),
        )

        for dataset, expected_status in datasets:
            with self.subTest(expected_status=expected_status):
                with (
                    patch.object(
                        self.pipeline,
                        "dicom_to_pil",
                        return_value=(image, dataset),
                    ),
                    patch.object(self.pipeline.inference, "predict_maternal") as maternal,
                    patch.object(self.pipeline.inference, "predict_mammography") as mammography,
                    patch.object(self.pipeline.inference, "predict_tb") as tb,
                ):
                    result = self.pipeline.process_dicom("synthetic.dcm")

                self.assertEqual(result["status"], expected_status)
                maternal.assert_not_called()
                mammography.assert_not_called()
                tb.assert_not_called()

    def test_file_meta_sop_class_uid_is_used_when_dataset_uid_is_missing(self):
        sop_class_uid = str(sop_classes.UltrasoundImageStorage)
        dataset = SimpleNamespace(
            Modality="US",
            file_meta=SimpleNamespace(MediaStorageSOPClassUID=sop_class_uid),
        )
        image = dicom_adapter.Image.new("RGB", (2, 2))
        with (
            patch.object(
                self.pipeline,
                "dicom_to_pil",
                return_value=(image, dataset),
            ),
            patch.object(
                self.pipeline.inference,
                "predict_maternal",
                return_value=self.valid_inference_result(),
            ) as predictor,
        ):
            result = self.pipeline.process_dicom("synthetic.dcm")

        self.assertEqual(result["status"], "analysis_completed")
        predictor.assert_called_once_with(image, allow_network=False)

    def test_receiver_rejects_missing_or_malformed_sop_instance_uid(self):
        for uid in ("", "not-a-uid"):
            with self.subTest(uid_length=len(uid)):
                dataset = SyntheticDataset(sop_instance_uid=uid)
                with contextlib.redirect_stdout(io.StringIO()):
                    status = self.receiver.handle_store(self.event_for(dataset))
                self.assertEqual(status, 0xC211)
                self.assertEqual(list(Path(self.receiver.OUTPUT_DIR).iterdir()), [])

    def test_result_save_success_and_failure_cleanup(self):
        result = self.process_result()
        with tempfile.TemporaryDirectory() as directory:
            dicom_path = Path(directory) / "synthetic.dcm"
            saved_path = Path(self.pipeline.save_result(dicom_path, result))
            self.assertEqual(json.loads(saved_path.read_text()), result)

            with patch.object(self.pipeline.os, "replace", side_effect=OSError):
                with self.assertRaises(OSError):
                    self.pipeline.save_result(dicom_path, result)
            self.assertEqual(list(Path(directory).glob(".result-*.tmp")), [])

    def test_duplicate_without_valid_result_retries_and_preserves_scan(self):
        for suffix, sidecar_state in enumerate(("missing", "corrupt", "incomplete"), 1):
            with self.subTest(sidecar_state=sidecar_state):
                uid = f"1.2.826.0.1.3680043.10.543.{suffix}"
                dataset = SyntheticDataset(payload=b"stored synthetic bytes", sop_instance_uid=uid)
                path, _ = self.receiver._save_received_instance(dataset, uid)
                original_bytes = Path(path).read_bytes()
                sidecar = Path(path).with_name(Path(path).stem + "_result.json")
                if sidecar_state == "corrupt":
                    sidecar.write_text("not json")
                elif sidecar_state == "incomplete":
                    sidecar.write_text(json.dumps({"status": "analysis_completed"}))

                with patch.object(
                    self.receiver,
                    "process_dicom",
                    return_value=self.process_result(),
                ) as processor, contextlib.redirect_stdout(io.StringIO()):
                    status = self.receiver.handle_store(self.event_for(dataset))

                self.assertEqual(status, 0x0000)
                processor.assert_called_once_with(path)
                self.assertEqual(Path(path).read_bytes(), original_bytes)
                self.assertTrue(self.receiver._has_completed_analysis(path))

    def test_completed_duplicate_skips_inference(self):
        dataset = SyntheticDataset(payload=b"completed synthetic bytes")
        path, _ = self.receiver._save_received_instance(
            dataset, dataset.SOPInstanceUID
        )
        self.receiver.save_result(path, self.process_result())

        with patch.object(self.receiver, "process_dicom") as processor, contextlib.redirect_stdout(io.StringIO()):
            status = self.receiver.handle_store(self.event_for(dataset))

        self.assertEqual(status, 0x0000)
        processor.assert_not_called()

    def test_concurrent_identical_duplicate_runs_inference_once(self):
        uid = "1.2.826.0.1.3680043.10.543.101"
        dataset = SyntheticDataset(payload=b"concurrent synthetic bytes", sop_instance_uid=uid)
        first_inference_started = threading.Event()
        allow_first_to_finish = threading.Event()
        second_lock_blocked = threading.Event()
        inference_count = 0
        inference_count_guard = threading.Lock()

        class ObservedLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.acquire_count = 0
                self.count_guard = threading.Lock()

            def acquire(self):
                with self.count_guard:
                    self.acquire_count += 1
                    acquire_number = self.acquire_count
                if acquire_number == 2:
                    if not self.lock.acquire(blocking=False):
                        second_lock_blocked.set()
                        self.lock.acquire()
                        return True
                    return True
                return self.lock.acquire()

            def release(self):
                self.lock.release()

        self.receiver._SOP_INSTANCE_LOCKS[uid] = {
            "lock": ObservedLock(),
            "users": 0,
        }

        def process_once_at_a_time(path):
            nonlocal inference_count
            with inference_count_guard:
                inference_count += 1
                call_number = inference_count
            if call_number == 1:
                first_inference_started.set()
                if not allow_first_to_finish.wait(timeout=5):
                    raise TimeoutError("test did not release first inference")
            return self.process_result()

        statuses = []
        errors = []

        def receive():
            try:
                statuses.append(self.receiver.handle_store(self.event_for(dataset)))
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=receive)
        second = threading.Thread(target=receive)
        try:
            with (
                patch.object(self.receiver, "process_dicom", side_effect=process_once_at_a_time),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                first.start()
                self.assertTrue(first_inference_started.wait(timeout=5))
                second.start()
                self.assertTrue(second_lock_blocked.wait(timeout=5))
                with inference_count_guard:
                    self.assertEqual(inference_count, 1)
        finally:
            allow_first_to_finish.set()
            if first.ident is not None:
                first.join(timeout=5)
            if second.ident is not None:
                second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(statuses, [0x0000, 0x0000])
        self.assertEqual(inference_count, 1)
        self.assertNotIn(uid, self.receiver._SOP_INSTANCE_LOCKS)
        stored_path = Path(self.receiver.OUTPUT_DIR) / f"{uid}.dcm"
        self.assertEqual(stored_path.read_bytes(), dataset.payload)

    def test_distinct_uid_receipts_can_infer_concurrently(self):
        datasets = (
            SyntheticDataset(
                payload=b"first independent bytes",
                sop_instance_uid="1.2.826.0.1.3680043.10.543.102",
            ),
            SyntheticDataset(
                payload=b"second independent bytes",
                sop_instance_uid="1.2.826.0.1.3680043.10.543.103",
            ),
        )
        inference_barrier = threading.Barrier(2)
        statuses = []
        errors = []

        def process_concurrently(path):
            inference_barrier.wait(timeout=5)
            return self.process_result()

        def receive(dataset):
            try:
                statuses.append(self.receiver.handle_store(self.event_for(dataset)))
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=receive, args=(dataset,)) for dataset in datasets]
        try:
            with (
                patch.object(self.receiver, "process_dicom", side_effect=process_concurrently),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=7)
        finally:
            inference_barrier.abort()
            for worker in workers:
                if worker.ident is not None and worker.is_alive():
                    worker.join(timeout=1)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(errors, [])
        self.assertEqual(statuses, [0x0000, 0x0000])
        for dataset in datasets:
            path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
            result_path = path.with_name(path.stem + "_result.json")
            self.assertEqual(path.read_bytes(), dataset.payload)
            self.assertEqual(json.loads(result_path.read_text())["status"], "analysis_completed")
            self.assertNotIn(dataset.SOPInstanceUID, self.receiver._SOP_INSTANCE_LOCKS)

    def test_lock_registry_is_cleaned_after_inference_failure(self):
        dataset = SyntheticDataset(
            payload=b"synthetic lock cleanup failure",
            sop_instance_uid="1.2.826.0.1.3680043.10.543.104",
        )
        with (
            patch.object(self.receiver, "process_dicom", side_effect=RuntimeError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        self.assertEqual(status, 0x0000)
        self.assertNotIn(dataset.SOPInstanceUID, self.receiver._SOP_INSTANCE_LOCKS)
        stored_path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
        self.assertEqual(stored_path.read_bytes(), dataset.payload)

    def test_inference_exception_is_saved_as_failure_after_receipt(self):
        dataset = SyntheticDataset(payload=b"synthetic inference failure")
        with (
            patch.object(self.receiver, "process_dicom", side_effect=RuntimeError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
        result_path = path.with_name(path.stem + "_result.json")
        self.assertEqual(status, 0x0000)
        self.assertTrue(path.exists())
        self.assertEqual(json.loads(result_path.read_text())["status"], "inference_failed")
        self.assertNotIn("synthetic", result_path.read_text())

    def test_result_write_failure_still_acknowledges_durable_receipt(self):
        dataset = SyntheticDataset(payload=b"synthetic result write failure")
        with (
            patch.object(self.receiver, "process_dicom", return_value=self.process_result()),
            patch.object(self.receiver, "save_result", side_effect=OSError("synthetic")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = self.receiver.handle_store(self.event_for(dataset))

        path = Path(self.receiver.OUTPUT_DIR) / f"{dataset.SOPInstanceUID}.dcm"
        self.assertEqual(status, 0x0000)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
