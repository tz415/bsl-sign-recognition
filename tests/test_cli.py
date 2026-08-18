import argparse
import unittest

from bsl_sign_recognition.cli import build_parser, confidence, positive_int

class ArgumentTypeTests(unittest.TestCase):
    def test_positive_int_accepts_positive_value(self) -> None:
        self.assertEqual(positive_int("30"), 30)

    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_confidence_accepts_boundaries(self) -> None:
        self.assertEqual(confidence("0"), 0.0)
        self.assertEqual(confidence("1"), 1.0)

    def test_confidence_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            confidence("1.1")

    def test_parser_supports_second_unmirrored_camera(self) -> None:
        args = build_parser().parse_args(["--camera", "1", "--no-mirror"])
        self.assertEqual(args.camera, 1)
        self.assertTrue(args.no_mirror)


if __name__ == "__main__":
    unittest.main()
