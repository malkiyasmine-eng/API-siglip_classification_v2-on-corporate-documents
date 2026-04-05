# test_api.py
# ─────────────────────────────────────────────────────────────────────────────
# Test the running SigLIP classifier API
# Run AFTER starting the server with: uvicorn app:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import sys
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Check that the API is alive and the model is loaded."""
    print("─" * 50)
    print("1️⃣  GET /health")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"   Status code : {resp.status_code}")
    print(f"   Response    : {resp.json()}")
    assert resp.status_code == 200, "Health check failed!"
    assert resp.json()["status"] == "ok", "Model not loaded!"
    print("   ✅ Health check passed.\n")


def test_classify(image_path: str):
    """Send an image to POST /classify and print the result."""
    print("─" * 50)
    print(f"2️⃣  POST /classify  ←  {image_path}")
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/classify",
            files={"file": (image_path, f, "image/jpeg")},
        )
    print(f"   Status code : {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   🏆 Label      : {data['label']}")
        print(f"   📊 Confidence : {data['confidence']:.4f}  ({data['confidence']*100:.1f}%)")
        print("   📋 Top-3 predictions:")
        for i, pred in enumerate(data["top3"], 1):
            bar = "█" * int(pred["confidence"] * 30)
            print(f"      {i}. {pred['label']:<30} {pred['confidence']:.4f}  {bar}")
    else:
        print(f"   ❌ Error: {resp.json()}")
    print()


def test_invalid_file():
    """Send a non-image file — should return 422."""
    print("─" * 50)
    print("3️⃣  POST /classify  ←  invalid file (should return 422)")
    fake_bytes = b"this is not an image at all"
    resp = requests.post(
        f"{BASE_URL}/classify",
        files={"file": ("fake.jpg", fake_bytes, "image/jpeg")},
    )
    print(f"   Status code : {resp.status_code}  (expected 422)")
    print(f"   Response    : {resp.json()}")
    assert resp.status_code == 422, "Expected 422 for invalid image!"
    print("   ✅ Error handling works correctly.\n")


if __name__ == "__main__":
    # Usage: python test_api.py path/to/your/image.jpg
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image>")
        print("Example: python test_api.py my_document.jpg\n")
        image_path = None
    else:
        image_path = sys.argv[1]

    test_health()

    if image_path:
        test_classify(image_path)
    else:
        print("⚠️  No image path provided — skipping classify test.")
        print("   Run: python test_api.py path/to/your/document.jpg\n")

    test_invalid_file()

    print("─" * 50)
    print("🎉 All tests passed!")
