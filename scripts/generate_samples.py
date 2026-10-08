"""Create deterministic RGB and auxiliary-modality samples for the demo."""

from pathlib import Path

import cv2
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "static" / "assets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    height, width = 540, 840
    image = np.full((height, width, 3), (242, 244, 247), dtype=np.uint8)

    cv2.rectangle(image, (0, 330), (width, height), (210, 221, 207), -1)
    cv2.rectangle(image, (65, 70), (775, 360), (226, 229, 233), -1)
    cv2.rectangle(image, (105, 110), (735, 345), (250, 250, 247), -1)
    cv2.circle(image, (290, 260), 105, (74, 129, 203), -1, cv2.LINE_AA)
    cv2.circle(image, (290, 260), 72, (96, 156, 231), -1, cv2.LINE_AA)
    cv2.rectangle(image, (445, 155), (630, 330), (65, 169, 127), -1)
    cv2.rectangle(image, (475, 185), (600, 330), (90, 196, 150), -1)
    cv2.ellipse(image, (540, 155), (80, 35), 0, 180, 360, (223, 154, 56), -1, cv2.LINE_AA)
    cv2.line(image, (80, 360), (760, 360), (127, 138, 150), 4, cv2.LINE_AA)
    cv2.putText(image, "RGB", (120, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (45, 55, 72), 2, cv2.LINE_AA)

    auxiliary = np.full((height, width), 35, dtype=np.uint8)
    cv2.rectangle(auxiliary, (105, 110), (735, 345), 92, -1)
    cv2.circle(auxiliary, (290, 260), 107, 220, -1, cv2.LINE_AA)
    cv2.rectangle(auxiliary, (445, 155), (630, 330), 180, -1)
    cv2.ellipse(auxiliary, (540, 155), (82, 37), 0, 180, 360, 205, -1, cv2.LINE_AA)
    auxiliary = cv2.GaussianBlur(auxiliary, (11, 11), 0)

    cv2.imwrite(str(OUT / "sample_rgb.png"), image)
    cv2.imwrite(str(OUT / "sample_aux.png"), auxiliary)

    # An illustrative geometric phantom, not a patient image or clinical simulation.
    rng = np.random.default_rng(20261008)
    phantom = np.full((height, width), 14, np.float32)
    body = np.zeros((height, width), np.uint8)
    cv2.ellipse(body, (420, 270), (265, 220), 0, 0, 360, 1, -1)
    phantom[body > 0] = 82
    cv2.ellipse(phantom, (335, 255), (75, 125), -15, 0, 360, 125, -1)
    cv2.ellipse(phantom, (515, 255), (75, 125), 15, 0, 360, 128, -1)
    cv2.ellipse(phantom, (420, 390), (40, 30), 0, 0, 360, 220, -1)
    cv2.ellipse(phantom, (350, 235), (38, 50), 20, 0, 360, 185, -1)
    phantom = np.clip(phantom + rng.normal(0, 3, phantom.shape), 0, 255).astype(np.uint8)
    auxiliary = cv2.GaussianBlur(phantom, (15, 15), 0)
    cv2.imwrite(str(OUT / "phantom_rgb.png"), cv2.cvtColor(phantom, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(OUT / "phantom_aux.png"), auxiliary)


if __name__ == "__main__":
    main()
