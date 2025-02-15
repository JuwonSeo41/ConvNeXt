import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval2d


def apply_chebyshev_transform(image, order):
    """
    Apply Chebyshev polynomial transformation to an image channel.
    """
    h, w = image.shape
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)

    coeffs = np.zeros((order + 1, order + 1))
    coeffs[order, order] = 1  # 특정 차수의 Chebyshev 다항식 적용
    transformed = chebval2d(X, Y, coeffs)

    # Normalize to [0, 255] and apply to image
    transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min()) * 255
    transformed = transformed.astype(np.uint8)

    return cv2.addWeighted(image, 0.5, transformed, 0.5, 0)  # Blend with original


def generate_augmented_images(image):
    augmented_images = []

    # 1️⃣ 개별 채널 변형 (각 채널별로 독립적으로 변형)
    for order in range(2, 6):  # Orders 2 to 5 (4개)
        for channel in range(3):  # R, G, B (3개)
            modified = image.copy()
            modified[:, :, channel] = apply_chebyshev_transform(modified[:, :, channel], order)
            augmented_images.append(modified)

    # 2️⃣ 전체 채널 동시 변형 (RGB 모두 변형)
    for order in range(2, 6):  # Orders 2 to 5 (4개)
        modified = image.copy()
        for channel in range(3):  # R, G, B 채널 동시에 적용
            modified[:, :, channel] = apply_chebyshev_transform(modified[:, :, channel], order)
        augmented_images.append(modified)

    for order in range(2, 6):  # Orders 2 to 5 (4개)
        for channel_pair in [(0, 1), (1, 2), (0, 2)]:  # (R, G), (G, B), (R, B)
            modified = image.copy()
            # Apply Chebyshev transform on the selected pair of channels
            modified[:, :, channel_pair[0]] = apply_chebyshev_transform(modified[:, :, channel_pair[0]], order)
            modified[:, :, channel_pair[1]] = apply_chebyshev_transform(modified[:, :, channel_pair[1]], order)
            augmented_images.append(modified)

    return augmented_images


# # Example usage
# image_path = "C:/AP_C_14.jpg"  # Change to your image file
# augmented_images = generate_augmented_images(image_path)
#
# # Display some augmented images
# fig, axes = plt.subplots(4, 7, figsize=(12, 9))  # 16개 이미지 배치
# for i, ax in enumerate(axes.flat):
#     ax.imshow(augmented_images[i])
#     ax.axis("off")
# plt.show()
