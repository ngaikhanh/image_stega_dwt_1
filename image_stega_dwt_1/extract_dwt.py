import pywt
import cv2
import numpy as np

def bin_to_text(binary_str):
    binary_str = binary_str[:len(binary_str) - (len(binary_str) % 8)]
    byte_data = bytearray()

    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        byte_data.append(int(byte, 2))

    try:
        return byte_data.decode('utf-8')
    except UnicodeDecodeError as e:
        print("❌ UnicodeDecodeError:", e)
        return "Failed to decode text"

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh.")

    b, _, _ = cv2.split(image)
    coeffs = pywt.dwt2(b, 'haar')
    _, (cH, _, _) = coeffs

    flat_cH = cH.flatten().astype(np.int32)
    binary_data = ''
    for val in flat_cH:
        binary_data += str(val & 1)
        if binary_data.endswith('1111111111111110'):
            break

    if '1111111111111110' not in binary_data:
        print("❗ Không tìm thấy cờ kết thúc. Có thể ảnh không chứa dữ liệu.")
        return

    binary_data = binary_data[:-16]  # Bỏ cờ kết thúc
    text = bin_to_text(binary_data)
    print("📥 Văn bản trích xuất:")
    print(text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("❗ Cách dùng: python extract_dwt.py <ảnh chứa dữ liệu>")
    else:
        extract_text_from_image(sys.argv[1])
