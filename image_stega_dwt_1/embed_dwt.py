import pywt
import cv2
import numpy as np

def text_to_bin(text):
    return ''.join([format(b, '08b') for b in text.encode('utf-8')])

def embed_text_in_image(image_path, text, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Không thể đọc ảnh đầu vào.")
    
    b, g, r = cv2.split(image)
    coeffs = pywt.dwt2(b, 'haar')
    cA, (cH, cV, cD) = coeffs

    binary_data = text_to_bin(text) + '1111111111111110'  # End flag
    print("🔢 Độ dài dữ liệu nhị phân:", len(binary_data))

    flat_cH = cH.flatten()
    if len(binary_data) > len(flat_cH):
        raise ValueError("Văn bản quá dài để giấu trong ảnh.")

    for i in range(len(binary_data)):
        val = int(round(flat_cH[i]))
        if binary_data[i] == '1':
            val |= 1
        else:
            val &= ~1
        flat_cH[i] = val

    cH_mod = flat_cH.reshape(cH.shape)
    coeffs_mod = (cA, (cH_mod, cV, cD))
    b_stego = pywt.idwt2(coeffs_mod, 'haar')
    b_stego = np.clip(b_stego, 0, 255).astype(np.uint8)

    image_stego = cv2.merge((b_stego, g, r))
    cv2.imwrite(output_path, image_stego)
    print("✅ Văn bản đã được giấu vào ảnh:", output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("❗ Cách dùng: python embed_dwt.py <ảnh gốc> <văn bản> <ảnh đầu ra>")
    else:
        embed_text_in_image(sys.argv[1], sys.argv[2], sys.argv[3])
