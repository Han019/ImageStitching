import cv2
import numpy as np

def gray_img(path):
    """이미지를 읽어 그레이스케일로 변환합니다."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")
    return img

def im_show(img, window_name='Image'):
    """이미지를 화면에 표시합니다."""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1. 스티칭된 파노라마 이미지 불러오기
img_path = "./panorama_aaa.jpg"
print(f"'{img_path}' 파일을 불러옵니다...")
original_img = gray_img(img_path)

# 2. 블렌딩에서 검은색 배경 영역을 제외하기 위한 마스크 생성
# 이미지에서 픽셀 값이 0이 아닌 모든 영역을 찾습니다.
print("블렌딩을 위한 마스크를 생성합니다...")
_, mask = cv2.threshold(original_img, 1, 255, cv2.THRESH_BINARY)

# 3. 양방향 필터(Bilateral Filter)를 이용한 톤 블렌딩
# 경계는 보존하면서 노이즈와 톤의 차이를 부드럽게 만들어 줍니다.
# 파라미터 값(d, sigmaColor, sigmaSpace)을 조절하여 블렌딩 강도를 변경할 수 있습니다.
print("양방향 필터를 적용하여 이미지를 블렌딩합니다...")
blended_img = cv2.bilateralFilter(original_img, d=15, sigmaColor=80, sigmaSpace=80)

# 4. 마스크를 사용하여 이미지 영역에만 블렌딩 결과 적용
# 필터링 과정에서 검은 배경에 생길 수 있는 노이즈를 제거하고 원본 배경을 유지합니다.
print("블렌딩 결과를 원본 이미지 영역에만 적용합니다...")
final_img = cv2.bitwise_and(blended_img, blended_img, mask=mask)

# 5. 결과 비교 및 저장
# 원본 이미지와 블렌딩 처리된 이미지를 차례로 보여줍니다.
print("처리가 완료되었습니다. 원본과 결과 이미지를 순서대로 표시합니다.")
im_show(original_img, window_name='Original Panorama')
im_show(final_img, window_name='Blended Panorama')

# 처리된 이미지를 파일로 저장
output_path = "panorama_blended.jpg"
cv2.imwrite(output_path, final_img)
print(f"블렌딩된 결과가 '{output_path}' 파일로 저장되었습니다.")
