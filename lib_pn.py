import cv2
import sys

def stitch_images(image_paths, output_filename):
    """
    여러 개의 이미지를 읽어 하나의 파노라마 이미지로 만듭니다.

    :param image_paths: (list) 이미지 파일 경로 리스트
    :param output_filename: (str) 저장할 파노라마 파일 이름
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"오류: '{path}' 파일을 읽을 수 없습니다.")
            return

        print(f"'{path}' 로드 성공. 크기: {img.shape}")
        images.append(img)

    if len(images) < 2:
        print("오류: 파노라마를 만들려면 2개 이상의 이미지가 필요합니다.")
        return

    print("이미지 스티칭을 시작합니다...")

    # 1. Stitcher 객체 생성
    # (참고: OpenCV 버전에 따라 cv2.Stitcher_create() 또는 cv2.createStitcher())
    try:
        stitcher = cv2.Stitcher_create()
    except AttributeError:
        # 이전 OpenCV 버전을 위한 예외 처리
        stitcher = cv2.createStitcher()

    # 2. 이미지 스티칭 시도
    # stitch() 메서드는 (상태 코드, 파노라마 이미지) 튜플을 반환
    (status, pano) = stitcher.stitch(images)

    # 3. 결과 확인
    if status == cv2.Stitcher_OK:
        print("파노라마 생성 성공!")
        
        # 결과 이미지 저장
        cv2.imwrite(output_filename, pano)
        print(f"'{output_filename}'으로 저장되었습니다.")

        # # 결과 이미지를 화면에 표시 (선택 사항)
        # # 너무 클 수 있으므로 크기를 조절해서 보여줍니다.
        # h, w = pano.shape[:2]
        # scale = 800 / w  # 너비를 800px 기준으로 조절
        # pano_resized = cv2.resize(pano, (int(w * scale), int(h * scale)))
        
        # cv2.imshow('Panorama', pano_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("오류: 특징점을 찾지 못했습니다. 이미지가 충분히 겹치는지 확인하세요.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("오류: 호모그래피 추정에 실패했습니다. 이미지가 충분히 겹치지 않거나 너무 다를 수 있습니다.")
    else:
        print(f"파노라마 생성 실패. 상태 코드: {status}")

# --- 코드 실행 ---
if __name__ == "__main__":
    # 파노라마로 만들 이미지 파일 리스트
    # (파일 이름을 실제 파일명으로 변경하세요
    my_image_files = [f'./rio/testimg{n+1}.png' for n in range(10)] 
    
    # 저장될 결과 파일 이름
    result_file = 'panorama_result_10.jpg'

    stitch_images(my_image_files, result_file)