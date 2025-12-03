import numpy as np
import cv2
import os
from tdm import tqdm

def gray_img(img_path):
    """
    1-1 단계: 이미지를 그레이스케일로 변환
    """
    img_matrix = cv2.imread(img_path)
    if img_matrix is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {img_path}")
    
    img_matrix = img_matrix.astype(np.float32)
    
    # 흑백 만들기 (RGB to Grayscale)
    gray_weight = np.array([0.114, 0.587, 0.299])
    gray_img = img_matrix @ gray_weight
    gray_matrix = gray_img.astype(np.uint8)
    
    return gray_matrix


def gaussian_kernel(kernel_size, sigma):
    """
    1-1 단계: 가우시안 커널 생성
    """
    x = np.arange(-(kernel_size) // 2, (kernel_size) // 2 + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_smoothing(img, sigma=1.0):
    """
    1-1 단계: 가우시안 필터를 적용해 스무딩
    """
    kernel_size = 7
    kernel = gaussian_kernel(kernel_size, sigma)
    r, c = img.shape
    
    img = img.astype(np.float32)
    img_smooth_x = np.zeros_like(img, dtype=np.float32)
    
    # 가로 방향 컨볼루션
    for y in range(r):
        img_smooth_x[y, :] = np.convolve(img[y, :], kernel, mode='same')
    
    img_smooth_final = np.zeros_like(img, dtype=np.float32)
    
    # 세로 방향 컨볼루션
    for x in range(c):
        img_smooth_final[:, x] = np.convolve(img_smooth_x[:, x], kernel, mode='same')
    
    return img_smooth_final


def derivative_filter(img):
    """
    1-1 단계: 미분 필터 적용 (Ix, Iy 계산)
    """
    img = img.astype(np.float32)
    r, c = img.shape
    
    Ix = np.zeros_like(img, dtype=np.float32)
    Iy = np.zeros_like(img, dtype=np.float32)
    
    # 미분 커널
    kernel = np.array([1, 0, -1])
    
    # X 방향 미분 (수평 방향)
    for y in range(r):
        Ix[y, :] = np.convolve(img[y, :], kernel, mode='same')
    
    # Y 방향 미분 (수직 방향)
    for x in range(c):
        Iy[:, x] = np.convolve(img[:, x], kernel, mode='same')
    
    return Ix, Iy


def harris_corner_detection(img):
    """
    1-2 단계: Harris Corner Detection을 이용한 코너 포인트 찾기
    """
    # 스무딩 적용
    img_smooth = gaussian_smoothing(img, sigma=1.3)
    img_smooth = img_smooth.astype(np.float32)
    
    # 미분 필터 적용
    Ix, Iy = derivative_filter(img_smooth)
    
    r, c = img.shape
    
    # 제품 계산
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2
    
    # 구조 텐서 요소들에 가우시안 가중치 적용
    Mxx = gaussian_smoothing(Ixx, sigma=1.0)
    Mxy = gaussian_smoothing(Ixy, sigma=1.0)
    Myy = gaussian_smoothing(Iyy, sigma=1.0)
    
    # Harris 응답 함수 계산
    detM = Mxx * Myy - Mxy * Mxy
    trM = Mxx + Myy
    
    k = 0.05
    R_score = detM - k * (trM ** 2)
    
    # 코너 임계값
    threshold = 10000
    
    corners = []
    window_size = 5
    w = window_size // 2
    
    # 로컬 최대값만 코너로 선택
    for y in range(w, r - w):
        for x in range(w, c - w):
            val = R_score[y, x]
            if val > threshold:
                # 근처에서 가장 큰 값만 코너로 선정
                local_maxima = R_score[y - w:y + w + 1, x - w:x + w + 1]
                if val == np.max(local_maxima):
                    corners.append((x, y))
    
    return corners, img_smooth


def intensity_normalization(patch):
    """
    패치의 강도를 정규화 (평균을 빼고 표준편차로 나누기)
    """
    patch = patch.astype(np.float32)
    m = np.mean(patch)
    denom = np.sqrt(np.sum((patch - m) ** 2))
    
    if denom == 0:
        return patch
    
    patch_normalized = (patch - m) / denom
    return patch_normalized


def compute_ssd(patch1, patch2):
    """
    1-3 단계: SSD (Sum of Squared Differences) 계산
    """
    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)
    ssd = np.sum((patch2 - patch1) ** 2)
    return ssd


def correspondence_matching(img1, img2, patch_size=9, ssd_threshold=2.0, ratio_threshold=0.8):
    """
    1-3 단계: SSD를 사용해 포인트 간의 correspondence를 구해 포인트 매칭
    ssd_threshold: SSD 값이 이보다 크면 매칭 제외
    ratio_threshold: Ratio test 임계값 (가장 좋은 매칭 / 두 번째로 좋은 매칭)
    """
    # 코너 포인트 찾기
    corner1, img1_smooth = harris_corner_detection(img1)
    corner2, img2_smooth = harris_corner_detection(img2)
    
    radius = patch_size // 2
    r1, c1 = img1_smooth.shape
    r2, c2 = img2_smooth.shape
    
    matches = []
    
    print(f"이미지1의 코너 포인트: {len(corner1)}개")
    print(f"이미지2의 코너 포인트: {len(corner2)}개")
    print("포인트 매칭 중...")
    
    for x1, y1 in tqdm(corner1):
        # 경계 체크
        if (x1 - radius < 0) or (x1 + radius >= c1) or (y1 - radius < 0) or (y1 + radius >= r1):
            continue
        
        # 패치 추출 및 정규화
        patch1 = img1_smooth[y1 - radius:y1 + radius + 1, x1 - radius:x1 + radius + 1]
        patch1 = intensity_normalization(patch1)
        
        # 최고 매칭과 두 번째로 좋은 매칭 저장 (Ratio test용)
        best_ssd = float('inf')
        second_best_ssd = float('inf')
        best_match = None
        
        # 이미지2의 모든 코너와 비교
        for x2, y2 in corner2:
            # 경계 체크
            if (x2 - radius < 0) or (x2 + radius >= c2) or (y2 - radius < 0) or (y2 + radius >= r2):
                continue
            
            # 패치 추출 및 정규화
            patch2 = img2_smooth[y2 - radius:y2 + radius + 1, x2 - radius:x2 + radius + 1]
            patch2 = intensity_normalization(patch2)
            
            # SSD 계산
            ssd_score = compute_ssd(patch1, patch2)
            
            # 최고 매칭과 두 번째로 좋은 매칭 업데이트
            if ssd_score < best_ssd:
                second_best_ssd = best_ssd
                best_ssd = ssd_score
                best_match = (x2, y2, ssd_score)
            elif ssd_score < second_best_ssd:
                second_best_ssd = ssd_score
        
        # 매칭 조건 확인
        if best_match is not None:
            # SSD 임계값 체크
            if best_ssd > ssd_threshold:
                continue
            
            # Ratio test: 두 번째로 좋은 매칭이 너무 좋으면 불확실한 매칭
            if second_best_ssd > 0:
                ratio = best_ssd / second_best_ssd
                if ratio > ratio_threshold:
                    continue  # 불확실한 매칭 제외
            
            matches.append((x1, y1, best_match[0], best_match[1]))
    
    print(f"매칭된 점의 개수: {len(matches)}개")
    return matches


def compute_homography(src_pts, dst_pts):
    """
    1-5 단계: 4개의 점 쌍으로 Homography 행렬 계산 (DLT 알고리즘)
    src_pts를 dst_pts로 변환하는 Homography H 계산
    dst = H @ src 형태
    """
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        xp, yp = dst_pts[i][0], dst_pts[i][1]
        
        # DLT 공식: [x', y', 1]^T = H @ [x, y, 1]^T
        # 변환: src (x, y) -> dst (xp, yp)
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    
    A = np.array(A)
    
    # SVD로 Ah = 0 해 구하기
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1]  # 마지막 행 (가장 작은 고유값에 대응하는 고유벡터)
    
    H = L.reshape(3, 3)
    
    # H[2,2]을 1로 정규화
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    
    return H


def ransac_homography(matches, k=2000, threshold=5.0):
    """
    1-4 단계: RANSAC 알고리즘으로 Homography 행렬 추정
    """
    if len(matches) < 4:
        print("매칭 점이 부족합니다. (최소 4개 필요)")
        return None, []
    
    best_H = None
    max_inliers_count = -1
    best_inliers = []
    
    print(f"RANSAC 실행 중... (반복 횟수: {k})")
    
    for i in tqdm(range(k), desc="RANSAC"):
        # 1. 랜덤하게 4개 샘플 선택
        indices = np.random.choice(len(matches), 4, replace=False)
        # img1의 점 (src)을 img2의 점 (dst)로 변환하는 Homography 계산
        src_sample = [matches[idx][:2] for idx in indices]  # img1의 점
        dst_sample = [matches[idx][2:] for idx in indices]  # img2의 점
        
        # 2. Homography 계산: img1 -> img2 변환
        H = compute_homography(src_sample, dst_sample)
        
        # 3. 인라이어 개수 계산
        inliers = []
        for j in range(len(matches)):
            # img1의 점
            src_pt = np.array([matches[j][0], matches[j][1], 1.0])
            # img2의 점 (실제값)
            dst_pt = np.array([matches[j][2], matches[j][3]])
            
            # img1의 점을 img2 좌표계로 변환 (예측값)
            pred = H @ src_pt
            if abs(pred[2]) < 1e-10:
                continue
            
            pred_x = pred[0] / pred[2]
            pred_y = pred[1] / pred[2]
            
            # 예측값과 실제값의 거리 계산
            dist = np.sqrt((dst_pt[0] - pred_x) ** 2 + (dst_pt[1] - pred_y) ** 2)
            
            if dist < threshold:
                inliers.append(matches[j])
        
        # 4. 베스트 모델 갱신
        if len(inliers) > max_inliers_count:
            max_inliers_count = len(inliers)
            best_H = H
            best_inliers = inliers
    
    print(f"RANSAC 결과: 총 {len(matches)}개 중 {max_inliers_count}개 인라이어 발견")
    return best_H, best_inliers


def draw_matches(img1, img2, matches, max_display=50):
    """
    매칭 결과를 시각화하는 함수
    """
    # 두 이미지를 가로로 이어 붙이기
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 높이가 다르면 맞춰줌 (검은색 패딩)
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # 그레이스케일 이미지를 컬러로 변환
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
    
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    vis_img[:h1, :w1] = img1_color
    vis_img[:h2, w1:w1+w2] = img2_color
    
    # 매칭 선 그리기 (최대 max_display개만)
    display_matches = matches[:max_display] if len(matches) > max_display else matches
    
    for (x1, y1, x2, y2) in display_matches:
        # 랜덤 색상
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w1), int(y2))
        
        cv2.circle(vis_img, pt1, 5, color, -1)
        cv2.circle(vis_img, pt2, 5, color, -1)
        cv2.line(vis_img, pt1, pt2, color, 1)
    
    return vis_img


def warp_image(img, H, output_size):
    """
    이미지를 Homography 행렬로 변환 (Warping)
    """
    h, w = img.shape[:2]
    output_h, output_w = output_size
    
    warped = np.zeros((output_h, output_w), dtype=img.dtype)
    
    # 역변환 행렬 계산
    H_inv = np.linalg.inv(H)
    
    for y_out in range(output_h):
        for x_out in range(output_w):
            # 출력 이미지 좌표를 동차 좌표로 변환
            dst_pt = np.array([x_out, y_out, 1.0])
            
            # 원본 이미지 좌표로 변환
            src_pt = H_inv @ dst_pt
            if abs(src_pt[2]) < 1e-10:
                continue
            
            x_src = src_pt[0] / src_pt[2]
            y_src = src_pt[1] / src_pt[2]
            
            # 경계 체크
            if 0 <= x_src < w and 0 <= y_src < h:
                # 가장 가까운 픽셀 값 사용 (Nearest Neighbor Interpolation)
                x_int = int(round(x_src))
                y_int = int(round(y_src))
                
                if 0 <= x_int < w and 0 <= y_int < h:
                    warped[y_out, x_out] = img[y_int, x_int]
    
    return warped


def stitch_images(img1_path, img2_path, output_path, step_name="", use_img2_as_base=False):
    """
    1-6 단계: 두 이미지를 스티칭하여 파노라마 이미지 생성
    
    :param img1_path: 첫 번째 이미지 경로
    :param img2_path: 두 번째 이미지 경로
    :param output_path: 출력 경로
    :param step_name: 단계 이름 (여러 이미지 스티칭 시 구분용)
    :param use_img2_as_base: True면 img2를 기준으로 img1을 변환, False면 img1을 기준으로 img2를 변환
    """
    print("=" * 50)
    print("파노라마 이미지 스티칭 시작")
    print("=" * 50)
    
    # 1-1: 이미지를 그레이스케일로 변환
    print("\n[1-1] 이미지 로드 및 그레이스케일 변환")
    img1_gray = gray_img(img1_path)
    img2_gray = gray_img(img2_path)
    
    # 원본 컬러 이미지도 로드 (최종 결과용)
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)
    
    # 1-3: 포인트 매칭
    print("\n[1-3] 포인트 매칭 (SSD 사용)")
    # ssd_threshold와 ratio_threshold를 조정하여 매칭 품질 개선
    # ssd_threshold가 작을수록 더 엄격한 매칭 (기본: 2.0)
    # ratio_threshold가 작을수록 더 엄격한 Ratio test (기본: 0.8)
    matches = correspondence_matching(img1_gray, img2_gray, patch_size=9, ssd_threshold=1.5, ratio_threshold=0.75)
    
    if len(matches) < 4:
        print("매칭 점이 부족하여 스티칭을 수행할 수 없습니다.")
        return None
    
    # 매칭 결과 시각화
    print("\n매칭 결과 시각화 중...")
    matches_vis = draw_matches(img1_gray, img2_gray, matches, max_display=50)
    matches_file_before = f"matches_before_ransac{step_name}.jpg"
    cv2.imwrite(matches_file_before, matches_vis)
    print(f"매칭 결과 저장: {matches_file_before}")
    
    # 1-4, 1-5: RANSAC으로 Homography 계산
    print("\n[1-4, 1-5] RANSAC 및 Homography 계산")
    H, inliers = ransac_homography(matches, k=2000, threshold=5.0)
    
    if H is None:
        print("Homography 계산 실패")
        return None
    
    # 인라이어 매칭 결과 시각화
    if len(inliers) > 0:
        print("\n인라이어 매칭 결과 시각화 중...")
        inliers_vis = draw_matches(img1_gray, img2_gray, inliers, max_display=50)
        matches_file_after = f"matches_after_ransac{step_name}.jpg"
        cv2.imwrite(matches_file_after, inliers_vis)
        print(f"인라이어 매칭 결과 저장: {matches_file_after}")
    
    # 1-6: 스티칭
    print("\n[1-6] 이미지 스티칭")
    
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    
    # 기준 이미지에 따라 변환 행렬 결정
    if use_img2_as_base:
        # img2를 기준으로 img1을 변환
        print("기준 이미지: img2 (img1을 변환)")
        base_img = img2_color
        base_h, base_w = h2, w2
        transform_img = img1_color
        transform_h, transform_w = h1, w1
        
        # H는 img1 -> img2 변환 행렬
        # img2 기준이므로 img1을 img2 좌표계로 변환해야 함
        # transformed_corners = H_base_to_transform @ corners_transform
        # corners_transform은 transform(img1) 좌표계의 점
        # transformed_corners는 base(img2) 좌표계의 점
        # 따라서 H_base_to_transform은 transform -> base 변환, 즉 img1 -> img2 변환
        # 따라서 H를 사용
        H_base_to_transform = H  # transform(img1) -> base(img2) 변환
    else:
        # img1을 기준으로 img2를 변환 (기본)
        print("기준 이미지: img1 (img2를 변환)")
        base_img = img1_color
        base_h, base_w = h1, w1
        transform_img = img2_color
        transform_h, transform_w = h2, w2
        
        # H는 img1 -> img2 변환 행렬
        # img2를 img1 좌표계로 변환하려면 역행렬 필요
        H_base_to_transform = np.linalg.inv(H)
    
    # 변환할 이미지의 네 모서리를 기준 이미지 좌표계로 변환
    corners_transform = np.array([
        [0, 0, 1],
        [transform_w, 0, 1],
        [transform_w, transform_h, 1],
        [0, transform_h, 1]
    ]).T
    
    # 변환할 이미지의 코너들을 기준 이미지 좌표계로 변환
    transformed_corners = H_base_to_transform @ corners_transform
    transformed_corners = transformed_corners / transformed_corners[2, :]
    
    # 출력 이미지 크기 계산
    # 기준 이미지는 원점 (0, 0)에 위치
    min_x = min(0, np.min(transformed_corners[0, :]))
    max_x = max(base_w, np.max(transformed_corners[0, :]))
    min_y = min(0, np.min(transformed_corners[1, :]))
    max_y = max(base_h, np.max(transformed_corners[1, :]))
    
    # 패딩 추가
    offset_x = int(-min_x) + 50
    offset_y = int(-min_y) + 50
    output_w = int(max_x - min_x) + 100
    output_h = int(max_y - min_y) + 100
    
    # 출력 이미지 초기화
    panorama = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # panorama 좌표 -> 기준 이미지 좌표 변환 행렬
    T_pano_to_base = np.array([
        [1, 0, -offset_x],
        [0, 1, -offset_y],
        [0, 0, 1]
    ])
    
    # panorama 좌표 -> 변환할 이미지 좌표 변환 행렬
    # panorama -> base -> transform
    H_pano_to_transform = np.linalg.inv(H_base_to_transform) @ T_pano_to_base
    
    # 변환할 이미지의 각 픽셀을 변환하여 배치 (먼저 배치)
    print("변환할 이미지 변환 중...")
    for y_out in tqdm(range(output_h)):
        for x_out in range(output_w):
            # 출력 좌표를 동차 좌표로 변환
            pano_pt = np.array([x_out, y_out, 1.0])
            
            # 변환할 이미지 좌표로 변환
            transform_pt = H_pano_to_transform @ pano_pt
            
            if abs(transform_pt[2]) < 1e-10:
                continue
            
            x_transform = transform_pt[0] / transform_pt[2]
            y_transform = transform_pt[1] / transform_pt[2]
            
            # 경계 체크
            if 0 <= x_transform < transform_w and 0 <= y_transform < transform_h:
                # 가장 가까운 픽셀 값 사용
                x_int = int(round(x_transform))
                y_int = int(round(y_transform))
                
                if 0 <= x_int < transform_w and 0 <= y_int < transform_h:
                    panorama[y_out, x_out] = transform_img[y_int, x_int]
    
    # 기준 이미지를 출력 이미지에 배치 (나중에 배치하여 겹치는 영역에서 우선)
    print("기준 이미지 배치 중...")
    panorama[offset_y:offset_y + base_h, offset_x:offset_x + base_w] = base_img
    
    # 검은 부분 제거를 위한 크롭 (선택사항)
    # 간단하게는 검은 픽셀이 많은 부분을 제거할 수 있지만,
    # 여기서는 전체 결과를 저장
    
    # 결과 저장
    cv2.imwrite(output_path, panorama)
    print(f"\n파노라마 이미지 저장 완료: {output_path}")
    print(f"출력 이미지 크기: {output_h} x {output_w}")
    
    return panorama


def stitch_multiple_images(image_paths, output_path, max_images=None):
    """
    여러 이미지를 순차적으로 스티칭하여 파노라마 생성
    
    :param image_paths: 이미지 파일 경로 리스트
    :param output_path: 최종 파노라마 저장 경로
    :param max_images: 최대 스티칭할 이미지 개수 (None이면 모두 시도)
    """
    if len(image_paths) < 2:
        print("최소 2개 이상의 이미지가 필요합니다.")
        return None
    
    # 최대 이미지 개수 제한
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print("=" * 60)
    print(f"여러 이미지 스티칭 시작 (총 {len(image_paths)}장)")
    print("=" * 60)
    
    # 첫 번째 이미지를 기준으로 시작
    current_panorama_path = image_paths[0]
    print(f"\n[기준 이미지] {current_panorama_path}")
    
    # 임시 파일 경로
    temp_output = "temp_panorama.jpg"
    
    # 두 번째 이미지부터 순차적으로 붙이기
    for i in range(1, len(image_paths)):
        next_image_path = image_paths[i]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(image_paths)-1}] {next_image_path} 스티칭 시도")
        print(f"{'='*60}")
        
        try:
            # 홀수 번째는 첫 번째 이미지를 기준으로, 짝수 번째는 두 번째 이미지를 기준으로
            # i가 홀수면 use_img2_as_base=False (img1 기준)
            # i가 짝수면 use_img2_as_base=True (img2 기준)
            use_img2_as_base = (i % 2 == 0)
            
            step_name = f"_step{i}"
            result = stitch_images(
                current_panorama_path, 
                next_image_path, 
                temp_output,
                step_name=step_name,
                use_img2_as_base=use_img2_as_base
            )
            
            if result is None:
                print(f"\n⚠️  {next_image_path} 스티칭 실패 - 여기서 중단")
                print(f"현재까지 {i}장의 이미지가 스티칭되었습니다.")
                break
            
            # 다음 반복을 위해 현재 파노라마 경로 업데이트
            current_panorama_path = temp_output
            print(f"\n✓ {i+1}장의 이미지 스티칭 성공!")
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print(f"현재까지 {i}장의 이미지가 스티칭되었습니다.")
            import traceback
            traceback.print_exc()
            break
    
    # 최종 결과를 지정된 경로로 복사
    if os.path.exists(temp_output):
        import shutil
        shutil.copy(temp_output, output_path)
        print(f"\n{'='*60}")
        print(f"최종 파노라마 저장: {output_path}")
        print(f"{'='*60}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        return cv2.imread(output_path)
    else:
        print("\n❌ 파노라마 생성 실패")
        return None


if __name__ == "__main__":
    # 여러 이미지를 순차적으로 스티칭
    image_paths = [f"./rio/testimg{n+1}.png" for n in range(10)]
    
    output_path = "./panorama_aaa.jpg"
    
    try:
        # 5장의 이미지를 스티칭
        panorama = stitch_multiple_images(image_paths, output_path, max_images=3)
        
        if panorama is not None:
            print("\n✓ 스티칭 완료!")
        else:
            print("\n❌ 스티칭 실패")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

