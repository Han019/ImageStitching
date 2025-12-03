import numpy as np
import cv2
from tqdm import tqdm
from tqdm import trange
import time


# class Panorama(self):

#     def __init__(self):

#     def SSD(self, img1, img2):

#     def Homography():

#     def Mosaic():

#     def Smoothing():

#     def Warping():

#     def Interpolation():

#     def Stabilizing():
"""
이미지 보기

"""
def gray_img(img_path):
    img_matrix = cv2.imread(img_path)
    img_matrix.astype(np.float32)

    #흑백 만들기
    gray_weight=np.array([0.114,0.587,0.299])
    gray_img = img_matrix @ gray_weight
    gray_matrix=gray_img.astype(np.uint8)

    return gray_matrix


"""1. 이미지 전처리/ 노이즈 제거"""
# 가우시안 커널
def _gaussian_kernel(kernel_size, sigma): 

    #kernel_size = 3, 5, 7, ...
    #가운데를 0으로 만들기
    x = np.arange(-(kernel_size)//2 , (kernel_size)//2 +1)
    kernel = np.exp(-(x**2)/(2*sigma**2))/(2 * np.pi * sigma**2)

    kernel = kernel/ np.sum(kernel)

    return kernel

def _gaussian_smoothing(img,sigma=1):

    kernel_size=7
    kernel = _gaussian_kernel(kernel_size,sigma)
    r, c = img.shape
    
    img_smooth_x = np.zeros_like(img, dtype=np.float32)
    
    # 가로 방향
    for y in range(r):
        img_smooth_x[y,:] = np.convolve(img[y,:], kernel, mode='same')


    img_smooth_final = np.zeros_like(img, dtype=np.float32)

    #세로 방향 
    for x in range(c):
        img_smooth_final[:,x] = np.convolve(img_smooth_x[:,x], kernel, mode='same')
        
    
    # 출력 확인용
    # img_smooth_final= img_smooth_final.astype(np.uint8)
    
    # Defference of Gaussian 엣지
    # test= img-img_smooth_final
    # test = test.astype(np.uint8)
    # return test

    return img_smooth_final




"""2. 코너 포인트 찾기 (필수)"""
def harris_corner_detection(img):
    """
    img는 1단계 전처리가 끝난 흑백 이미지
    """
    img=_gaussian_smoothing(img=img,sigma=1.3)

    img=img.astype(np.float32)
    r,c = img.shape
    
    #미분 필터
    Ix=np.zeros_like(img).astype(np.float32)
    Iy=np.zeros_like(img).astype(np.float32)

    kernel=np.array([1,0,-1])
    for y in range(r):
        Ix[y,:]=np.convolve(img[y,:], kernel, mode='same')

    for x in range(c):
        Iy[:,x]=np.convolve(img[:,x], kernel, mode='same')
    
    Ixx=Ix**2
    Ixy=Ix*Iy
    Iyy=Iy**2

    """
    M=[Ixx Ixy]
      [Ixy Iyy]
    
    W= gaussian, sigma=1
    """

    Mxx=_gaussian_smoothing(Ixx)
    Mxy=_gaussian_smoothing(Ixy)
    Myy=_gaussian_smoothing(Iyy)

    detM= Mxx*Myy -Mxy*Mxy
    trM= Mxx+Myy
    # R score
    
    k=0.05
    R_score = detM -k*(trM)**2
    # threshold > 10000 corners
    # threshold < -10000 edges
    # -10000 < R < 10000 neither corners nor edges
    threshold = 10000
    
    corners=np.zeros((r,c),dtype=np.uint8)
    corner=[]
    
    # window_size= 3,5,7,9, ... 

    window_size=5
    w= window_size//2
    for y in range(w,r-w):
        for x in range(w,c-w):
            val=R_score[y,x]
            if val > threshold:
                # 근처에서 가장 큰 놈만 코너로 선정
                local_maxima = R_score[y-w:y+w+1, x-w:x+w+1]

                if val == np.max(local_maxima):
                    corner.append((x,y))
                    corners[y,x]= 255
    
    #테스트
    # im_show(corners.astype(np.uint8))
    
    # 코너 정보 배열, 스무딩이 적용된 이미지
    return corner, img

""" 3-1. SSD """ 
def compute_ssd(patch1, patch2):  
    
    patch1=patch1.astype(np.float32)
    patch2=patch2.astype(np.float32)
    ssd = np.sum((patch2-patch1)**2)
    
    return ssd



"""3.point matching(필수)"""
# 개선 필요.... ransac하면 괜찮아진다고는 하는데 너무 매칭되는 선들이 이상함

def correspondence_matching(img1, img2):

    corner1,img1 = harris_corner_detection(img1)
    corner2,img2 = harris_corner_detection(img2)
    
    patch_size=9

    radius=patch_size//2

    r1,c1 =img1.shape
    r2,c2 =img2.shape
    
    matches=[]
    print("corner matching!!")
    
    for x1,y1 in tqdm(corner1):
        if (x1 - radius < 0) or (x1 + radius >= c1) or (y1 - radius < 0) or (y1 + radius >= r1):
            continue
        #앞의 이미지, 코너를 중심으로 한 패치 떼어내기
        patch1 = img1[y1-radius:y1+radius+1, x1-radius:x1+radius+1]
        patch1 = intensity_normalization(patch1)

        optimal_ssd=float('inf')
        best_match= None

        for x2, y2 in corner2:
            if (x2 - radius < 0) or (x2 + radius >= c2) or (y2 - radius < 0) or (y2 + radius >= r2):
                continue

            patch2 = img2[y2-radius:y2+radius+1, x2-radius:x2+radius+1]
            patch2 = intensity_normalization(patch2)

            ssd_score = compute_ssd(patch1,patch2)

            if ssd_score < optimal_ssd:
                optimal_ssd=ssd_score
                best_match=(x2,y2)

        if best_match != None:
            matches.append((x1,y1, best_match[0], best_match[1]))

    return matches

def intensity_normalization(patch):

    patch = patch.astype(np.float32)
    m = np.mean(patch)
    denom=np.sqrt(np.sum((patch-m)**2))
    if denom ==0:
        return patch
        
    patch_hat = (patch-m)/denom

    return patch_hat



def draw_matches(img1, img2, matches):
    # 두 이미지를 가로로 이어 붙이기
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # 높이가 다르면 맞춰줌 (검은색 패딩)
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w), dtype=np.uint8)
    
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2
    
    # 컬러로 변환 (선을 색깔로 그리기 위해)
    vis_img_color = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    
    for (x1, y1, x2, y2) in matches:
        # img1의 점 (x1, y1)
        # img2의 점 (x2, y2) -> 시각화 이미지에서는 x좌표가 w1만큼 이동됨
        pt1 = (x1, y1)
        pt2 = (x2 + w1, y2)
        
        # 랜덤 색상
        color = np.random.randint(0, 255, 3).tolist()
        
        cv2.circle(vis_img_color, pt1, 4, color, 1)
        cv2.circle(vis_img_color, pt2, 4, color, 1)
        cv2.line(vis_img_color, pt1, pt2, color, 1)
        
    cv2.imshow("Matches", vis_img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""4.RANSAC"""
""" 4 & 5. RANSAC 및 Homography 계산 """

def compute_homography(src_pts, dst_pts):
    """
    4개의 점 쌍으로 Homography H (3x3)를 구하는 함수 (DLT 알고리즘)
    src_pts: 왼쪽 이미지 점 [(x, y), ...]
    dst_pts: 오른쪽 이미지 점 [(x, y), ...]
    """
    A = []
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        xp, yp = dst_pts[i][0], dst_pts[i][1]
        
        # DLT 공식 (CVLecture9.pdf Slide 28)
        # [-x, -y, -1, 0, 0, 0, x*x', y*x', x']
        # [0, 0, 0, -x, -y, -1, x*y', y*y', y']
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        
    A = np.array(A)
    
    # SVD로 Ah = 0 해 구하기 (마지막 행이 해가 됨)
    # numpy.linalg.svd는 사용 가능 (Matrix 계산 함수 예외 허용)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1] # V의 마지막 행 (가장 작은 고유값에 대응하는 고유벡터)
    
    H = L.reshape(3, 3)
    
    # H[3,3]을 1로 정규화 (선택사항이나 일반적)
    if H[2, 2] != 0:
        H = H / H[2, 2]
        
    return H

def ransac_homography(matches, k=2000, threshold=5.0):
    """
    matches: point_matching의 결과 리스트 [(x1, y1, x2, y2), ...]
    k: 반복 횟수 (1000~2000 추천)
    threshold: 인라이어 판단 거리 기준 (픽셀 단위)
    """
    best_H = None
    max_inliers_count = -1
    best_inliers = []
    
    # 매칭 점이 4개 미만이면 계산 불가
    if len(matches) < 4:
        print("매칭 점이 부족합니다.")
        return None, []

    src_pts_all = np.float32([ [m[0], m[1]] for m in matches ]).reshape(-1, 1, 2)
    dst_pts_all = np.float32([ [m[2], m[3]] for m in matches ]).reshape(-1, 1, 2)

    for i in trange(k):
        # 1. 랜덤하게 4개 샘플 뽑기
        # random.sample 대신 numpy choice 사용
        indices = np.random.choice(len(matches), 4, replace=False)
        src_sample = [matches[idx][:2] for idx in indices]
        dst_sample = [matches[idx][2:] for idx in indices]
        
        # 2. 모델 추정 (Homography 계산)
        H = compute_homography(src_sample, dst_sample)
        
        # 3. 검증 (모든 점에 대해 에러 계산)
        # 행렬 연산으로 한 번에 변환: p' = H * p
        # src_pts_all을 동차 좌표계(Homogeneous)로 변환: [x, y, 1]
        ones = np.ones((len(matches), 1))
        pts_homogeneous = np.hstack([src_pts_all.squeeze(), ones]) # (N, 3)
        
        # H와 곱하기 (Transpose 주의: H * P.T)
        projected = H @ pts_homogeneous.T # (3, N)
        
        # 동차 좌표계 정규화 (x/w, y/w)
        # w가 0에 가까우면 제외 (무한대)
        w = projected[2, :]
        valid_idx = np.abs(w) > 1e-10
        
        projected_x = projected[0, valid_idx] / w[valid_idx]
        projected_y = projected[1, valid_idx] / w[valid_idx]
        
        # 실제 목표점(dst)과의 거리(에러) 계산
        dst_valid = dst_pts_all[valid_idx].squeeze()
        dx = projected_x - dst_valid[:, 0]
        dy = projected_y - dst_valid[:, 1]
        errors = np.sqrt(dx**2 + dy**2)
        
        # 4. 인라이어 개수 세기
        current_inliers_idx = np.where(errors < threshold)[0]
        # valid_idx 때문에 인덱스가 꼬일 수 있으므로 원래 인덱스 매핑 필요하지만,
        # 간단하게 구현하기 위해 루프 내에서 처리하거나 위 벡터 연산 사용
        
        if len(current_inliers_idx) > max_inliers_count:
            max_inliers_count = len(current_inliers_idx)
            best_H = H
            # 인라이어 매칭 정보 저장 (원래 matches 리스트에서 가져옴)
            # 주의: 위 벡터 연산에서 valid_idx로 필터링된 인덱스를 고려해야 함.
            # 여기서는 편의상 단순 루프나 리스트 컴프리헨션으로 정확한 인라이어를 저장하는게 안전할 수 있음.
            # (아래는 안전한 리스트 방식 재구현)
            best_inliers = []
            for idx, error in enumerate(errors):
                if error < threshold:
                    # valid_idx가 True인 것들 중에서의 인덱스이므로
                    # 원본 matches에서의 인덱스를 찾으려면 복잡해짐.
                    # --> 쉬운 구현: 그냥 루프 돌려서 거리 재는게 파이썬에선 느려도 확실함.
                    pass 

    # --- (벡터 연산이 복잡하면 아래의 '쉬운 루프 버전'으로 대체하세요) ---
    # RANSAC 루프 내부 (쉬운 버전)
        inliers = []
        count = 0
        for j in range(len(matches)):
            src_pt = np.array([matches[j][0], matches[j][1], 1])
            dst_pt = np.array([matches[j][2], matches[j][3]])
            
            # 예측 좌표 계산
            pred = H @ src_pt
            if abs(pred[2]) < 1e-10: continue
            pred_x = pred[0] / pred[2]
            pred_y = pred[1] / pred[2]
            
            # 거리 계산
            dist = np.sqrt((dst_pt[0] - pred_x)**2 + (dst_pt[1] - pred_y)**2)
            
            if dist < threshold:
                count += 1
                inliers.append(matches[j])
        
        # 베스트 갱신
        if count > max_inliers_count:
            max_inliers_count = count
            best_H = H
            best_inliers = inliers

    print(f"매칭된 점 개수: {len(matches)}")
    print(f"RANSAC 인라이어 개수: {len(inliers)}")
    print(inliers)
    print(f"RANSAC 결과: 총 {len(matches)}개 중 {max_inliers_count}개 인라이어 발견")
    return best_H, best_inliers




"""5. Homography(필수)"""
"""6. Stitching(필수)"""
"""7. Group Adjustment"""
"""8. Tone Mapping"""










def im_show(matrix):

    cv2.imshow("corner",matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    
    im1="./rio/testimg1.png"#"./im1.png"#
    im2="./rio/testimg2.png"#"./im2.png"#

    im1=gray_img(im1)
    im2=gray_img(im2)

    
    # patch_size가 클수록 구별력이 좋아지지만 계산이 느려짐 (9~15 추천)
    matches = correspondence_matching(im1, im2)
    print(f"매칭된 점의 개수: {len(matches)}개")
    H, inliners= ransac_homography(matches,2000, 5.0)
    # 결과 확인
    draw_matches(im1, im2, inliners)
