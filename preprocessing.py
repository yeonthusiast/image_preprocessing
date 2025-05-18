#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import imutils
from imutils.contours import sort_contours


def four_point_transform(image, pts):
    """
    4개의 꼭짓점을 이용해 원근 보정(perspective transform) 후 크롭된 이미지를 반환합니다.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def detect_and_crop_mask(org_path, debug_dir):
    """
    마스크 기반으로 영수증을 크롭한 컬러 이미지를 리턴합니다.
    """
    org = cv2.imread(org_path)
    if org is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {org_path}")

    base_name = os.path.splitext(os.path.basename(org_path))[0]

    # 1) 그레이스케일 → OTSU 역이진화로 종이 마스크 생성
    gray_full = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{debug_dir}/1_{base_name}_gray_full.jpg", gray_full)
    _, mask = cv2.threshold(
        gray_full, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    cv2.imwrite(f"{debug_dir}/2_{base_name}_mask_otsu.jpg", mask)

    # 2) 모폴로지 Closing → 틈·구멍 메우기
    h, w = org.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//20, h//20))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{debug_dir}/3_{base_name}_mask_closed.jpg", mask_closed)

    # 3) 면적 50% 이상 컨투어 필터 → 최대 contour 선택
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = w * h
    cnts = [c for c in cnts if cv2.contourArea(c) >= 0.5 * total_area]
    if not cnts:
        raise RuntimeError("유의미한 영수증 영역을 찾지 못했습니다.")
    c = max(cnts, key=cv2.contourArea)

    # 4) 4점 얻기 (approxPolyDP or fallback)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect).astype("float32")

    # 5) 디버그: 윤곽 그려 저장
    outline = org.copy()
    cv2.drawContours(outline, [pts.astype(int)], -1, (0, 255, 0), 3)
    cv2.imwrite(f"{debug_dir}/4_{base_name}_outline.jpg", outline)

    # 6) 컬러 크롭 및 저장
    cropped_color = four_point_transform(org, pts)
    cv2.imwrite(f"{debug_dir}/5_{base_name}_transformed_color.jpg", cropped_color)

    return cropped_color, base_name


def binarize_for_ocr(cropped_color, debug_dir, base_name):
    """
    크롭된 컬러 이미지를 그레이→OTSU 이진화하여,
    OCR에 최적화된 선명한 흑백 이미지를 리턴합니다.
    """
    gray = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{debug_dir}/6_{base_name}_cropped_gray.jpg", gray)

    _, cropped_bin = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    cv2.imwrite(f"{debug_dir}/7_{base_name}_cropped_bin.jpg", cropped_bin)

    return cropped_bin


def main():
    parser = argparse.ArgumentParser(
        description="Mask 기반 크롭 + 이진화(선명)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="원본 이미지 경로 (예: images/sample1.jpg)")
    parser.add_argument("--output_bin", "-b", required=True,
                        help="최종 OCR용 이진 크롭 저장 폴더 (예: output/)")
    args = parser.parse_args()

    out_bin_dir = args.output_bin.rstrip("/")
    os.makedirs(out_bin_dir, exist_ok=True)
    debug_dir = os.path.join(out_bin_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # 1) 컬러 크롭
    cropped_color, base_name = detect_and_crop_mask(args.input, debug_dir)
    # 2) OCR용 이진화
    cropped_bin = binarize_for_ocr(cropped_color, debug_dir, base_name)
    # 최종 이진 크롭 저장: receipt17_bin.jpg 형태
    final_bin_path = os.path.join(out_bin_dir, f"{base_name}_bin.jpg")
    cv2.imwrite(final_bin_path, cropped_bin)
    print(f"✅ OCR용 이진화 크롭 완료 → {final_bin_path}")

if __name__ == "__main__":
    main()
