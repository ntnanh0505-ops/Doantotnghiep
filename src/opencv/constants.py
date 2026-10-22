"""
constants.py — Hằng số chuẩn quốc tế ISO/IEC 7810 ID-1 và cài đặt toàn cục.
"""

# ==============================================================================
# HẰNG SỐ CHUẨN QUỐC TẾ (ISO/IEC 7810 ID-1)
# ==============================================================================
CARD_LONG_MM = 85.60    # Chiều dài chuẩn thẻ ID-1 (CCCD / thẻ ngân hàng / bằng lái)
CARD_SHORT_MM = 53.98   # Chiều rộng chuẩn thẻ ID-1
CARD_ASPECT_RATIO = CARD_LONG_MM / CARD_SHORT_MM  # ~ 1.58577...
MAX_IMAGE_DIM = 2500    # Giới hạn kích thước tối đa của ảnh đã nắn để tránh tràn RAM
DEBUG_MODE = False      # Biến cấu hình bật/tắt chế độ debug trung gian toàn cục
