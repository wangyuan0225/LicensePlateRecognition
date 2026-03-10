package com.wy0225.service;

import java.util.List;
import java.util.Map;

public interface AdminService {
    List<Map<String, Object>> getAllUsersForDropdown();
    Map<String, Object> getAllHistoryWithFilters(int page, int size, Long userId, String modelType);
    List<Map<String, Object>> getAllFeedbackWithFilters(Long userId, String modelType);
    boolean updateFeedbackStatus(Long id, String status);
}
