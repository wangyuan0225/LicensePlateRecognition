package com.wy0225.service;

import java.util.Map;

public interface HistoryService {
    Map<String, Object> getHistoryList(Long userId, int page, int size, String keyword, String startDate, String endDate);
    void deleteRecord(Long id);
    long getTotalCount();
}
