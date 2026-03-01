package com.wy0225.service;

import com.wy0225.entity.RecognitionRecord;
import com.wy0225.repository.RecognitionRecordRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

@Service
@RequiredArgsConstructor
public class HistoryService {

    private final RecognitionRecordRepository recordRepository;

    public Map<String, Object> getHistoryList(Long userId, int page, int size, String keyword, String startDate,
            String endDate) {
        Pageable pageable = PageRequest.of(page - 1, size);

        LocalDateTime startDateTime = parseDate(startDate, true);
        LocalDateTime endDateTime = parseDate(endDate, false);

        String searchKeyword = (keyword != null && !keyword.isBlank()) ? keyword : null;

        Page<RecognitionRecord> recordPage = recordRepository.findByUserIdWithFilters(
                userId, searchKeyword, startDateTime, endDateTime, pageable);

        List<Map<String, Object>> records = new ArrayList<>();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        for (RecognitionRecord record : recordPage.getContent()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", record.getId());
            item.put("createdAt", record.getCreatedAt() != null ? record.getCreatedAt().format(formatter) : "");
            item.put("plateNumber", record.getPlateNumber());
            item.put("plateColor", record.getPlateColor());
            item.put("plateType", record.getPlateType());
            item.put("modelType", record.getModelType());
            item.put("processingTimeMs", record.getProcessingTimeMs());
            item.put("detectCount", record.getDetectCount());
            item.put("thumbnailUrl", "/static/upload/" + record.getUserId() + "/" + record.getOriginalImage());
            item.put("resultImageUrl", "/static/result/" + record.getUserId() + "/" + record.getResultImage());
            item.put("originalImageUrl", "/static/upload/" + record.getUserId() + "/" + record.getOriginalImage());
            records.add(item);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("total", recordPage.getTotalElements());
        result.put("current", page);
        result.put("size", size);
        result.put("records", records);
        return result;
    }

    public void deleteRecord(Long id) {
        if (!recordRepository.existsById(id)) {
            throw new RuntimeException("记录不存在");
        }
        recordRepository.deleteById(id);
    }

    public long getTotalCount() {
        return recordRepository.count();
    }

    private LocalDateTime parseDate(String dateStr, boolean isStart) {
        if (dateStr == null || dateStr.isBlank()) {
            return null;
        }
        try {
            LocalDate date = LocalDate.parse(dateStr, DateTimeFormatter.ISO_LOCAL_DATE);
            return isStart ? date.atStartOfDay() : date.atTime(23, 59, 59);
        } catch (DateTimeParseException e) {
            try {
                return LocalDateTime.parse(dateStr, DateTimeFormatter.ISO_LOCAL_DATE_TIME);
            } catch (DateTimeParseException e2) {
                return null;
            }
        }
    }
}
