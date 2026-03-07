package com.wy0225.service;

import com.wy0225.config.AlgorithmConfig;
import com.wy0225.entity.RecognitionRecord;
import com.wy0225.repository.RecognitionRecordRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class HistoryService {

    private final RecognitionRecordRepository recordRepository;
    private final AlgorithmConfig algorithmConfig;

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
        RecognitionRecord record = recordRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("记录不存在"));

        // Delete original uploaded image from upload/images/{userId}/
        deleteFileIfExists(algorithmConfig.getUpload().getDir(),
                record.getUserId().toString(), record.getOriginalImage());

        // Delete result image from upload/results/{userId}/
        deleteFileIfExists(algorithmConfig.getResult().getDir(),
                record.getUserId().toString(), record.getResultImage());

        recordRepository.deleteById(id);
        log.info("Record {} deleted along with its image files.", id);
    }

    /**
     * Deletes a file at baseDir/subDir/filename. Silently skips if the file
     * does not exist; logs a warning on unexpected IO errors.
     */
    private void deleteFileIfExists(String baseDir, String subDir, String filename) {
        if (filename == null || filename.isBlank()) {
            return;
        }
        Path target = Paths.get(baseDir, subDir, filename).toAbsolutePath();
        try {
            boolean deleted = Files.deleteIfExists(target);
            if (deleted) {
                log.info("Deleted file: {}", target);
            } else {
                log.warn("File not found (skipped): {}", target);
            }
        } catch (IOException e) {
            log.warn("Failed to delete file {}: {}", target, e.getMessage());
        }
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
