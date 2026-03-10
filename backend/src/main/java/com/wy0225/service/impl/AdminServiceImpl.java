package com.wy0225.service.impl;

import com.wy0225.service.*;

import com.wy0225.entity.Feedback;
import com.wy0225.entity.RecognitionRecord;
import com.wy0225.repository.FeedbackRepository;
import com.wy0225.repository.RecognitionRecordRepository;
import com.wy0225.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AdminServiceImpl implements AdminService {

    private final UserRepository userRepository;
    private final RecognitionRecordRepository recordRepository;
    private final FeedbackRepository feedbackRepository;

    public List<Map<String, Object>> getAllUsersForDropdown() {
        return userRepository.findAll().stream()
                .map(u -> {
                    Map<String, Object> map = new HashMap<>();
                    map.put("id", u.getId());
                    map.put("username", u.getUsername());
                    return map;
                })
                .collect(Collectors.toList());
    }

    public Map<String, Object> getAllHistoryWithFilters(int page, int size, Long userId, String modelType) {
        Pageable pageable = PageRequest.of(page - 1, size);

        Page<RecognitionRecord> recordPage = recordRepository.findAllWithFilters(
                userId, (modelType != null && !modelType.isBlank()) ? modelType : null, pageable);

        List<Map<String, Object>> records = new ArrayList<>();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        // Fetch user cache to append username
        Map<Long, String> userCache = new HashMap<>();

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

            // Append Username
            Long uId = record.getUserId();
            item.put("userId", uId);
            if (!userCache.containsKey(uId)) {
                userRepository.findById(uId).ifPresent(u -> userCache.put(uId, u.getUsername()));
            }
            item.put("username", userCache.getOrDefault(uId, "Unknown"));

            records.add(item);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("total", recordPage.getTotalElements());
        result.put("current", page);
        result.put("size", size);
        result.put("records", records);
        return result;
    }

    public List<Map<String, Object>> getAllFeedbackWithFilters(Long userId, String modelType) {
        List<Feedback> feedbacks = feedbackRepository.findAllWithFilters(
                userId, (modelType != null && !modelType.isBlank()) ? modelType : null);

        Map<Long, String> userCache = new HashMap<>();
        List<Map<String, Object>> result = new ArrayList<>();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        for (Feedback f : feedbacks) {
            Map<String, Object> item = new HashMap<>();
            item.put("id", f.getId());
            item.put("originalImageUrl", f.getOriginalImageUrl());
            item.put("resultImageUrl", f.getResultImageUrl());
            item.put("recognizedPlate", f.getRecognizedPlate());
            item.put("correctedPlate", f.getCorrectedPlate());
            item.put("modelType", f.getModelType());
            item.put("status", f.getStatus());
            item.put("createdAt", f.getCreatedAt() != null ? f.getCreatedAt().format(formatter) : "");

            Long uId = f.getUserId();
            item.put("userId", uId);
            if (!userCache.containsKey(uId)) {
                userRepository.findById(uId).ifPresent(u -> userCache.put(uId, u.getUsername()));
            }
            item.put("username", userCache.getOrDefault(uId, "Unknown"));

            result.add(item);
        }

        return result;
    }

    public boolean updateFeedbackStatus(Long id, String status) {
        return feedbackRepository.findById(id).map(f -> {
            f.setStatus(status);
            feedbackRepository.save(f);
            return true;
        }).orElse(false);
    }
}
