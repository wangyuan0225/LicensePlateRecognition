package com.wy0225.controller;

import com.wy0225.common.JwtUtil;
import com.wy0225.common.Result;
import com.wy0225.entity.Feedback;
import com.wy0225.service.FeedbackService;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/v1/feedback")
@RequiredArgsConstructor
public class FeedbackController {

    private final FeedbackService feedbackService;
    private final JwtUtil jwtUtil;

    @PostMapping
    public Result<Feedback> submitFeedback(
            @RequestHeader(value = "Authorization", required = false) String authHeader,
            @RequestBody FeedbackRequest request) {

        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        try {
            Feedback feedback = feedbackService.submitFeedback(
                    userId,
                    request.getOriginalImageUrl(),
                    request.getResultImageUrl(),
                    request.getRecognizedPlate(),
                    request.getCorrectedPlate(),
                    request.getModelType());
            return Result.success("Feedback submitted successfully", feedback);
        } catch (Exception e) {
            log.error("Failed to submit feedback", e);
            return Result.error(500, "提交反馈失败: " + e.getMessage());
        }
    }

    @GetMapping("/list")
    public Result<List<Feedback>> listFeedbacks(
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        try {
            List<Feedback> feedbacks = feedbackService.getUserFeedbacks(userId);
            return Result.success("Success", feedbacks);
        } catch (Exception e) {
            log.error("Failed to list feedbacks", e);
            return Result.error(500, "获取反馈列表失败: " + e.getMessage());
        }
    }

    @DeleteMapping("/{id}")
    public Result<String> deleteFeedback(
            @PathVariable Long id,
            @RequestHeader(value = "Authorization", required = false) String authHeader) {

        Long userId = extractUserId(authHeader);
        if (userId == null) {
            return Result.error(401, "请先登录");
        }

        try {
            boolean success = feedbackService.deleteFeedback(id, userId);
            if (success) {
                return Result.success("Feedback deleted successfully", null);
            } else {
                return Result.error(404, "记录不存在或无权限删除");
            }
        } catch (Exception e) {
            log.error("Failed to delete feedback", e);
            return Result.error(500, "删除反馈失败: " + e.getMessage());
        }
    }

    private Long extractUserId(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return null;
        }
        String token = authHeader.substring(7);
        if (!jwtUtil.validateToken(token)) {
            return null;
        }
        return jwtUtil.getUserIdFromToken(token);
    }

    @Data
    public static class FeedbackRequest {
        private String originalImageUrl;
        private String resultImageUrl;
        private String recognizedPlate;
        private String correctedPlate;
        private String modelType;
    }
}
