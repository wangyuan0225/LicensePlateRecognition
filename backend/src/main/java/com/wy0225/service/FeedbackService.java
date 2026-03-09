package com.wy0225.service;

import com.wy0225.entity.Feedback;

import java.util.List;

public interface FeedbackService {
    Feedback submitFeedback(Long userId, String originalImageUrl, String resultImageUrl, String recognizedPlate,
            String correctedPlate, String modelType);

    List<Feedback> getUserFeedbacks(Long userId);
}
