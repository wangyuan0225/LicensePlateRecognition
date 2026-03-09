package com.wy0225.service.impl;

import com.wy0225.entity.Feedback;
import com.wy0225.repository.FeedbackRepository;
import com.wy0225.service.FeedbackService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class FeedbackServiceImpl implements FeedbackService {

    private final FeedbackRepository feedbackRepository;

    @Override
    @Transactional
    public Feedback submitFeedback(Long userId, String originalImageUrl, String resultImageUrl, String recognizedPlate,
            String correctedPlate, String modelType) {
        Feedback feedback = new Feedback();
        feedback.setUserId(userId);
        feedback.setOriginalImageUrl(originalImageUrl);
        feedback.setResultImageUrl(resultImageUrl);
        feedback.setRecognizedPlate(recognizedPlate);
        feedback.setCorrectedPlate(correctedPlate);
        feedback.setModelType(modelType);

        return feedbackRepository.save(feedback);
    }

    @Override
    public List<Feedback> getUserFeedbacks(Long userId) {
        return feedbackRepository.findByUserIdOrderByCreatedAtDesc(userId);
    }
}
