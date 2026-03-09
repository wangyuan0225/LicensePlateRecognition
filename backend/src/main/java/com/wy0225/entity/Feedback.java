package com.wy0225.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "feedbacks")
public class Feedback {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id")
    private Long userId;

    @Column(name = "original_image_url", length = 500)
    private String originalImageUrl;

    @Column(name = "result_image_url", length = 500)
    private String resultImageUrl;

    @Column(name = "recognized_plate", length = 50)
    private String recognizedPlate;

    @Column(name = "corrected_plate", length = 50)
    private String correctedPlate;

    @Column(name = "model_type", length = 50)
    private String modelType;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}
