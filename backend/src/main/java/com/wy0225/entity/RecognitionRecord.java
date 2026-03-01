package com.wy0225.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "recognition_records")
public class RecognitionRecord {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id")
    private Long userId;

    /** Original uploaded image filename (stored in upload/images/) */
    @Column(name = "original_image", length = 500)
    private String originalImage;

    /** Result image filename (stored in algorithm result dir) */
    @Column(name = "result_image", length = 500)
    private String resultImage;

    /** Recognized plate number text, e.g. "皖1149885" */
    @Column(name = "plate_number", length = 50)
    private String plateNumber;

    /** Plate color, e.g. "绿色" */
    @Column(name = "plate_color", length = 20)
    private String plateColor;

    /** Plate type description, e.g. "绿色双层" */
    @Column(name = "plate_type", length = 50)
    private String plateType;

    /** Algorithm model used */
    @Column(name = "model_type", length = 50)
    private String modelType;

    /** Processing time in milliseconds */
    @Column(name = "processing_time_ms")
    private Double processingTimeMs;

    /** Number of plates detected */
    @Column(name = "detect_count")
    private Integer detectCount;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }
}
