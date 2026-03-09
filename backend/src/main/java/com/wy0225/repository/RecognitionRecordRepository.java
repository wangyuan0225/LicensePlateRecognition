package com.wy0225.repository;

import com.wy0225.entity.RecognitionRecord;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;

public interface RecognitionRecordRepository extends JpaRepository<RecognitionRecord, Long> {

        Page<RecognitionRecord> findAllByOrderByCreatedAtDesc(Pageable pageable);

        @Query("SELECT r FROM RecognitionRecord r WHERE " +
                        "r.userId = :userId AND " +
                        "(:keyword IS NULL OR r.plateNumber LIKE %:keyword%) AND " +
                        "(:startDate IS NULL OR r.createdAt >= :startDate) AND " +
                        "(:endDate IS NULL OR r.createdAt <= :endDate) " +
                        "ORDER BY r.createdAt DESC")
        Page<RecognitionRecord> findByUserIdWithFilters(
                        @Param("userId") Long userId,
                        @Param("keyword") String keyword,
                        @Param("startDate") LocalDateTime startDate,
                        @Param("endDate") LocalDateTime endDate,
                        Pageable pageable);

        long countByUserId(Long userId);

        @Query("SELECT r FROM RecognitionRecord r WHERE " +
                        "(:userId IS NULL OR r.userId = :userId) AND " +
                        "(:modelType IS NULL OR r.modelType = :modelType) " +
                        "ORDER BY r.createdAt DESC")
        Page<RecognitionRecord> findAllWithFilters(
                        @Param("userId") Long userId,
                        @Param("modelType") String modelType,
                        Pageable pageable);
}
